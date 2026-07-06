import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from ou_env import normalize_signal_tensor, normalize_inventory_tensor, normalize_signal, normalize_inventory
from gru_filters import GRUEncoder, GRUClassifier, GRURegressor
from ddpg import Actor, Critic, soft_update

# Set up device: MPS for Apple Silicon, CUDA for Nvidia, CPU otherwise
device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
print(f"Using device: {device}")

def train_prob_filter(env, classifier, epochs=10000, batch_size=512, W=10, lr=0.001):
    """
    Offline pre-training of the GRU classifier to predict the theta regime.
    """
    classifier.to(device)
    classifier.train()
    optimizer = optim.AdamW(classifier.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    
    print("Pre-training GRU Classifier...")
    for epoch in range(epochs):
        # Generate batch of signal history and parameters
        # sequence length is W + 1 (from t-W to t)
        S, theta_vals, theta_indices, _, _ = env.generate_paths(batch_size, W + 1)
        
        # Normalize signal history
        S_norm = normalize_signal(S)
        
        # Inputs: shape (batch_size, W + 1, 1)
        x = torch.tensor(S_norm, dtype=torch.float32, device=device).unsqueeze(-1)
        # Targets: regime index at the current step t (index -1)
        y = torch.tensor(theta_indices[:, -1], dtype=torch.long, device=device)
        
        optimizer.zero_grad()
        probs, logits = classifier(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 2000 == 0:
            print(f"Classifier Epoch {epoch+1}/{epochs} - Loss: {loss.item():.6f}")
            
    classifier.eval()
    print("Pre-training GRU Classifier completed.")
    return classifier

def train_reg_filter(env, regressor, epochs=10000, batch_size=512, W=50, lr=0.001):
    """
    Offline pre-training of the GRU regressor to predict the next normalized signal S_{t+1}.
    """
    regressor.to(device)
    regressor.train()
    optimizer = optim.AdamW(regressor.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    print("Pre-training GRU Regressor...")
    for epoch in range(epochs):
        # Generate batch of signal history
        # sequence length is W + 2 (from t-W to t+1)
        S, _, _, _, _ = env.generate_paths(batch_size, W + 2)
        
        # Normalize signal
        S_norm = normalize_signal(S)
        
        # Inputs: S_norm[:, :W+1] (from t-W to t)
        x = torch.tensor(S_norm[:, :W+1], dtype=torch.float32, device=device).unsqueeze(-1)
        # Targets: S_norm[:, W+1] (the value at t+1)
        y = torch.tensor(S_norm[:, W+1], dtype=torch.float32, device=device).unsqueeze(-1)
        
        optimizer.zero_grad()
        pred = regressor(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 2000 == 0:
            print(f"Regressor Epoch {epoch+1}/{epochs} - Loss: {loss.item():.6f}")
            
    regressor.eval()
    print("Pre-training GRU Regressor completed.")
    return regressor

def train_ddpg_agent(env, model_type="prob-DDPG", W=10, N=10000, batch_size=512,
                     lr=0.001, lambda_cost=0.05, gamma=0.99, tau=0.001,
                     l_act=5, ell_crit=1, a_noise=100.0, eps_min=0.01,
                     gru_model=None, layers_ddpg=5, hidden_ddpg=64):
    """
    Trains the DDPG Actor-Critic agent using Algorithm 1.
    model_type: 'hid-DDPG', 'prob-DDPG', or 'reg-DDPG'
    """
    # 1. Determine state dimension based on representation model
    if model_type == "prob-DDPG":
        # State: S_t (1) + I_t (1) + 3 regime probabilities = 5 features
        # (For real data, num_classes is 2, so state_dim is 4)
        num_classes = gru_model.ffn[-1].out_features
        state_dim = 2 + num_classes
    elif model_type == "reg-DDPG":
        # State: S_t (1) + I_t (1) + S_pred_{t+1} (1) = 3 features
        state_dim = 3
    elif model_type == "hid-DDPG":
        # State: S_t (1) + I_t (1) + GRU hidden state (10) = 12 features
        # Initialize GRU encoder online
        gru_hidden_dim = 10
        gru_layers = 2 if env.scenario == 3 else 1
        gru_encoder = GRUEncoder(input_dim=1, hidden_dim=gru_hidden_dim, num_layers=gru_layers).to(device)
        gru_optimizer = optim.AdamW(gru_encoder.parameters(), lr=lr, weight_decay=1e-5)
        gru_criterion = nn.MSELoss()
        state_dim = 2 + gru_hidden_dim
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
        
    # 2. Initialize Actor-Critic networks and targets
    actor = Actor(state_dim, num_layers=layers_ddpg, hidden_dim=hidden_ddpg).to(device)
    critic = Critic(state_dim, num_layers=layers_ddpg, hidden_dim=hidden_ddpg).to(device)
    
    target_actor = Actor(state_dim, num_layers=layers_ddpg, hidden_dim=hidden_ddpg).to(device)
    target_critic = Critic(state_dim, num_layers=layers_ddpg, hidden_dim=hidden_ddpg).to(device)
    
    target_actor.load_state_dict(actor.state_dict())
    target_critic.load_state_dict(critic.state_dict())
    
    actor_optimizer = optim.AdamW(actor.parameters(), lr=lr, weight_decay=1e-5)
    critic_optimizer = optim.AdamW(critic.parameters(), lr=lr, weight_decay=1e-5)
    
    print(f"Training {model_type} DDPG Agent...")
    
    for m in range(1, N + 1):
        # Decay exploration noise variance
        eps = max(a_noise / (a_noise + m), eps_min)
        
        # a) Generate batch of trajectories of length W + 2
        S, theta_vals, theta_indices, kappa_vals, sigma_vals = env.generate_paths(batch_size, W + 2)
        
        # Normalize signal paths
        S_norm = normalize_signal(S)
        
        # Generate random initial inventories I_t ~ U[-10, 10]
        I_t = np.random.uniform(-10.0, 10.0, size=batch_size)
        I_t_norm = normalize_inventory(I_t)
        
        # Prepare inputs to PyTorch
        S_norm_tensor = torch.tensor(S_norm, dtype=torch.float32, device=device)
        I_t_norm_tensor = torch.tensor(I_t_norm, dtype=torch.float32, device=device).unsqueeze(-1)
        
        # b) representation extraction
        if model_type == "prob-DDPG":
            # Pass first W+1 signals (t-W to t) to get regime probs at t
            x_t = S_norm_tensor[:, :W+1].unsqueeze(-1)
            with torch.no_grad():
                probs_t, _ = gru_model(x_t)
            
            # State G_t = (S_t_norm, I_t_norm, probs_t)
            S_t_norm = S_norm_tensor[:, W].unsqueeze(-1)
            G_t = torch.cat([S_t_norm, I_t_norm_tensor, probs_t], dim=-1)
            
            # Pass signals (t-W+1 to t+1) to get regime probs at t+1
            x_next = S_norm_tensor[:, 1:W+2].unsqueeze(-1)
            with torch.no_grad():
                probs_next, _ = gru_model(x_next)
                
        elif model_type == "reg-DDPG":
            x_t = S_norm_tensor[:, :W+1].unsqueeze(-1)
            with torch.no_grad():
                pred_t = gru_model(x_t)
                
            S_t_norm = S_norm_tensor[:, W].unsqueeze(-1)
            G_t = torch.cat([S_t_norm, I_t_norm_tensor, pred_t], dim=-1)
            
            x_next = S_norm_tensor[:, 1:W+2].unsqueeze(-1)
            with torch.no_grad():
                pred_next = gru_model(x_next)
                
        elif model_type == "hid-DDPG":
            # For hid-DDPG: Train the GRU encoder online using auxiliary head
            x_t = S_norm_tensor[:, :W+1].unsqueeze(-1)
            y_aux = S_norm_tensor[:, W+1].unsqueeze(-1)
            
            gru_encoder.train()
            gru_optimizer.zero_grad()
            o_t, pred_aux = gru_encoder(x_t)
            loss_gru = gru_criterion(pred_aux, y_aux)
            loss_gru.backward()
            gru_optimizer.step()
            gru_encoder.eval()
            
            # Extract features after update
            with torch.no_grad():
                o_t, _ = gru_encoder(x_t)
                
            S_t_norm = S_norm_tensor[:, W].unsqueeze(-1)
            G_t = torch.cat([S_t_norm, I_t_norm_tensor, o_t], dim=-1)
            
            # Get representation at t+1
            x_next = S_norm_tensor[:, 1:W+2].unsqueeze(-1)
            with torch.no_grad():
                o_next, _ = gru_encoder(x_next)
        
        # c) Select Action (inventory I_{t+1}) with exploration noise
        actor.eval()
        with torch.no_grad():
            a_t = actor(G_t) # shape (batch_size, 1)
        # Add exploration noise
        noise = torch.randn_like(a_t) * eps
        a_t = torch.clamp(a_t + noise, -10.0, 10.0)
        
        # d) Execute action and calculate rewards
        # Action is I_{t+1}. Current inventory is I_t.
        # r_t = I_{t+1} * (S_{t+1} - S_t) - lambda * |I_{t+1} - I_t|
        # S values are unnormalized!
        S_t = torch.tensor(S[:, W], dtype=torch.float32, device=device).unsqueeze(-1)
        S_next = torch.tensor(S[:, W+1], dtype=torch.float32, device=device).unsqueeze(-1)
        I_t_tensor = torch.tensor(I_t, dtype=torch.float32, device=device).unsqueeze(-1)
        
        reward = a_t * (S_next - S_t) - lambda_cost * torch.abs(a_t - I_t_tensor)
        
        # e) Construct next state representation G_{t+1}
        I_next_norm = normalize_inventory_tensor(a_t)
        S_next_norm = S_norm_tensor[:, W+1].unsqueeze(-1)
        
        if model_type == "prob-DDPG":
            G_next = torch.cat([S_next_norm, I_next_norm, probs_next], dim=-1)
        elif model_type == "reg-DDPG":
            G_next = torch.cat([S_next_norm, I_next_norm, pred_next], dim=-1)
        elif model_type == "hid-DDPG":
            G_next = torch.cat([S_next_norm, I_next_norm, o_next], dim=-1)
            
        # f) Update Critic
        critic.train()
        for _ in range(ell_crit):
            # Compute target Q
            target_actor.eval()
            target_critic.eval()
            with torch.no_grad():
                next_action = target_actor(G_next)
                target_Q = target_critic(G_next, next_action)
                y_target = reward + gamma * target_Q
                
            # Critic loss
            Q_val = critic(G_t, a_t)
            critic_loss = nn.MSELoss()(Q_val, y_target)
            
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()
            
            # Soft update of target critic
            soft_update(target_critic, critic, tau)
            
        # g) Update Actor
        actor.train()
        for _ in range(l_act):
            critic.eval()
            # Predict actions
            pred_actions = actor(G_t)
            # Minimize -Q(G_t, Actor(G_t))
            actor_loss = -critic(G_t, pred_actions).mean()
            
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()
            
            # Soft update of target actor
            soft_update(target_actor, actor, tau)
            
        if m % 2000 == 0:
            print(f"DDPG Step {m}/{N} - Actor Loss: {actor_loss.item():.4f}, Critic Loss: {critic_loss.item():.4f}")
            
    actor.eval()
    critic.eval()
    print(f"Training {model_type} DDPG Agent completed.")
    
    if model_type == "hid-DDPG":
        return actor, gru_encoder
    else:
        return actor
