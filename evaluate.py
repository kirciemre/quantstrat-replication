import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from ou_env import normalize_signal, normalize_inventory

device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

def generate_test_paths(env, batch_size, n, W):
    """
    Generate test paths with S_0 = 1.0 (at index W).
    """
    seq_len = W + n + 1
    
    # Fetch regimes from env
    theta_regimes = env.regimes_theta
    kappa_regimes = env.regimes_kappa if hasattr(env, "regimes_kappa") else np.array([env.const_kappa])
    sigma_regimes = env.regimes_sigma if hasattr(env, "regimes_sigma") else np.array([env.const_sigma])
    
    # Initialize states dynamically based on transition matrix shapes
    theta_indices = np.random.choice(env.P_theta.shape[0], size=batch_size)
    if env.scenario >= 2 and hasattr(env, "P_kappa"):
        kappa_indices = np.random.choice(env.P_kappa.shape[0], size=batch_size)
    else:
        kappa_indices = np.zeros(batch_size, dtype=int)
        
    if env.scenario == 3 and hasattr(env, "P_sigma"):
        sigma_indices = np.random.choice(env.P_sigma.shape[0], size=batch_size)
    else:
        sigma_indices = np.zeros(batch_size, dtype=int)
        
    S = np.zeros((batch_size, seq_len))
    theta_val_arr = np.zeros((batch_size, seq_len))
    theta_idx_arr = np.zeros((batch_size, seq_len), dtype=int)
    kappa_val_arr = np.zeros((batch_size, seq_len))
    sigma_val_arr = np.zeros((batch_size, seq_len))
    
    # Evolve from step -W to 0 (indices 0 to W)
    S[:, 0] = np.random.normal(env.mu_inv, env.init_std, size=batch_size)
    theta_val_arr[:, 0] = theta_regimes[theta_indices]
    theta_idx_arr[:, 0] = theta_indices
    kappa_val_arr[:, 0] = kappa_regimes[kappa_indices] if env.scenario >= 2 else env.const_kappa
    sigma_val_arr[:, 0] = sigma_regimes[sigma_indices] if env.scenario == 3 else env.const_sigma
    
    for t in range(1, W + 1):
        theta_indices = env.step_markov_chain(theta_indices, env.P_theta)
        if env.scenario >= 2 and hasattr(env, "P_kappa"):
            kappa_indices = env.step_markov_chain(kappa_indices, env.P_kappa)
        if env.scenario == 3 and hasattr(env, "P_sigma"):
            sigma_indices = env.step_markov_chain(sigma_indices, env.P_sigma)
            
        theta = theta_regimes[theta_indices]
        kappa = kappa_regimes[kappa_indices] if env.scenario >= 2 else env.const_kappa
        sigma = sigma_regimes[sigma_indices] if env.scenario == 3 else env.const_sigma
        
        theta_val_arr[:, t] = theta
        theta_idx_arr[:, t] = theta_indices
        kappa_val_arr[:, t] = kappa
        sigma_val_arr[:, t] = sigma
        
        prev_S = S[:, t-1]
        exp_term = np.exp(-kappa * env.dt)
        mean_t = prev_S * exp_term + theta * (1.0 - exp_term)
        std_t = sigma * np.sqrt((1.0 - np.exp(-2.0 * kappa * env.dt)) / (2.0 * kappa))
        S[:, t] = mean_t + std_t * np.random.normal(size=batch_size)
        
    # Overwrite S_0 (index W) with 1.0 exactly as per paper's testing specification
    S[:, W] = 1.0
    
    # Evolve from index W+1 to W+n (indices W+1 to seq_len-1)
    for t in range(W + 1, seq_len):
        theta_indices = env.step_markov_chain(theta_indices, env.P_theta)
        if env.scenario >= 2 and hasattr(env, "P_kappa"):
            kappa_indices = env.step_markov_chain(kappa_indices, env.P_kappa)
        if env.scenario == 3 and hasattr(env, "P_sigma"):
            sigma_indices = env.step_markov_chain(sigma_indices, env.P_sigma)
            
        theta = theta_regimes[theta_indices]
        kappa = kappa_regimes[kappa_indices] if env.scenario >= 2 else env.const_kappa
        sigma = sigma_regimes[sigma_indices] if env.scenario == 3 else env.const_sigma
        
        theta_val_arr[:, t] = theta
        theta_idx_arr[:, t] = theta_indices
        kappa_val_arr[:, t] = kappa
        sigma_val_arr[:, t] = sigma
        
        prev_S = S[:, t-1]
        exp_term = np.exp(-kappa * env.dt)
        mean_t = prev_S * exp_term + theta * (1.0 - exp_term)
        std_t = sigma * np.sqrt((1.0 - np.exp(-2.0 * kappa * env.dt)) / (2.0 * kappa))
        S[:, t] = mean_t + std_t * np.random.normal(size=batch_size)
        
    return S, theta_val_arr, theta_idx_arr, kappa_val_arr, sigma_val_arr

def evaluate_agent(env, actor, model_type="prob-DDPG", gru_model=None,
                   M=500, n=2000, W=10, lambda_cost=0.05):
    """
    Evaluates the trained DDPG agent over M episodes of n steps.
    """
    actor.to(device)
    actor.eval()
    if gru_model is not None:
        gru_model.to(device)
        gru_model.eval()
        
    # Generate M paths in parallel
    S, _, _, _, _ = generate_test_paths(env, M, n, W)
    
    # Normalize signal path
    S_norm = normalize_signal(S)
    
    # Initialize trading variables
    I = np.zeros(M)
    cum_rewards = np.zeros(M)
    
    S_norm_tensor = torch.tensor(S_norm, dtype=torch.float32, device=device)
    
    for t in range(n):
        k = W + t
        # Get history from k-W to k
        S_hist = S_norm_tensor[:, k-W : k+1].unsqueeze(-1)
        
        # Current normalized inventory and signal
        I_norm = normalize_inventory(I)
        I_norm_tensor = torch.tensor(I_norm, dtype=torch.float32, device=device).unsqueeze(-1)
        S_t_norm = S_norm_tensor[:, k].unsqueeze(-1)
        
        # representation
        with torch.no_grad():
            if model_type == "prob-DDPG":
                probs_t, _ = gru_model(S_hist)
                G_t = torch.cat([S_t_norm, I_norm_tensor, probs_t], dim=-1)
            elif model_type == "reg-DDPG":
                pred_t = gru_model(S_hist)
                G_t = torch.cat([S_t_norm, I_norm_tensor, pred_t], dim=-1)
            elif model_type == "hid-DDPG":
                # gru_model is the online GRU encoder here
                o_t, _ = gru_model(S_hist)
                G_t = torch.cat([S_t_norm, I_norm_tensor, o_t], dim=-1)
            
            # Predict action
            action = actor(G_t).cpu().numpy().squeeze(-1)
            
        I_next = np.clip(action, -10.0, 10.0)
        
        # Calculate step reward
        # r_t = I_{t+1} * (S_{t+1} - S_t) - lambda * |I_{t+1} - I_t|
        r_t = I_next * (S[:, k+1] - S[:, k]) - lambda_cost * np.abs(I_next - I)
        cum_rewards += r_t
        I = I_next
        
    mean_reward = np.mean(cum_rewards)
    std_reward = np.std(cum_rewards)
    
    return cum_rewards, mean_reward, std_reward

def plot_rewards_distribution(rewards_dict, scenario_name, save_dir="plots"):
    """
    Creates a histogram comparison plot of rewards matching Figure 5/6/7.
    """
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    
    colors = {"prob-DDPG": "blue", "hid-DDPG": "green", "reg-DDPG": "orange"}
    
    for model_name, rewards in rewards_dict.items():
        plt.hist(rewards, bins=50, alpha=0.5, label=f"{model_name} (Mean: {np.mean(rewards):.2f})", color=colors.get(model_name, None))
        plt.axvline(np.mean(rewards), color=colors.get(model_name, "black"), linestyle="dashed", linewidth=1.5)
        
    plt.title(f"Histogram of rewards for 500 episodes under {scenario_name}")
    plt.xlabel("Cumulative rewards")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_path = os.path.join(save_dir, f"rewards_{scenario_name.lower().replace(' ', '_')}.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved histogram to {plot_path}")
