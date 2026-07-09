import numpy as np
import torch
import torch.nn as nn

from src.env.trading_env import sample_batch, step_reward
from src.models.gru import GRUEncoder
from src.models.ddpg import DDPG


# --- reproducibility (numpy drives the simulator; torch drives net init + noise) ---
seed = 0
rng = np.random.default_rng(seed)
torch.manual_seed(seed)

# --- OU / regime params (Section 2 model, theta-only) ---
regimes = np.array([0.9, 1.0, 1.1])
A = np.array([[-0.1, 0.05, 0.05],
              [0.05, -0.1, 0.05],
              [0.05, 0.05, -0.1]])
kappa = 5
sigma = 0.2
dt = 0.2

# --- environment ---
W = 10              # look-back window length
I_max = 10          # inventory bound
lam = 0.05          # transaction cost (Eq. 4)
batch_size = 512    # Table 1

# --- network dims ---
d_h = 10                    # GRU hidden size (Table 3)
d_l = 1                     # GRU layers
hidden_dim = 20             # Actor/Critic width
n_layers = 4                # Actor/Critic hidden blocks
action_dim = 1
state_dim = 1 + 1 + d_h     # S_t, I_t, o_t  -> derive from d_h, never hard-code 12

# --- DDPG ---
gamma = 0.999
tau = 0.001
lr = 0.001

# --- training ---
N = 10000           # iterations (Table 1)
log_every = 500

# --- epsilon schedule: eps = max(eps_a / (eps_a + m), eps_min) ---
eps_a = 100
eps_min = 0.01


ddpg = DDPG(state_dim, action_dim, hidden_dim, n_layers, I_max, gamma, tau, lr)
encoder = GRUEncoder(hidden_size=d_h, num_layers=d_l)

head = nn.Linear(d_h, 1)
gru_opt = torch.optim.Adam(list(encoder.parameters()) + list(head.parameters()), lr=lr)

for m in range(N):
    epsilon = max(eps_a / (eps_a + m), eps_min)

    batch = sample_batch(batch_size, W, rng, regimes, A, kappa, sigma, dt, I_max)

    windows_tensor = torch.tensor(batch["windows"], dtype=torch.float32)
    S_t_col = torch.tensor(batch["S_t"], dtype=torch.float32).reshape(-1,1)
    I_t_col = torch.tensor(batch["I_t"], dtype=torch.float32).reshape(-1,1)
    next_windows_tensor = torch.tensor(batch["next_windows"], dtype=torch.float32)
    S_next_col = torch.tensor(batch["S_next"], dtype=torch.float32).reshape(-1, 1)

    # --- GRU auxiliary update: predict S_next from the window ---
    o_t = encoder(windows_tensor)
    pred = head(o_t)
    loss = torch.nn.functional.mse_loss(pred, S_next_col)

    gru_opt.zero_grad()
    loss.backward()
    gru_opt.step()
    o_t = o_t.detach()

    # --- assemble current state ---
    state = torch.cat([S_t_col, I_t_col, o_t], dim=1)
    
    # --- action taken (with exploration noise), as a constant for the critic ---
    action = ddpg.select_action(state, epsilon)

    # --- reward (ground truth, Eq. 4) ---
    reward = step_reward(I_t_col, action, S_t_col, S_next_col, lam)

    # --- assemble next state ---
    o_t_next = encoder(next_windows_tensor).detach()
    next_state = torch.cat([S_next_col, action, o_t_next], dim=1)

    # --- learn ---
    critic_loss = ddpg.update_critic(state, action, reward, next_state)
    actor_loss = ddpg.update_actor(state)
    ddpg.update_targets()

    if m % log_every == 0:
        print(f"iter {m:5d} | critic {critic_loss:.4f} | actor {actor_loss:.4f} | eps {epsilon:.3f}")

torch.save({"actor": ddpg.actor.state_dict(), "critic": ddpg.critic.state_dict(), "encoder": encoder.state_dict()}, "artifacts/hid_ddpg.pt")
