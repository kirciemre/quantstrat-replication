import numpy as np
import torch

from src.env.trading_env import sample_batch
from src.models.gru import GRUEncoder
from src.models.actor import Actor
from src.models.critic import Critic


if __name__ == "__main__":
    batch = sample_batch(batch_size=8, W=10, rng=np.random.default_rng(0), regimes=np.array([0.9, 1.0, 1.1]), 
                         A=np.array([[-0.1, 0.05, 0.05], [0.05, -0.1, 0.05], [0.05, 0.05, -0.1]]), kappa=5, sigma=0.2, dt=0.2, I_max=10)
    
    windows_tensor = torch.tensor(batch["windows"], dtype=torch.float32)
    S_t_tensor = torch.tensor(batch["S_t"], dtype=torch.float32)
    I_t_tensor = torch.tensor(batch["I_t"], dtype=torch.float32)

    encoder = GRUEncoder(hidden_size=10, num_layers=1)
    o_t = encoder(windows_tensor)

    S_t_col = S_t_tensor.reshape(-1, 1)
    I_t_col = I_t_tensor.reshape(-1, 1)

    state = torch.cat([S_t_col, I_t_col, o_t], dim=1)

    actor = Actor(12, 20, 4, 10)
    critic = Critic(12, 1, 20, 4)

    action = actor(state)
    q = critic(state, action)

    print(f"o_t.shape == {o_t.shape}")
    print(f"state.shape == {state.shape}")
    print(f"action.shape == {action.shape}")
    print(f"q.shape == {q.shape}")

