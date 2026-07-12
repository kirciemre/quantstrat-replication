"""
Diagnostic eval: run ONE episode and dump every timestep to a CSV so the full
trajectory (S_t, dS, inventory, action, pnl, cost, reward, cumulative) can be
inspected in a spreadsheet.

Columns:
  t        - timestep index
  S_t      - signal now
  S_next   - signal next step
  dS       - S_next - S_t (the move the reward is earned on)
  I_t      - inventory held coming in
  I_next   - action = new inventory (pi output)
  pnl      - I_next * dS
  cost     - lam * |I_next - I_t|
  reward   - pnl - cost
  cum      - running cumulative reward
"""

import csv
import numpy as np
import torch

from src.data.ou_simulator import simulate_path
from src.env.trading_env import step_reward, normalize_state_features
from src.models.ddpg import DDPG
from src.models.gru import GRUEncoder
from src.utils.config import load_config
from src.utils.ou_args import ou_args_from_cfg


cfg = load_config("configs/scenario3_hid.yaml")
kappa, sigma, ou_kw = ou_args_from_cfg(cfg)

rng = np.random.default_rng(0)          # fixed seed -> reproducible trajectory

encoder = GRUEncoder(cfg.d_h, cfg.d_l, enc_dim=cfg.enc_dim)
ddpg = DDPG(cfg.state_dim, cfg.action_dim, cfg.d_NN, cfg.l_NN, cfg.I_max, cfg.gamma, cfg.tau, cfg.lr)

ckpt = torch.load("artifacts/scenario3_hid.pt")
ddpg.actor.load_state_dict(ckpt["actor"])
encoder.load_state_dict(ckpt["encoder"])
ddpg.actor.eval(); encoder.eval()

OUT = "eval_trajectory_scenario3.csv"

with torch.no_grad(), open(OUT, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["t", "S_t", "S_next", "dS", "I_t", "I_next",
                     "pnl", "cost", "reward", "cum"])

    S, _ = simulate_path(cfg.n, rng, cfg.regimes, cfg.A, cfg.dt,
                         kappa=kappa, sigma=sigma, s0=1.0, **ou_kw)
    inventory = torch.zeros(1, 1)
    cum = 0.0

    for t in range(cfg.W, cfg.n - 1):
        window = torch.tensor(S[t-cfg.W:t+1], dtype=torch.float32).reshape(1, -1)
        S_t_col = torch.tensor([[S[t]]], dtype=torch.float32)

        o_t, _ = encoder(window)
        S_t_norm = normalize_state_features(S_t_col)
        state = torch.cat([S_t_norm, inventory, o_t], dim=1)

        action = ddpg.actor(state)                        # I_{t+1}
        r = step_reward(inventory, action, S[t], S[t+1], cfg.lam)

        dS = S[t+1] - S[t]
        pnl = action.item() * dS
        cost = cfg.lam * abs(action.item() - inventory.item())
        cum += r.item()

        writer.writerow([t, f"{S[t]:.6f}", f"{S[t+1]:.6f}", f"{dS:.6f}",
                         f"{inventory.item():.6f}", f"{action.item():.6f}",
                         f"{pnl:.6f}", f"{cost:.6f}", f"{r.item():.6f}", f"{cum:.6f}"])

        inventory = action

print(f"wrote {OUT}")
print(f"total reward (cum): {cum:.4f}")
# quick stats to eyeball without opening the file
dS_all = np.diff(S[cfg.W:cfg.n])
print(f"dS over episode: min={dS_all.min():.4f} max={dS_all.max():.4f} "
      f"std={dS_all.std():.4f} mean|dS|={np.abs(dS_all).mean():.4f}")
print(f"S range: [{S.min():.4f}, {S.max():.4f}]")