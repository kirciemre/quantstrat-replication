"""
Scenario 2 reg-TD3 v2 -- clean ablation vs the DDPG baseline.

Unlike td3.py / scenario2_reg_td3_train.py (v1), this version keeps the
EXACT SAME update cadence as the DDPG baseline (cfg.ell critic updates +
cfg.l=5 actor updates per iteration, targets soft-updated every actor call).
The ONLY change relative to DDPG is the RL engine itself: twin critics with
min-Q target + target policy smoothing (TD3's core overestimation fix),
nothing about training tempo. This isolates whether twin-critic /
smoothing alone helps, without confounding it with actor update frequency
(v1's mistake).
"""

import argparse

import numpy as np
import torch

from src.env.trading_env import sample_batch, step_reward, normalize_state_features
from src.models.regressor import GRURegressor
from src.models.td3_v2 import TD3
from src.utils.config import load_config
from src.utils.ou_args import ou_args_from_cfg


parser = argparse.ArgumentParser()
parser.add_argument("--run_id", type=str, required=True,
                    help="Tag for this run, e.g. 't1', 't2', 't3'.")
args = parser.parse_args()

CKPT_PATH = f"artifacts/scenario2_reg_td3v2_{args.run_id}.pt"

cfg = load_config("configs/scenario2_reg.yaml")
kappa, sigma, ou_kw = ou_args_from_cfg(cfg)

seed = cfg.train_seed if cfg.train_seed is not None else np.random.randint(0, 1_000_000)
rng = np.random.default_rng(seed)
torch.manual_seed(seed)
print(f"run_id: {args.run_id}")
print(f"seed: {seed}")

epsilon = 1.0

regressor = GRURegressor(
    hidden_dim=cfg.reg_hidden_dim,
    num_layers=cfg.reg_layers,
    ffn_layers=cfg.reg_ffn_layers,
    ffn_hidden=cfg.reg_ffn_hidden,
)

reg_opt = torch.optim.AdamW(regressor.parameters(), lr=cfg.lr, weight_decay=1e-5)
regressor.train()

print("algorithm: reg-TD3-v2 (experimental, actor cadence matches DDPG baseline)")
print("window length:", cfg.W)
print("starting regressor pretraining...")

for step in range(cfg.reg_pretrain_steps):
    batch = sample_batch(
        cfg.batch_size, cfg.W, rng, cfg.regimes, cfg.A, kappa, sigma, cfg.dt, cfg.I_max, **ou_kw,
    )
    windows = torch.tensor(batch["windows"], dtype=torch.float32)
    targets = torch.tensor(batch["S_next"], dtype=torch.float32).reshape(-1, 1)
    predictions = regressor(windows)
    reg_loss = torch.nn.functional.mse_loss(predictions, targets)
    reg_opt.zero_grad()
    reg_loss.backward()
    reg_opt.step()
    if step % 500 == 0:
        print(f"pretrain {step:4d} | reg loss {reg_loss.item():.6f}")

print("regressor pretraining completed")

regressor.eval()
for parameter in regressor.parameters():
    parameter.requires_grad = False
print("regressor frozen")

td3 = TD3(
    cfg.state_dim, cfg.action_dim, cfg.d_NN, cfg.l_NN, cfg.I_max, cfg.gamma, cfg.tau, cfg.lr,
    policy_noise=0.2, noise_clip=0.5,
)

epsilon = 1.0
print("starting reg-TD3-v2 training...")

for m in range(cfg.N):
    batch = sample_batch(
        cfg.batch_size, cfg.W, rng, cfg.regimes, cfg.A, kappa, sigma, cfg.dt, cfg.I_max, **ou_kw,
    )
    windows_tensor = torch.tensor(batch["windows"], dtype=torch.float32)
    next_windows_tensor = torch.tensor(batch["next_windows"], dtype=torch.float32)
    S_t_col = torch.tensor(batch["S_t"], dtype=torch.float32).reshape(-1, 1)
    I_t_col = torch.tensor(batch["I_t"], dtype=torch.float32).reshape(-1, 1)
    S_next_col = torch.tensor(batch["S_next"], dtype=torch.float32).reshape(-1, 1)

    with torch.no_grad():
        pred_t = regressor(windows_tensor)
        pred_next = regressor(next_windows_tensor)

    S_t_norm = normalize_state_features(S_t_col)
    state = torch.cat([S_t_norm, I_t_col, pred_t], dim=1)

    action = td3.select_action(state, epsilon)
    reward = step_reward(I_t_col, action, S_t_col, S_next_col, cfg.lam)

    S_next_norm = normalize_state_features(S_next_col)
    next_state = torch.cat([S_next_norm, action, pred_next], dim=1)

    # --- ell critic updates per iteration, same as DDPG baseline ---
    for _ in range(cfg.ell):
        critic1_loss, critic2_loss = td3.update_critic(state, action, reward, next_state)

    # --- l=5 actor updates per iteration, same as DDPG baseline; targets
    #     soft-updated after EACH actor step (matching DDPG's per-call cadence) ---
    for _ in range(cfg.l):
        actor_loss = td3.update_actor(state)
        td3.soft_update(td3.actor_target, td3.actor)
        td3.soft_update(td3.critic1_target, td3.critic1)
        td3.soft_update(td3.critic2_target, td3.critic2)

    if m % cfg.log_every == 0:
        print(
            f"iter {m:5d} | critic1 {critic1_loss:.4f} | critic2 {critic2_loss:.4f} "
            f"| actor {actor_loss:.4f} | eps {epsilon:.3f}"
        )

    epsilon = max(cfg.eps_a / (cfg.eps_a + (m + 1)), cfg.eps_min)

print("reg-TD3-v2 training completed")

import os
os.makedirs("artifacts", exist_ok=True)
torch.save(
    {
        "actor": td3.actor.state_dict(),
        "critic1": td3.critic1.state_dict(),
        "critic2": td3.critic2.state_dict(),
        "regressor": regressor.state_dict(),
        "train_seed": seed,
    },
    CKPT_PATH,
)
print(f"saved {CKPT_PATH}")
