"""
Scenario 2 reg-DDPG baseline, re-run with --run_id support.

This is scenario2_reg_train.py unchanged in every algorithmic respect --
only difference is --run_id, so we can run this 3x in parallel (different
random seeds) to measure how much DDPG's own result varies run-to-run,
before comparing against TD3. Saves to a DISTINCT path so the original
artifacts/scenario2_reg.pt (already evaluated at 20.01 +/- 5.70) is
untouched.
"""

import argparse

import numpy as np
import torch

from src.env.trading_env import sample_batch, step_reward, normalize_state_features
from src.models.regressor import GRURegressor
from src.models.ddpg import DDPG
from src.utils.config import load_config
from src.utils.ou_args import ou_args_from_cfg


parser = argparse.ArgumentParser()
parser.add_argument("--run_id", type=str, required=True,
                    help="Tag for this baseline re-run, e.g. 'b1', 'b2', 'b3'.")
args = parser.parse_args()

CKPT_PATH = f"artifacts/scenario2_reg_baseline_{args.run_id}.pt"

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

print("algorithm:", cfg.algorithm)
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

ddpg = DDPG(cfg.state_dim, cfg.action_dim, cfg.d_NN, cfg.l_NN, cfg.I_max, cfg.gamma, cfg.tau, cfg.lr)

epsilon = 1.0
print("starting reg-DDPG training...")

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

    action = ddpg.select_action(state, epsilon)
    reward = step_reward(I_t_col, action, S_t_col, S_next_col, cfg.lam)

    S_next_norm = normalize_state_features(S_next_col)
    next_state = torch.cat([S_next_norm, action, pred_next], dim=1)

    for _ in range(cfg.ell):
        critic_loss = ddpg.update_critic(state, action, reward, next_state)
        ddpg.soft_update(ddpg.critic_target, ddpg.critic)

    for _ in range(cfg.l):
        actor_loss = ddpg.update_actor(state)
        ddpg.soft_update(ddpg.actor_target, ddpg.actor)

    if m % cfg.log_every == 0:
        print(f"iter {m:5d} | critic {critic_loss:.4f} | actor {actor_loss:.4f} | eps {epsilon:.3f}")

    epsilon = max(cfg.eps_a / (cfg.eps_a + (m + 1)), cfg.eps_min)

print("reg-DDPG training completed")

import os
os.makedirs("artifacts", exist_ok=True)
torch.save(
    {
        "actor": ddpg.actor.state_dict(),
        "critic": ddpg.critic.state_dict(),
        "regressor": regressor.state_dict(),
        "train_seed": seed,
    },
    CKPT_PATH,
)
print(f"saved {CKPT_PATH}")
