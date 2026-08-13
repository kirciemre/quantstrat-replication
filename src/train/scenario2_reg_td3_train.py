"""
Scenario 2 reg-DDPG, but with TD3 as the RL engine instead of vanilla DDPG.

EXPERIMENTAL / diagnostic script -- not part of the paper replication. Purpose:
test whether TD3's twin-critic + delayed-update design fixes the instability
observed in scenario2_reg_train.py (actor loss climbing 0.09 -> 0.68 without
converging, test-time mean ~7x and std ~19x the paper's reported values).

Everything is IDENTICAL to scenario2_reg_train.py except the RL engine:
  - same config (configs/scenario2_reg.yaml)
  - same regressor pretraining (unchanged, frozen the same way)
  - same batch sampling / reward / state construction
  - ONLY the DDPG -> TD3 swap in the RL training loop

Loop shape note: DDPG's original loop does 1 critic update + l=5 actor updates
per iteration. TD3's whole point is the OPPOSITE emphasis -- the actor should
update LESS often than the critic, so the critics settle first. Here we do 1
critic update + 1 actor *attempt* per iteration; TD3.update_actor() internally
skips all but every policy_delay-th attempt, so the actor effectively updates
every `policy_delay` iterations while the critics update every iteration.
"""

import argparse

import numpy as np
import torch

from src.env.trading_env import sample_batch, step_reward, normalize_state_features
from src.models.regressor import GRURegressor
from src.models.td3 import TD3
from src.utils.config import load_config
from src.utils.ou_args import ou_args_from_cfg


parser = argparse.ArgumentParser()
parser.add_argument(
    "--run_id",
    type=str,
    default="0",
    help="Tag for this run (e.g. '1', '2', ...). Keeps parallel runs from "
         "overwriting each other's checkpoint / reward files.",
)
args = parser.parse_args()

CKPT_PATH = f"artifacts/scenario2_reg_td3_run{args.run_id}.pt"

cfg = load_config("configs/scenario2_reg.yaml")
kappa, sigma, ou_kw = ou_args_from_cfg(cfg)

# --- reproducibility (numpy drives the simulator; torch drives net init + noise) ---
seed = cfg.train_seed if cfg.train_seed is not None else np.random.randint(0, 1_000_000)
rng = np.random.default_rng(seed)
torch.manual_seed(seed)
print(f"run_id: {args.run_id}")
print(f"seed: {seed}")

epsilon = 1.0

# --- networks: encoder is OWNED HERE (trained separately from the RL step) ---
regressor = GRURegressor(
    hidden_dim=cfg.reg_hidden_dim,
    num_layers=cfg.reg_layers,
    ffn_layers=cfg.reg_ffn_layers,
    ffn_hidden=cfg.reg_ffn_hidden,
)

reg_opt = torch.optim.AdamW(
    regressor.parameters(),
    lr=cfg.lr,
    weight_decay=1e-5,
)

regressor.train()

print("algorithm: reg-TD3 (experimental)")
print("window length:", cfg.W)
print("starting regressor pretraining...")

for step in range(cfg.reg_pretrain_steps):
    batch = sample_batch(
        cfg.batch_size,
        cfg.W,
        rng,
        cfg.regimes,
        cfg.A,
        kappa,
        sigma,
        cfg.dt,
        cfg.I_max,
        **ou_kw,
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

# --- quick validation on a fresh batch ---
regressor.eval()

val_batch = sample_batch(
    cfg.batch_size, cfg.W, rng, cfg.regimes, cfg.A, kappa, sigma, cfg.dt, cfg.I_max, **ou_kw,
)
val_windows = torch.tensor(val_batch["windows"], dtype=torch.float32)
val_targets = torch.tensor(val_batch["S_next"], dtype=torch.float32).reshape(-1, 1)

with torch.no_grad():
    val_predictions = regressor(val_windows)
    val_loss = torch.nn.functional.mse_loss(val_predictions, val_targets)

print(f"validation MSE: {val_loss.item():.6f}")

# --- freeze the pretrained regressor ---
regressor.eval()
for parameter in regressor.parameters():
    parameter.requires_grad = False

print("regressor frozen")

# --- TD3 network (drop-in replacement for DDPG) ---
td3 = TD3(
    cfg.state_dim,
    cfg.action_dim,
    cfg.d_NN,
    cfg.l_NN,
    cfg.I_max,
    cfg.gamma,
    cfg.tau,
    cfg.lr,
    policy_delay=2,     # actor updates half as often as the critics
    policy_noise=0.2,   # smoothing noise std added to the target action
    noise_clip=0.5,     # clip range for that noise
)

epsilon = 1.0

print("starting reg-TD3 training...")

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

    # --- 1 critic update per iteration (both critic1 and critic2 trained together) ---
    critic1_loss, critic2_loss = td3.update_critic(state, action, reward, next_state)

    # --- 1 actor *attempt* per iteration; internally a no-op unless it's this
    #     iteration's turn (every policy_delay iterations) ---
    actor_loss = td3.update_actor(state)

    if m % cfg.log_every == 0:
        actor_display = f"{actor_loss:.4f}" if actor_loss is not None else "  --  "
        print(
            f"iter {m:5d} | critic1 {critic1_loss:.4f} | critic2 {critic2_loss:.4f} "
            f"| actor {actor_display} | eps {epsilon:.3f}"
        )

    epsilon = max(cfg.eps_a / (cfg.eps_a + (m + 1)), cfg.eps_min)

print("reg-TD3 training completed")

# --- save actor, both critics, and pretrained regressor ---
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
