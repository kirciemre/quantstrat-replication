"""
Scenario 3 trajectory checker — single-episode rollout with full regime tracking.

Generates:
  1. CSV: step, S_t, I_t, action, pnl, cost, reward, cum_reward,
          theta_t, kappa_t, sigma_t, theta_regime, kappa_regime, sigma_regime
  2. Multi-panel plot: signal + theta-regime background (top), inventory (middle),
     cumulative reward (bottom).

Why manual OU simulation: simulate_path() only returns theta regime indices.
This diagnostic needs ALL three regime chains (theta, kappa, sigma) tracked
explicitly step-by-step, so we reproduce the OU loop inline.

Usage:
    python -m src.eval.s3_trajectory_check
"""

import csv
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.linalg import expm

from src.env.trading_env import step_reward, normalize_state_features
from src.models.ddpg import DDPG
from src.models.gru import GRUEncoder
from src.utils.config import load_config
from src.utils.ou_args import ou_args_from_cfg


# ---------------------------------------------------------------------------
# 1.  load config + route OU params via the shared helper
# ---------------------------------------------------------------------------
cfg = load_config("configs/scenario3_hid.yaml")
kappa, sigma, ou_kw = ou_args_from_cfg(cfg)

seed = cfg.train_seed if cfg.train_seed is not None else 42
rng = np.random.default_rng(seed)
torch.manual_seed(seed)

# --- networks (same pipeline as scenario3_hid_eval.py) ---
encoder = GRUEncoder(cfg.d_h, cfg.d_l, enc_dim=cfg.enc_dim)
ddpg = DDPG(cfg.state_dim, cfg.action_dim, cfg.d_NN, cfg.l_NN,
            cfg.I_max, cfg.gamma, cfg.tau, cfg.lr)

ckpt = torch.load("artifacts/scenario3_hid.pt", weights_only=True)
ddpg.actor.load_state_dict(ckpt["actor"])
ddpg.critic.load_state_dict(ckpt["critic"])
encoder.load_state_dict(ckpt["encoder"])
ddpg.actor.eval(); ddpg.critic.eval(); encoder.eval()

# ---------------------------------------------------------------------------
# 2.  simulate ONE path — track ALL three regime chains explicitly
# ---------------------------------------------------------------------------
# simulate_path() only exposes theta regimes, so we reproduce the OU loop
# here to capture kappa and sigma regime tracks for diagnostic inspection.
n_steps = cfg.n

P_theta = expm(cfg.A * cfg.dt)
P_kappa = expm(ou_kw["A_kappa"] * cfg.dt)
P_sigma = expm(ou_kw["A_sigma"] * cfg.dt)

S = np.zeros(n_steps)
theta_val = np.zeros(n_steps)
kappa_val = np.zeros(n_steps)
sigma_val = np.zeros(n_steps)
theta_idx = np.zeros(n_steps, dtype=int)
kappa_idx = np.zeros(n_steps, dtype=int)
sigma_idx = np.zeros(n_steps, dtype=int)

# --- initialise at t=0 ---
theta_row = rng.integers(len(cfg.regimes))
kappa_row = rng.integers(len(ou_kw["regimes_kappa"]))
sigma_row = rng.integers(len(ou_kw["regimes_sigma"]))

theta_idx[0] = theta_row; kappa_idx[0] = kappa_row; sigma_idx[0] = sigma_row
theta_val[0] = cfg.regimes[theta_row]
kappa_val[0] = ou_kw["regimes_kappa"][kappa_row]
sigma_val[0] = ou_kw["regimes_sigma"][sigma_row]

# S_0 = 1.0 (paper's test-phase convention)
S[0] = 1.0

for t in range(1, n_steps):
    # advance each Markov chain independently
    theta_row = rng.choice(len(cfg.regimes), p=P_theta[theta_row])
    kappa_row = rng.choice(len(ou_kw["regimes_kappa"]), p=P_kappa[kappa_row])
    sigma_row = rng.choice(len(ou_kw["regimes_sigma"]), p=P_sigma[sigma_row])

    theta_idx[t] = theta_row; kappa_idx[t] = kappa_row; sigma_idx[t] = sigma_row
    theta_val[t] = cfg.regimes[theta_row]
    kappa_val[t] = ou_kw["regimes_kappa"][kappa_row]
    sigma_val[t] = ou_kw["regimes_sigma"][sigma_row]

    # OU discretisation (same as ou_simulator.py)
    kt = kappa_val[t]; st = sigma_val[t]
    a = np.exp(-kt * cfg.dt)
    b = st * np.sqrt((1.0 - np.exp(-2.0 * kt * cfg.dt)) / (2.0 * kt))
    S[t] = theta_val[t] + (S[t-1] - theta_val[t]) * a + b * rng.standard_normal()

# ---------------------------------------------------------------------------
# 3.  rollout policy (same pipeline as scenario3_hid_eval.py)
# ---------------------------------------------------------------------------
rows = []
inventory = torch.zeros(1, 1)
cum_reward = 0.0

with torch.no_grad():
    for t in range(cfg.W, n_steps - 1):
        window = torch.tensor(S[t-cfg.W:t+1], dtype=torch.float32).reshape(1, -1)
        S_t_col = torch.tensor([[S[t]]], dtype=torch.float32)

        o_t, _ = encoder(window)                          # unpack: encoding head only
        S_t_norm = normalize_state_features(S_t_col)
        state = torch.cat([S_t_norm, inventory, o_t], dim=1)

        action = ddpg.actor(state)                        # I_{t+1} (deterministic, no noise)
        r = step_reward(inventory, action, S[t], S[t+1], cfg.lam)
        cum_reward += r.item()

        # P&L decomposition (same as eval_to_csv.py)
        dS = S[t+1] - S[t]
        pnl = action.item() * dS
        cost = cfg.lam * abs(action.item() - inventory.item())

        rows.append({
            "step": t,
            "S_t": S[t],
            "I_t": inventory.item(),
            "action": action.item(),                             # = I_{t+1}
            "pnl": pnl,
            "cost": cost,
            "reward": r.item(),
            "cum_reward": cum_reward,
            "theta_t": theta_val[t],
            "kappa_t": kappa_val[t],
            "sigma_t": sigma_val[t],
            "theta_regime": theta_idx[t],
            "kappa_regime": kappa_idx[t],
            "sigma_regime": sigma_idx[t],
        })

        inventory = action                                # roll forward: new inventory = action

# ---------------------------------------------------------------------------
# 4.  write CSV
# ---------------------------------------------------------------------------
out_dir = "plots"
os.makedirs(out_dir, exist_ok=True)
csv_path = os.path.join(out_dir, "trajectory_scenario3.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
print(f"CSV saved to {csv_path}  ({len(rows)} rows)")

# ---------------------------------------------------------------------------
# 5.  trajectory plot
# ---------------------------------------------------------------------------
steps = np.array([r["step"] for r in rows])
S_arr = np.array([r["S_t"] for r in rows])
I_arr = np.array([r["I_t"] for r in rows])
cum_arr = np.array([r["cum_reward"] for r in rows])
theta_arr = np.array([r["theta_t"] for r in rows])
kappa_arr = np.array([r["kappa_t"] for r in rows])
sigma_arr = np.array([r["sigma_t"] for r in rows])

fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

# --- panel 1: signal S_t + theta regime background ---
ax1 = axes[0]
ax1.plot(steps, S_arr, linewidth=0.6, color="black", label=r"$S_t$")
for regime_id, color in enumerate(["#e8f0fe", "#fce8e6", "#e6f4ea"]):
    mask = theta_arr == cfg.regimes[regime_id]
    if mask.any():
        ax1.fill_between(steps, S_arr.min()-0.05, S_arr.max()+0.05,
                         where=mask, color=color, alpha=0.4,
                         label=f"$\\theta$={cfg.regimes[regime_id]}")
ax1.axhline(y=1.0, color="gray", ls="--", lw=0.8, label="$S=1.0$")
ax1.set_ylabel("signal $S_t$")
ax1.legend(loc="upper right", ncol=4, fontsize=7)
ax1.set_title("Scenario 3 — trajectory check (theta + kappa + sigma switching)")

# --- panel 2: inventory ---
ax2 = axes[1]
ax2.fill_between(steps, 0, I_arr, alpha=0.25, color="steelblue")
ax2.plot(steps, I_arr, linewidth=0.6, color="steelblue", label=r"$I_t$")
ax2.axhline(y=0, color="gray", ls="--", lw=0.6)
ax2.set_ylabel("inventory $I_t$")
ax2.legend(loc="upper right", fontsize=7)

# --- panel 3: cumulative reward ---
ax3 = axes[2]
ax3.plot(steps, cum_arr, linewidth=0.8, color="darkgreen", label="cumulative reward")
ax3.axhline(y=0, color="gray", ls="--", lw=0.6)
ax3.set_xlabel("step")
ax3.set_ylabel("cum. reward")
ax3.legend(loc="upper left", fontsize=7)

plt.tight_layout()
plot_path = os.path.join(out_dir, "trajectory_scenario3.png")
plt.savefig(plot_path, dpi=150)
plt.close()
print(f"Plot saved to {plot_path}")

# ---------------------------------------------------------------------------
# 6.  summary statistics
# ---------------------------------------------------------------------------
print(f"\nFinal cum. reward: {cum_arr[-1]:.2f}")
print(f"Mean |I_t|:          {np.abs(I_arr).mean():.3f}")
print(f"Max |I_t|:           {np.abs(I_arr).max():.3f}")
print(f"Theta switches:       {(np.diff(theta_arr) != 0).sum()}")
print(f"Kappa switches:       {(np.diff(kappa_arr) != 0).sum()}")
print(f"Sigma switches:       {(np.diff(sigma_arr) != 0).sum()}")
