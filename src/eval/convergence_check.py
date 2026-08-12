"""
Convergence diagnostic for reg-DDPG (answers "did training actually converge?").

Instruments the training loop: records actor/critic loss every LOG_EVERY iters and
runs a small eval (reward + mean|I|) every EVAL_EVERY iters, for several seeds on
one scenario. Plots four learning curves vs iteration:

  1. eval reward      -> does it plateau, or keep drifting?
  2. eval mean|I|     -> does the policy get MORE aggressive over training? (drift)
  3. actor loss (=-meanQ) -> Q magnitude; unbounded growth = overestimation/divergence
  4. critic loss      -> does the Bellman fit settle?

If reward keeps climbing and |I| keeps growing, training has NOT converged -- the
end-of-training snapshot is a moving target (relevant to the seed-variance and
reward-inflation findings). If curves flatten, it converged (to a higher-than-paper
value), which is a cleaner story.

Usage: python -m src.eval.convergence_check
"""

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.env.trading_env import sample_batch, step_reward
from src.models.ddpg import DDPG
from src.utils.config import load_config
from src.utils.ou_args import ou_args_from_cfg
from src.eval.sweep_reg import pretrain_regressor, evaluate, build_state

SCEN = 2
SEEDS = [0, 1, 2]
LOG_EVERY = 50       # record losses this often
EVAL_EVERY = 1000    # run a (small) eval this often
M_EVAL = 40          # episodes per mid-training eval (vectorized, cheap)

cfg = load_config(f"configs/scenario{SCEN}_reg.yaml")
kappa, sigma, ou_kw = ou_args_from_cfg(cfg)

print("pretraining regressor (shared across seeds)...", flush=True)
reg = pretrain_regressor(cfg, kappa, sigma, ou_kw, cfg.reg_pretrain_steps)

hist = {}   # seed -> dict of curves
for seed in SEEDS:
    print(f"seed {seed}: training with instrumentation...", flush=True)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    ddpg = DDPG(cfg.state_dim, cfg.action_dim, cfg.d_NN, cfg.l_NN,
                cfg.I_max, cfg.gamma, cfg.tau, cfg.lr)
    epsilon = 1.0
    loss_it, aL, cL = [], [], []
    ev_it, ev_r, ev_I = [], [], []
    last_a, last_c = np.nan, np.nan
    for m in range(cfg.N):
        batch = sample_batch(cfg.batch_size, cfg.W, rng, cfg.regimes, cfg.A,
                             kappa, sigma, cfg.dt, cfg.I_max, **ou_kw)
        windows = torch.tensor(batch["windows"], dtype=torch.float32)
        next_windows = torch.tensor(batch["next_windows"], dtype=torch.float32)
        S_t = torch.tensor(batch["S_t"], dtype=torch.float32).reshape(-1, 1)
        I_t = torch.tensor(batch["I_t"], dtype=torch.float32).reshape(-1, 1)
        S_next = torch.tensor(batch["S_next"], dtype=torch.float32).reshape(-1, 1)
        with torch.no_grad():
            pred_t = reg(windows); pred_next = reg(next_windows)
        state = build_state(S_t, I_t, pred_t, cfg.I_max, False)
        action = ddpg.select_action(state, epsilon)
        reward = step_reward(I_t, action, S_t, S_next, cfg.lam)
        next_state = build_state(S_next, action, pred_next, cfg.I_max, False)
        for _ in range(cfg.ell):
            last_c = ddpg.update_critic(state, action, reward, next_state)
            ddpg.soft_update(ddpg.critic_target, ddpg.critic)
        for _ in range(cfg.l):
            last_a = ddpg.update_actor(state)
            ddpg.soft_update(ddpg.actor_target, ddpg.actor)
        epsilon = max(cfg.eps_a / (cfg.eps_a + (m + 1)), cfg.eps_min)

        if m % LOG_EVERY == 0:
            loss_it.append(m); aL.append(last_a); cL.append(last_c)
        if m % EVAL_EVERY == 0 or m == cfg.N - 1:
            r, mi = evaluate(cfg, kappa, sigma, ou_kw, reg, ddpg, M_EVAL)
            ev_it.append(m); ev_r.append(float(r.mean())); ev_I.append(mi)
            print(f"  iter {m:5d} | eval R {r.mean():6.2f} | mean|I| {mi:4.2f} "
                  f"| actorL {last_a:+.3f} | criticL {last_c:.3f}", flush=True)
    hist[seed] = dict(loss_it=loss_it, aL=aL, cL=cL, ev_it=ev_it, ev_r=ev_r, ev_I=ev_I)

# --- plot ---
colors = ["#2a78d6", "#eb6834", "#1baf7a"]
fig, ax = plt.subplots(2, 2, figsize=(13, 8))
for i, seed in enumerate(SEEDS):
    h = hist[seed]; c = colors[i % len(colors)]
    ax[0, 0].plot(h["ev_it"], h["ev_r"], "-o", color=c, ms=4, label=f"seed {seed}")
    ax[0, 1].plot(h["ev_it"], h["ev_I"], "-o", color=c, ms=4, label=f"seed {seed}")
    ax[1, 0].plot(h["loss_it"], h["aL"], color=c, alpha=0.85, label=f"seed {seed}")
    ax[1, 1].plot(h["loss_it"], h["cL"], color=c, alpha=0.85, label=f"seed {seed}")
ax[0, 0].set_title("eval reward vs iteration  (plateau?)")
ax[0, 0].axhline(2.95, color="black", ls="--", lw=1, label="paper 2.95")
ax[0, 1].set_title("eval mean|I| vs iteration  (policy drifting more aggressive?)")
ax[1, 0].set_title("actor loss (= -mean Q) vs iteration  (Q diverging?)")
ax[1, 1].set_title("critic loss vs iteration  (Bellman fit settling?)")
for a in ax.ravel():
    a.set_xlabel("training iteration"); a.grid(alpha=0.25); a.legend(fontsize=8)
fig.suptitle(f"reg-DDPG convergence check — scenario {SCEN} (θ,κ), N={cfg.N}", fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.97])
import os
os.makedirs("figures", exist_ok=True)
plt.savefig("figures/convergence_s2_reg.png", dpi=150, bbox_inches="tight")
print("saved figures/convergence_s2_reg.png", flush=True)
