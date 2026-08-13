"""
H1: does STABILIZING training make reg-DDPG converge to the paper's low-reward,
low-variance operating point?

The convergence check showed reg-DDPG overshoots (~15 at iter 2000) then wanders to
scattered endpoints (2.5-12.3) -> huge cross-seed variance. Seed 2 happened to settle
near the paper (2.55, cautious |I|). Hypothesis: the paper's point is reachable but
we only hit it inconsistently; a more stable training recipe should make ALL seeds
converge there, collapsing the variance toward the paper's std (0.30).

Combined stabilization (first pass -- attribution later if it works):
  - actor LR x0.1 (actor stops racing ahead of the critic)
  - l = 1 actor update per iter (paper baseline uses 5)
  - gradient clipping (max-norm) on both actor and critic

Everything else identical to the baseline sweep (same frozen regressor, same eval
set). Reports per-seed final reward (M=500) + a mid-training eval at iter 2000 to
check whether the overshoot is gone. Compare std to paper 2.95 +/- 0.30 and to the
baseline 4.26 +/- 4.11.

Usage: python -m src.eval.stabilize_h1 --seeds 0,1,2,3,4
"""

import argparse
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from src.env.trading_env import sample_batch, step_reward
from src.models.ddpg import DDPG
from src.utils.config import load_config
from src.utils.ou_args import ou_args_from_cfg
from src.eval.sweep_reg import pretrain_regressor, evaluate, build_state

SCEN = 2
ACTOR_LR_SCALE = 0.1
L_ACTOR = 1
GRAD_CLIP = 1.0


def train_stable(cfg, kappa, sigma, ou_kw, reg, seed, N):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    ddpg = DDPG(cfg.state_dim, cfg.action_dim, cfg.d_NN, cfg.l_NN,
                cfg.I_max, cfg.gamma, cfg.tau, cfg.lr)
    for g in ddpg.actor_opt.param_groups:      # lower ONLY the actor LR
        g["lr"] = cfg.lr * ACTOR_LR_SCALE
    epsilon = 1.0
    mid = None
    for m in range(N):
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

        # --- critic update(s) with grad clipping ---
        for _ in range(cfg.ell):
            with torch.no_grad():
                na = ddpg.actor_target(next_state)
                q_next = ddpg.critic_target(next_state, na)
                target = reward + ddpg.gamma * q_next
            q_pred = ddpg.critic(state, action)
            closs = torch.nn.functional.mse_loss(q_pred, target)
            ddpg.critic_opt.zero_grad(); closs.backward()
            clip_grad_norm_(ddpg.critic.parameters(), GRAD_CLIP)
            ddpg.critic_opt.step()
            ddpg.soft_update(ddpg.critic_target, ddpg.critic)

        # --- actor update(s): only L_ACTOR per iter, clipped ---
        for _ in range(L_ACTOR):
            a = ddpg.actor(state)
            aloss = -ddpg.critic(state, a).mean()
            ddpg.actor_opt.zero_grad(); aloss.backward()
            clip_grad_norm_(ddpg.actor.parameters(), GRAD_CLIP)
            ddpg.actor_opt.step()
            ddpg.soft_update(ddpg.actor_target, ddpg.actor)

        epsilon = max(cfg.eps_a / (cfg.eps_a + (m + 1)), cfg.eps_min)
        if m == 2000:
            r, _ = evaluate(cfg, kappa, sigma, ou_kw, reg, ddpg, 100)
            mid = float(r.mean())
    ddpg.actor.eval()
    return ddpg, mid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4")
    ap.add_argument("--N", type=int, default=None)
    ap.add_argument("--M", type=int, default=None)
    ap.add_argument("--pretrain_steps", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(f"configs/scenario{SCEN}_reg.yaml")
    kappa, sigma, ou_kw = ou_args_from_cfg(cfg)
    seeds = [int(s) for s in args.seeds.split(",")]
    N = args.N or cfg.N
    M = args.M or cfg.M

    print(f"=== H1 stabilized reg-DDPG — scenario {SCEN} ===", flush=True)
    print(f"actor_lr x{ACTOR_LR_SCALE}  l={L_ACTOR} (baseline 5)  grad_clip={GRAD_CLIP}", flush=True)
    print(f"seeds={seeds}  N={N}  M={M}", flush=True)
    print("baseline 4.26 +/- 4.11  |  paper 2.95 +/- 0.30", flush=True)

    reg = pretrain_regressor(cfg, kappa, sigma, ou_kw,
                             args.pretrain_steps or cfg.reg_pretrain_steps)
    print("[regressor pretrained]", flush=True)

    means = []
    for seed in seeds:
        ddpg, mid = train_stable(cfg, kappa, sigma, ou_kw, reg, seed, N)
        rewards, mean_absI = evaluate(cfg, kappa, sigma, ou_kw, reg, ddpg, M)
        means.append(float(rewards.mean()))
        np.save(f"artifacts/sweep_s{SCEN}_stableH1_seed{seed}.npy", rewards)
        mid_s = f"{mid:6.2f}" if mid is not None else "  n/a "
        print(f"  seed {seed:2d} | final {rewards.mean():6.2f} +/- {rewards.std():4.2f} "
              f"| mean|I| {mean_absI:4.2f} | iter2000 {mid_s} (overshoot check)", flush=True)

    means = np.array(means)
    print(f"--- H1: mean-of-means {means.mean():.2f} +/- {means.std():.2f} across "
          f"{len(seeds)} seeds (baseline 4.26 +/- 4.11, paper 2.95 +/- 0.30) ---", flush=True)
    print(f"    variance collapsed? baseline std 4.11 -> H1 std {means.std():.2f}", flush=True)


if __name__ == "__main__":
    main()
