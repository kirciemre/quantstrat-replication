"""
Multi-seed sweep for reg-DDPG (scenarios 1/2/3), to compare against paper Table 4.

Why: reg-DDPG in the switching-kappa/sigma scenarios is seed-sensitive (single-critic
overestimation -> the actor either over-trades or collapses). A single checkpoint is
not representative. This script fixes seeds and reports mean-of-means +/- std across
seeds, the fair way to compare with the paper's reported averages.

Design choices (documented so the numbers are interpretable):
  - The GRU regressor (the "filter") is pretrained ONCE per scenario with a fixed base
    seed and then frozen, and REUSED across all DDPG seeds. Regressor pretraining is the
    stable part; the variance we are studying lives in the DDPG stage. This isolates the
    DDPG seed variance and roughly halves compute.
  - Each DDPG seed gets its own np rng (drives sample_batch) and torch seed (net init +
    exploration noise) -- this is the variance source.
  - Evaluation uses ONE fixed test set (fixed eval seed) shared across all DDPG seeds, so
    cross-seed differences reflect the policy, not test-set noise. Eval is vectorized
    across the M episodes (one batched GRU forward per timestep) for speed.

Usage:
    python -m src.eval.sweep_reg --scenario 2 --seeds 0,1,2,3,4
    python -m src.eval.sweep_reg --scenario 2 --seeds 0,1 --pretrain_steps 500 --N 500 --M 50   # quick calib
"""

import argparse
import time

import numpy as np
import torch

from src.data.ou_simulator import simulate_path
from src.env.trading_env import sample_batch, step_reward, normalize_state_features
from src.models.regressor import GRURegressor


def build_state(S_col, I_col, pred_col, I_max, full_norm):
    """
    Assemble G_t = (S_t, I_t, pred_{t+1}) for the networks.

    Default (full_norm=False): only S_t normalized to [0,1]; inventory and the
    prediction stay raw (the repo's documented hid-DDPG-tuned choice).

    full_norm=True: ALL three features normalized to [0,1] (the paper's "features
    normalised in [0,1]"). Prediction is on the signal scale, so it uses the same
    S-bounds as S_t; inventory maps [-I_max, I_max] -> [0,1]. Reward is ALWAYS
    computed from raw quantities by the caller -- this only rescales network input.
    """
    S_n = normalize_state_features(S_col)
    if full_norm:
        I_n = ((I_col + I_max) / (2 * I_max)).clamp(0.0, 1.0)
        pred_n = normalize_state_features(pred_col)   # prediction ~ signal scale
        return torch.cat([S_n, I_n, pred_n], dim=1)
    return torch.cat([S_n, I_col, pred_col], dim=1)
from src.models.ddpg import DDPG
from src.utils.config import load_config
from src.utils.ou_args import ou_args_from_cfg


def pretrain_regressor(cfg, kappa, sigma, ou_kw, steps, base_seed=12345):
    """Pretrain + freeze the GRU regressor once (shared across DDPG seeds)."""
    rng = np.random.default_rng(base_seed)
    torch.manual_seed(base_seed)
    reg = GRURegressor(cfg.reg_hidden_dim, cfg.reg_layers, cfg.reg_ffn_layers, cfg.reg_ffn_hidden)
    opt = torch.optim.AdamW(reg.parameters(), lr=cfg.lr, weight_decay=1e-5)
    reg.train()
    for step in range(steps):
        batch = sample_batch(cfg.batch_size, cfg.W, rng, cfg.regimes, cfg.A,
                             kappa, sigma, cfg.dt, cfg.I_max, **ou_kw)
        windows = torch.tensor(batch["windows"], dtype=torch.float32)
        targets = torch.tensor(batch["S_next"], dtype=torch.float32).reshape(-1, 1)
        loss = torch.nn.functional.mse_loss(reg(windows), targets)
        opt.zero_grad(); loss.backward(); opt.step()
    reg.eval()
    for p in reg.parameters():
        p.requires_grad = False
    return reg


def train_ddpg(cfg, kappa, sigma, ou_kw, reg, seed, N, full_norm=False, lr_sched=None):
    """Train a fresh DDPG with the frozen regressor, for one seed.

    lr_sched: None (fixed lr), 'cosine' (CosineAnnealingLR lr->~0 over N), or
    'step' (StepLR x0.5 every N/4). The paper mentions "W-ADAM with a scheduler"
    but not which; this tests whether a decaying LR undertrains the actor into the
    paper's cautious, lower-reward policy. Applied to the DDPG optimizers only
    (the regressor is already frozen).
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    ddpg = DDPG(cfg.state_dim, cfg.action_dim, cfg.d_NN, cfg.l_NN,
                cfg.I_max, cfg.gamma, cfg.tau, cfg.lr)
    scheds = []
    if lr_sched == "cosine":
        scheds = [torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=N, eta_min=cfg.lr * 0.01)
                  for o in (ddpg.actor_opt, ddpg.critic_opt)]
    elif lr_sched == "step":
        scheds = [torch.optim.lr_scheduler.StepLR(o, step_size=max(1, N // 4), gamma=0.5)
                  for o in (ddpg.actor_opt, ddpg.critic_opt)]
    epsilon = 1.0
    for m in range(N):
        batch = sample_batch(cfg.batch_size, cfg.W, rng, cfg.regimes, cfg.A,
                             kappa, sigma, cfg.dt, cfg.I_max, **ou_kw)
        windows = torch.tensor(batch["windows"], dtype=torch.float32)
        next_windows = torch.tensor(batch["next_windows"], dtype=torch.float32)
        S_t = torch.tensor(batch["S_t"], dtype=torch.float32).reshape(-1, 1)
        I_t = torch.tensor(batch["I_t"], dtype=torch.float32).reshape(-1, 1)
        S_next = torch.tensor(batch["S_next"], dtype=torch.float32).reshape(-1, 1)
        with torch.no_grad():
            pred_t = reg(windows)
            pred_next = reg(next_windows)
        state = build_state(S_t, I_t, pred_t, cfg.I_max, full_norm)
        action = ddpg.select_action(state, epsilon)
        reward = step_reward(I_t, action, S_t, S_next, cfg.lam)
        next_state = build_state(S_next, action, pred_next, cfg.I_max, full_norm)
        for _ in range(cfg.ell):
            ddpg.update_critic(state, action, reward, next_state)
            ddpg.soft_update(ddpg.critic_target, ddpg.critic)
        for _ in range(cfg.l):
            ddpg.update_actor(state)
            ddpg.soft_update(ddpg.actor_target, ddpg.actor)
        epsilon = max(cfg.eps_a / (cfg.eps_a + (m + 1)), cfg.eps_min)
        for s in scheds:
            s.step()
    ddpg.actor.eval()
    return ddpg


def evaluate(cfg, kappa, sigma, ou_kw, reg, ddpg, M, eval_seed=999, full_norm=False):
    """Vectorized eval over M episodes on ONE fixed test set. Returns (M,) rewards + mean|I|."""
    rng = np.random.default_rng(eval_seed)
    S_all = np.stack([simulate_path(cfg.n, rng, cfg.regimes, cfg.A, cfg.dt,
                                    kappa=kappa, sigma=sigma, s0=1.0, **ou_kw)[0]
                      for _ in range(M)])                       # (M, n)
    inv = torch.zeros(M, 1)
    total = torch.zeros(M, 1)
    absI = torch.zeros(M, 1)
    steps = 0
    with torch.no_grad():
        for t in range(cfg.W, cfg.n - 1):
            windows = torch.tensor(S_all[:, t - cfg.W:t + 1], dtype=torch.float32)
            pred = reg(windows)
            S_t = torch.tensor(S_all[:, t], dtype=torch.float32).reshape(-1, 1)
            S_next = torch.tensor(S_all[:, t + 1], dtype=torch.float32).reshape(-1, 1)
            state = build_state(S_t, inv, pred, cfg.I_max, full_norm)
            action = ddpg.actor(state)
            total += step_reward(inv, action, S_t, S_next, cfg.lam)
            absI += action.abs()
            inv = action
            steps += 1
    rewards = total.numpy().ravel()
    mean_absI = (absI / steps).mean().item()
    return rewards, mean_absI


PAPER = {1: (8.69, 0.60), 2: (2.95, 0.30), 3: (-0.07, 0.11)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", type=int, required=True, choices=[1, 2, 3])
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4")
    ap.add_argument("--pretrain_steps", type=int, default=None)
    ap.add_argument("--N", type=int, default=None)
    ap.add_argument("--M", type=int, default=None)
    ap.add_argument("--full_norm", action="store_true",
                    help="Normalize ALL features (S,I,pred) to [0,1] (paper); default only S.")
    ap.add_argument("--gamma", type=float, default=None,
                    help="Override the discount factor (config default is 0.99).")
    ap.add_argument("--lr_sched", choices=["cosine", "step"], default=None,
                    help="Decay the DDPG learning rate over training (paper: W-ADAM + scheduler).")
    args = ap.parse_args()

    cfg = load_config(f"configs/scenario{args.scenario}_reg.yaml")
    if args.gamma is not None:
        cfg.gamma = args.gamma
    kappa, sigma, ou_kw = ou_args_from_cfg(cfg)
    seeds = [int(s) for s in args.seeds.split(",")]
    pretrain_steps = args.pretrain_steps or cfg.reg_pretrain_steps
    N = args.N or cfg.N
    M = args.M or cfg.M

    tag = "fullnorm" if args.full_norm else "baseline"
    if args.gamma is not None:
        tag += f"_g{str(args.gamma).replace('.', '')}"
    if args.lr_sched:
        tag += f"_{args.lr_sched}"
    print(f"=== scenario {args.scenario} reg-DDPG sweep [{tag}] ===", flush=True)
    print(f"seeds={seeds}  pretrain_steps={pretrain_steps}  N={N}  M={M}  "
          f"full_norm={args.full_norm}  gamma={cfg.gamma}", flush=True)
    p_mean, p_std = PAPER[args.scenario]
    print(f"paper: {p_mean} +/- {p_std}", flush=True)

    t0 = time.time()
    reg = pretrain_regressor(cfg, kappa, sigma, ou_kw, pretrain_steps)
    print(f"[regressor pretrained in {time.time()-t0:.0f}s]", flush=True)

    means = []
    for seed in seeds:
        ts = time.time()
        ddpg = train_ddpg(cfg, kappa, sigma, ou_kw, reg, seed, N,
                          full_norm=args.full_norm, lr_sched=args.lr_sched)
        rewards, mean_absI = evaluate(cfg, kappa, sigma, ou_kw, reg, ddpg, M, full_norm=args.full_norm)
        means.append(rewards.mean())
        np.save(f"artifacts/sweep_s{args.scenario}_{tag}_seed{seed}.npy", rewards)
        print(f"  seed {seed:2d} | reward {rewards.mean():7.2f} +/- {rewards.std():5.2f} "
              f"| mean|I| {mean_absI:5.2f} | {time.time()-ts:.0f}s", flush=True)

    means = np.array(means)
    print(f"--- scenario {args.scenario}: mean-of-means {means.mean():.2f} "
          f"+/- {means.std():.2f} across {len(seeds)} seeds "
          f"(paper {p_mean} +/- {p_std}) ---", flush=True)
    print(f"[total {time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
