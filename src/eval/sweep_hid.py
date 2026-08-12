"""
Multi-seed sweep for hid-DDPG, with an optional inventory-normalization switch.

Companion to sweep_reg.py. Purpose: test whether normalizing the inventory feature
to [0,1] collapses the hid-DDPG policy (as the repo's normalize_state_features
docstring claims). This is the decisive control for the "normalization is the lever"
hypothesis:
  - if hid ALSO collapses under inventory-norm, then inventory-norm is uniformly
    catastrophic here and the paper's "features in [0,1]" cannot be the literal
    explanation for reg-DDPG's low numbers -> deeper difference.
  - if hid stays good, the docstring is stale and full-norm is safe globally.

hid pipeline (matches scenario*_hid_train.py): the GRU encoder is trained JOINTLY
with the DDPG each iteration via an auxiliary MSE head (predict S_{t+1}), then its
encoding o_t is detached and used as the third state feature. No frozen pretraining.

Note: o_t (the GRU encoding) has no natural [0,1] range, so we do NOT normalize it.
The only inventory-vs-signal normalization question is about I_t, which is exactly
what --norm_inventory toggles. Reward is always computed from RAW quantities.

Usage:
    python -m src.eval.sweep_hid --scenario 1 --seeds 0,1,2                  # baseline
    python -m src.eval.sweep_hid --scenario 1 --seeds 0,1,2 --norm_inventory # control
"""

import argparse
import time

import numpy as np
import torch

from src.data.ou_simulator import simulate_path
from src.env.trading_env import sample_batch, step_reward, normalize_state_features
from src.models.gru import GRUEncoder
from src.models.ddpg import DDPG
from src.utils.config import load_config
from src.utils.ou_args import ou_args_from_cfg


def build_state(S_col, I_col, o_col, I_max, norm_inventory):
    """
    G_t = (S_t, I_t, o_t). S_t always normalized to [0,1]; o_t (GRU encoding) left
    raw (no natural range). --norm_inventory maps I_t from [-I_max, I_max] -> [0,1].
    Reward is computed from raw I by the caller -- this only rescales network input.
    """
    S_n = normalize_state_features(S_col)
    if norm_inventory:
        I_col = ((I_col + I_max) / (2 * I_max)).clamp(0.0, 1.0)
    return torch.cat([S_n, I_col, o_col], dim=1)


def train_hid(cfg, kappa, sigma, ou_kw, seed, N, norm_inventory=False):
    """Train hid-DDPG (encoder trained jointly) for one seed."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    encoder = GRUEncoder(cfg.d_h, cfg.d_l, enc_dim=cfg.enc_dim)
    gru_opt = torch.optim.AdamW(encoder.parameters(), lr=cfg.lr, weight_decay=1e-5)
    ddpg = DDPG(cfg.state_dim, cfg.action_dim, cfg.d_NN, cfg.l_NN,
                cfg.I_max, cfg.gamma, cfg.tau, cfg.lr)
    epsilon = 1.0
    for m in range(N):
        batch = sample_batch(cfg.batch_size, cfg.W, rng, cfg.regimes, cfg.A,
                             kappa, sigma, cfg.dt, cfg.I_max, **ou_kw)
        windows = torch.tensor(batch["windows"], dtype=torch.float32)
        next_windows = torch.tensor(batch["next_windows"], dtype=torch.float32)
        S_t = torch.tensor(batch["S_t"], dtype=torch.float32).reshape(-1, 1)
        I_t = torch.tensor(batch["I_t"], dtype=torch.float32).reshape(-1, 1)
        S_next = torch.tensor(batch["S_next"], dtype=torch.float32).reshape(-1, 1)

        # --- GRU aux training (predict S_{t+1}), before the RL step ---
        o_t, s_pred = encoder(windows)
        gru_loss = torch.nn.functional.mse_loss(s_pred, S_next)
        gru_opt.zero_grad(); gru_loss.backward(); gru_opt.step()
        with torch.no_grad():
            o_t, _ = encoder(windows)
            o_next, _ = encoder(next_windows)

        state = build_state(S_t, I_t, o_t, cfg.I_max, norm_inventory)
        action = ddpg.select_action(state, epsilon)
        reward = step_reward(I_t, action, S_t, S_next, cfg.lam)
        next_state = build_state(S_next, action, o_next, cfg.I_max, norm_inventory)
        for _ in range(cfg.ell):
            ddpg.update_critic(state, action, reward, next_state)
            ddpg.soft_update(ddpg.critic_target, ddpg.critic)
        for _ in range(cfg.l):
            ddpg.update_actor(state)
            ddpg.soft_update(ddpg.actor_target, ddpg.actor)
        epsilon = max(cfg.eps_a / (cfg.eps_a + (m + 1)), cfg.eps_min)
    ddpg.actor.eval(); encoder.eval()
    return encoder, ddpg


def evaluate(cfg, kappa, sigma, ou_kw, encoder, ddpg, M, eval_seed=999, norm_inventory=False):
    """Vectorized eval over M episodes on one fixed test set. Returns (M,) rewards + mean|I|."""
    rng = np.random.default_rng(eval_seed)
    S_all = np.stack([simulate_path(cfg.n, rng, cfg.regimes, cfg.A, cfg.dt,
                                    kappa=kappa, sigma=sigma, s0=1.0, **ou_kw)[0]
                      for _ in range(M)])
    inv = torch.zeros(M, 1)
    total = torch.zeros(M, 1)
    absI = torch.zeros(M, 1)
    steps = 0
    with torch.no_grad():
        for t in range(cfg.W, cfg.n - 1):
            windows = torch.tensor(S_all[:, t - cfg.W:t + 1], dtype=torch.float32)
            o_t, _ = encoder(windows)
            S_t = torch.tensor(S_all[:, t], dtype=torch.float32).reshape(-1, 1)
            S_next = torch.tensor(S_all[:, t + 1], dtype=torch.float32).reshape(-1, 1)
            state = build_state(S_t, inv, o_t, cfg.I_max, norm_inventory)
            action = ddpg.actor(state)
            total += step_reward(inv, action, S_t, S_next, cfg.lam)
            absI += action.abs()
            inv = action
            steps += 1
    return total.numpy().ravel(), (absI / steps).mean().item()


PAPER = {1: (15.70, 1.39), 2: (8.08, 1.54), 3: (1.29, 3.49)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", type=int, required=True, choices=[1, 2, 3])
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--N", type=int, default=None)
    ap.add_argument("--M", type=int, default=None)
    ap.add_argument("--norm_inventory", action="store_true",
                    help="Normalize inventory to [0,1] (the paper's reading); default leaves it raw.")
    args = ap.parse_args()

    cfg = load_config(f"configs/scenario{args.scenario}_hid.yaml")
    kappa, sigma, ou_kw = ou_args_from_cfg(cfg)
    seeds = [int(s) for s in args.seeds.split(",")]
    N = args.N or cfg.N
    M = args.M or cfg.M
    tag = "invnorm" if args.norm_inventory else "baseline"

    print(f"=== scenario {args.scenario} hid-DDPG sweep [{tag}] ===", flush=True)
    print(f"seeds={seeds}  N={N}  M={M}  norm_inventory={args.norm_inventory}", flush=True)
    p_mean, p_std = PAPER[args.scenario]
    print(f"paper: {p_mean} +/- {p_std}", flush=True)

    t0 = time.time()
    means = []
    for seed in seeds:
        ts = time.time()
        encoder, ddpg = train_hid(cfg, kappa, sigma, ou_kw, seed, N, args.norm_inventory)
        rewards, mean_absI = evaluate(cfg, kappa, sigma, ou_kw, encoder, ddpg, M,
                                      norm_inventory=args.norm_inventory)
        means.append(rewards.mean())
        np.save(f"artifacts/sweep_hid_s{args.scenario}_{tag}_seed{seed}.npy", rewards)
        print(f"  seed {seed:2d} | reward {rewards.mean():7.2f} +/- {rewards.std():5.2f} "
              f"| mean|I| {mean_absI:5.2f} | {time.time()-ts:.0f}s", flush=True)

    means = np.array(means)
    print(f"--- scenario {args.scenario} hid [{tag}]: mean-of-means {means.mean():.2f} "
          f"+/- {means.std():.2f} across {len(seeds)} seeds "
          f"(paper {p_mean} +/- {p_std}) ---", flush=True)
    print(f"[total {time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
