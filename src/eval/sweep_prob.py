"""
Multi-seed sweep for prob-DDPG (scenarios 1/2/3), to complete the paper's
three-way ranking prob > hid > reg (Table 4).

prob-DDPG is the two-step method: step 1 pretrains a GRU classifier to output the
posterior over the three theta regimes P(theta_t = phi_k | window) via cross-
entropy on the TRUE regime label (from the simulator); step 2 freezes it and feeds
G_t = (S_t, I_t, Phi_t) -- a 5-dim state -- into DDPG.

Design matches sweep_reg.py: classifier pretrained ONCE per scenario (fixed base
seed) and frozen; DDPG retrained per seed (the variance source); eval on one fixed
test set, vectorized across the M episodes. The posterior Phi is already in [0,1]
and sums to 1, so no feature-normalization question arises for it.

Usage:
    python -m src.eval.sweep_prob --scenario 1 --seeds 0,1,2,3,4
    python -m src.eval.sweep_prob --scenario 2 --seeds 0,1 --clf_steps 500 --N 500 --M 50  # calib
"""

import argparse
import time

import numpy as np
import torch

from src.data.ou_simulator import simulate_path
from src.env.trading_env import sample_batch, step_reward, normalize_state_features
from src.models.classifier import GRUClassifier
from src.models.ddpg import DDPG
from src.utils.config import load_config
from src.utils.ou_args import ou_args_from_cfg


def pretrain_classifier(cfg, kappa, sigma, ou_kw, steps, num_classes, base_seed=12345):
    """Pretrain + freeze the GRU regime classifier (cross-entropy on true labels)."""
    rng = np.random.default_rng(base_seed)
    torch.manual_seed(base_seed)
    clf = GRUClassifier(cfg.clf_hidden_dim, cfg.clf_layers, cfg.clf_ffn_layers,
                        cfg.clf_ffn_hidden, num_classes=num_classes)
    opt = torch.optim.AdamW(clf.parameters(), lr=cfg.lr, weight_decay=1e-5)
    loss_fn = torch.nn.CrossEntropyLoss()
    clf.train()
    last_acc = 0.0
    for step in range(steps):
        batch = sample_batch(cfg.batch_size, cfg.W, rng, cfg.regimes, cfg.A,
                             kappa, sigma, cfg.dt, cfg.I_max, **ou_kw)
        windows = torch.tensor(batch["windows"], dtype=torch.float32)
        labels = torch.tensor(batch["regime"], dtype=torch.long)   # theta regime at t
        logits = clf(windows)
        loss = loss_fn(logits, labels)
        opt.zero_grad(); loss.backward(); opt.step()
        if step == steps - 1:
            last_acc = (logits.argmax(1) == labels).float().mean().item()
    clf.eval()
    for p in clf.parameters():
        p.requires_grad = False
    return clf, last_acc


def train_ddpg(cfg, kappa, sigma, ou_kw, clf, seed, N):
    """Train a fresh DDPG with the frozen classifier posterior as a feature."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
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
        with torch.no_grad():
            phi_t = clf.posterior(windows)          # (b, K)
            phi_next = clf.posterior(next_windows)
        state = torch.cat([normalize_state_features(S_t), I_t, phi_t], dim=1)
        action = ddpg.select_action(state, epsilon)
        reward = step_reward(I_t, action, S_t, S_next, cfg.lam)
        next_state = torch.cat([normalize_state_features(S_next), action, phi_next], dim=1)
        for _ in range(cfg.ell):
            ddpg.update_critic(state, action, reward, next_state)
            ddpg.soft_update(ddpg.critic_target, ddpg.critic)
        for _ in range(cfg.l):
            ddpg.update_actor(state)
            ddpg.soft_update(ddpg.actor_target, ddpg.actor)
        epsilon = max(cfg.eps_a / (cfg.eps_a + (m + 1)), cfg.eps_min)
    ddpg.actor.eval()
    return ddpg


def evaluate(cfg, kappa, sigma, ou_kw, clf, ddpg, M, eval_seed=999):
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
            phi = clf.posterior(windows)
            S_t = torch.tensor(S_all[:, t], dtype=torch.float32).reshape(-1, 1)
            S_next = torch.tensor(S_all[:, t + 1], dtype=torch.float32).reshape(-1, 1)
            state = torch.cat([normalize_state_features(S_t), inv, phi], dim=1)
            action = ddpg.actor(state)
            total += step_reward(inv, action, S_t, S_next, cfg.lam)
            absI += action.abs()
            inv = action
            steps += 1
    return total.numpy().ravel(), (absI / steps).mean().item()


PAPER = {1: (25.65, 3.35), 2: (15.59, 3.83), 3: (4.51, 3.75)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", type=int, required=True, choices=[1, 2, 3])
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4")
    ap.add_argument("--clf_steps", type=int, default=None)
    ap.add_argument("--N", type=int, default=None)
    ap.add_argument("--M", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(f"configs/scenario{args.scenario}_prob.yaml")
    kappa, sigma, ou_kw = ou_args_from_cfg(cfg)
    seeds = [int(s) for s in args.seeds.split(",")]
    num_classes = len(cfg.regimes)
    assert cfg.state_dim == 2 + num_classes, \
        f"state_dim {cfg.state_dim} != 2 + {num_classes}; set enc_dim={num_classes}"
    clf_steps = args.clf_steps or cfg.clf_pretrain_steps
    N = args.N or cfg.N
    M = args.M or cfg.M

    print(f"=== scenario {args.scenario} prob-DDPG sweep ===", flush=True)
    print(f"seeds={seeds}  clf_steps={clf_steps}  N={N}  M={M}  state_dim={cfg.state_dim}", flush=True)
    p_mean, p_std = PAPER[args.scenario]
    print(f"paper: {p_mean} +/- {p_std}", flush=True)

    t0 = time.time()
    clf, acc = pretrain_classifier(cfg, kappa, sigma, ou_kw, clf_steps, num_classes)
    print(f"[classifier pretrained in {time.time()-t0:.0f}s | train acc {acc:.3f}]", flush=True)

    means = []
    for seed in seeds:
        ts = time.time()
        ddpg = train_ddpg(cfg, kappa, sigma, ou_kw, clf, seed, N)
        rewards, mean_absI = evaluate(cfg, kappa, sigma, ou_kw, clf, ddpg, M)
        means.append(rewards.mean())
        np.save(f"artifacts/sweep_prob_s{args.scenario}_seed{seed}.npy", rewards)
        print(f"  seed {seed:2d} | reward {rewards.mean():7.2f} +/- {rewards.std():5.2f} "
              f"| mean|I| {mean_absI:5.2f} | {time.time()-ts:.0f}s", flush=True)

    means = np.array(means)
    print(f"--- scenario {args.scenario} prob: mean-of-means {means.mean():.2f} "
          f"+/- {means.std():.2f} across {len(seeds)} seeds "
          f"(paper {p_mean} +/- {p_std}) ---", flush=True)
    print(f"[total {time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
