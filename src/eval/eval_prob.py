"""
Evaluate the prob agent (frozen classifier + trained policy). Scores on Eq. 4.

    python3 -m src.eval.eval_prob --scenario 1 --model ppo
    python3 -m src.eval.eval_prob --scenario 3 --model ppo --seed 999 --results prob.csv

State = (S_t, I_t, Phi), Phi from the frozen scenario{N}_classifier.pt. Writes a
scatter to figures/scenario{n}_prob_{model}.png.
"""
import argparse
import csv
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.data.ou_simulator import simulate_path
from src.env.trading_env import step_reward, normalize_state_features
from src.models.agents import build_agent
from src.models.gru_classifier import GRUClassifier
from src.utils.config import load_config
from src.utils.ou_args import ou_args_from_cfg
from src.eval.plot_rewards import plot_reward_histogram


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", type=int, required=True, choices=[1, 2, 3])
    p.add_argument("--model", default="ppo", choices=["ddpg", "td3", "ppo"])
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--results", default=None)
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--theta-only", action="store_true")
    args = p.parse_args()

    cfg = load_config(f"configs/scenario{args.scenario}_prob.yaml")
    kappa, sigma, ou_kw = ou_args_from_cfg(cfg)
    seed = args.seed if args.seed is not None else (
        cfg.train_seed if cfg.train_seed is not None else np.random.randint(0, 1_000_000))
    rng = np.random.default_rng(seed)
    print(f"scenario {args.scenario} | model {args.model} | prob | seed {seed}")

    # frozen classifier -> phi_dim -> state_dim
    ctag = "_thetaonly" if args.theta_only else ""
    ck = torch.load(f"artifacts/scenario{args.scenario}_classifier{ctag}.pt")
    clf = GRUClassifier(cfg.d_h, cfg.d_l, n_theta=ck["n_theta"],
                        n_kappa=ck["n_kappa"], n_sigma=ck["n_sigma"])
    clf.load_state_dict(ck["state_dict"]); clf.eval()
    for pm in clf.parameters():
        pm.requires_grad_(False)
    state_dim = 2 + clf.phi_dim

    tau = None if args.model == "ppo" else cfg.tau
    agent = build_agent(args.model, state_dim, cfg.action_dim, cfg.d_NN, cfg.l_NN,
                        cfg.I_max, cfg.gamma, tau, cfg.lr)
    vtag = "prob_thetaonly" if args.theta_only else "prob"
    ack = torch.load(f"artifacts/scenario{args.scenario}_{args.model}_{vtag}.pt")
    agent.actor.load_state_dict(ack["actor"]); agent.actor.eval()

    rewards, signals, invs = [], [], []
    with torch.no_grad():
        for i in range(cfg.M):
            S, _ = simulate_path(cfg.n, rng, cfg.regimes, cfg.A, cfg.dt,
                                 kappa=kappa, sigma=sigma, s0=1.0, **ou_kw)
            inv = torch.zeros(1, 1); total = 0.0
            for t in range(cfg.W, cfg.n - 1):
                w = torch.tensor(S[t-cfg.W:t+1], dtype=torch.float32).reshape(1, -1)
                st = torch.tensor([[S[t]]], dtype=torch.float32)
                phi = clf.phi(w)
                state = torch.cat([normalize_state_features(st), inv, phi], dim=1)
                a = agent.act_eval(state)
                total += step_reward(inv, a, S[t], S[t+1], cfg.lam).item()
                if i == 0:
                    signals.append(S[t]); invs.append(a.item())
                inv = a
            rewards.append(total)

    rewards = np.array(rewards)
    mean, std = rewards.mean(), rewards.std()
    mean_I = float(np.abs(invs).mean())
    bound_hits = int((np.abs(invs) >= 8.0).sum())
    print(f"mean {mean:.2f} +/- {std:.2f}")
    print(f"(prob; trained eta={cfg.eta} psi={cfg.psi}; scored on paper Eq. 4)")
    print(f"mean |I| = {mean_I:.3f}, steps at |I|>=8: {bound_hits}/{len(invs)}")

    if args.results:
        new = not os.path.exists(args.results)
        with open(args.results, "a", newline="") as f:
            wr = csv.writer(f)
            if new:
                wr.writerow(["scenario", "variant", "model", "seed", "mean", "std", "mean_I", "bound_hits"])
            wr.writerow([args.scenario, "prob", args.model, seed, f"{mean:.4f}", f"{std:.4f}",
                         f"{mean_I:.4f}", bound_hits])

    if not args.no_plot:
        os.makedirs("figures", exist_ok=True); os.makedirs("artifacts", exist_ok=True)
        plt.figure(figsize=(7, 5))
        plt.scatter(signals, invs, s=4, alpha=0.4)
        plt.axvline(1.0, color="gray", ls="--", lw=0.8)
        plt.axhline(0.0, color="gray", ls=":", lw=0.6)
        plt.xlabel("signal $S_t$"); plt.ylabel("chosen inventory $I_{t+1}$")
        plt.title(f"Scenario {args.scenario} prob-{args.model}: inventory vs signal")
        plt.tight_layout()
        path = f"figures/scenario{args.scenario}_prob_{args.model}.png"
        plt.savefig(path, dpi=130); plt.close()
        print(f"saved {path}")
        np.save(f"artifacts/rewards_scenario{args.scenario}_{args.model}_prob.npy", rewards)
        plot_reward_histogram(rewards, cfg.scenario, variant="prob", model=args.model, bins=10)


if __name__ == "__main__":
    main()