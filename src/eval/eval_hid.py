"""
Unified evaluation: any scenario, any model. Scores on the paper's Eq. 4.

    python3 -m src.eval.eval_hid --scenario 1 --model ppo
    python3 -m src.eval.eval_hid --scenario 3 --model ddpg --results results.csv

Prints "mean X +/- Y". With --results, appends one row
(scenario, model, seed, mean, std, mean_I, bound_hits) to a CSV for aggregation.
Writes a policy scatter to figures/scenario{n}_hid_{model}.png (Fig 13a style)
and a reward histogram unless --no-plot.
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
from src.models.gru import GRUEncoder
from src.utils.config import load_config
from src.utils.ou_args import ou_args_from_cfg
from src.eval.plot_rewards import plot_reward_histogram


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", type=int, required=True, choices=[1, 2, 3])
    p.add_argument("--model", default="ddpg", choices=["ddpg", "td3", "ppo"])
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--results", default=None, help="CSV to append a summary row to")
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args()

    cfg = load_config(f"configs/scenario{args.scenario}_hid.yaml")
    kappa, sigma, ou_kw = ou_args_from_cfg(cfg)

    seed = args.seed if args.seed is not None else (cfg.train_seed if cfg.train_seed is not None else np.random.randint(0, 1_000_000))
    rng = np.random.default_rng(seed)
    print(f"scenario {args.scenario} | model {args.model} | seed {seed}")

    encoder = GRUEncoder(cfg.d_h, cfg.d_l, enc_dim=cfg.enc_dim)
    tau = None if args.model == "ppo" else cfg.tau
    agent = build_agent(args.model, cfg.state_dim, cfg.action_dim, cfg.d_NN, cfg.l_NN,
                        cfg.I_max, cfg.gamma, tau, cfg.lr)

    ckpt = torch.load(f"artifacts/scenario{args.scenario}_{args.model}_hid.pt")
    agent.actor.load_state_dict(ckpt["actor"])
    encoder.load_state_dict(ckpt["encoder"])
    agent.actor.eval(); encoder.eval()

    rewards, signals, invs = [], [], []
    with torch.no_grad():
        for i in range(cfg.M):
            S, _ = simulate_path(cfg.n, rng, cfg.regimes, cfg.A, cfg.dt,
                                 kappa=kappa, sigma=sigma, s0=1.0, **ou_kw)
            inv = torch.zeros(1, 1); total = 0.0
            for t in range(cfg.W, cfg.n - 1):
                w = torch.tensor(S[t-cfg.W:t+1], dtype=torch.float32).reshape(1, -1)
                st = torch.tensor([[S[t]]], dtype=torch.float32)
                o, _ = encoder(w)
                state = torch.cat([normalize_state_features(st), inv, o], dim=1)
                a = agent.act_eval(state)
                total += step_reward(inv, a, S[t], S[t+1], cfg.lam).item()   # Eq.4 scoring
                if i == 0:
                    signals.append(S[t]); invs.append(a.item())
                inv = a
            rewards.append(total)

    rewards = np.array(rewards)
    mean, std = rewards.mean(), rewards.std()
    mean_I = float(np.abs(invs).mean())
    bound_hits = int((np.abs(invs) >= 8.0).sum())
    print(f"mean {mean:.2f} +/- {std:.2f}")
    print(f"(trained eta={cfg.eta} psi={cfg.psi}; scored on paper Eq. 4)")
    print(f"mean |I| = {mean_I:.3f}, steps at |I|>=8: {bound_hits}/{len(invs)}")

    if args.results:
        new = not os.path.exists(args.results)
        with open(args.results, "a", newline="") as f:
            wr = csv.writer(f)
            if new:
                wr.writerow(["scenario", "model", "seed", "mean", "std", "mean_I", "bound_hits"])
            wr.writerow([args.scenario, args.model, seed, f"{mean:.4f}", f"{std:.4f}",
                         f"{mean_I:.4f}", bound_hits])

    if not args.no_plot:
        os.makedirs("figures", exist_ok=True)
        os.makedirs("artifacts", exist_ok=True)

        # --- policy scatter (Fig 13a style) -> figures/scenario{n}_hid_{model}.png ---
        plt.figure(figsize=(7, 5))
        plt.scatter(signals, invs, s=4, alpha=0.4)
        plt.axvline(1.0, color="gray", ls="--", lw=0.8)
        plt.axhline(0.0, color="gray", ls=":", lw=0.6)
        plt.xlabel("signal $S_t$")
        plt.ylabel("chosen inventory $I_{t+1}$")
        plt.title(f"Scenario {args.scenario} hid-{args.model}: inventory vs signal (cf. Fig 13a)")
        plt.tight_layout()
        scatter_path = f"figures/scenario{args.scenario}_hid_{args.model}.png"
        plt.savefig(scatter_path, dpi=130)
        plt.close()
        print(f"saved {scatter_path}")

        # --- reward histogram + raw rewards ---
        np.save(f"artifacts/rewards_scenario{args.scenario}_{args.model}_hid.npy", rewards)
        plot_reward_histogram(rewards, cfg.scenario, variant="hid", model=args.model, bins=10)


if __name__ == "__main__":
    main()