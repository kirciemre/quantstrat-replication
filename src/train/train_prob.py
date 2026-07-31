"""
Stage 2: train the prob agent (PPO or DDPG/TD3) on the FROZEN classifier's
posterior features. State = (S_t, I_t, Phi), Phi from the frozen GRUClassifier.

    python3 -m src.train.train_prob --scenario 3 --model ppo

Requires Stage 1 first:
    python3 -m src.train.train_classifier --scenario 3

The agent code is identical to hid; only the encoder (frozen classifier) and the
state vector (S_t, I_t, Phi) change. state_dim = 2 + phi_dim
(Sc1: 5, Sc2: 7, Sc3: 9). The classifier is frozen -- no aux loss, no grads.
"""
import argparse
import numpy as np
import torch

from src.data.ou_simulator import simulate_path
from src.env.trading_env import sample_batch, step_reward, normalize_state_features
from src.models.gru_classifier import GRUClassifier
from src.models.agents import build_agent
from src.models.ppo import compute_gae
from src.utils.config import load_config
from src.utils.ou_args import ou_args_from_cfg


def load_frozen_classifier(scenario, cfg):
    ck = torch.load(f"artifacts/scenario{scenario}_classifier.pt")
    clf = GRUClassifier(cfg.d_h, cfg.d_l, n_theta=ck["n_theta"],
                        n_kappa=ck["n_kappa"], n_sigma=ck["n_sigma"])
    clf.load_state_dict(ck["state_dict"])
    clf.eval()
    for p in clf.parameters():
        p.requires_grad_(False)               # FROZEN
    return clf, clf.phi_dim


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", type=int, required=True, choices=[1, 2, 3])
    p.add_argument("--model", default="ppo", choices=["ddpg", "td3", "ppo"])
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    cfg = load_config(f"configs/scenario{args.scenario}_prob.yaml")
    kappa, sigma, ou_kw = ou_args_from_cfg(cfg)

    seed = args.seed if args.seed is not None else (
        cfg.train_seed if cfg.train_seed is not None else np.random.randint(0, 1_000_000))
    rng = np.random.default_rng(seed); torch.manual_seed(seed)

    clf, phi_dim = load_frozen_classifier(args.scenario, cfg)
    state_dim = 2 + phi_dim                     # (S_t, I_t, Phi)
    print(f"scenario {args.scenario} | model {args.model} | prob | "
          f"phi_dim {phi_dim} state_dim {state_dim} | seed {seed}")
    print(f"reward penalties: eta={cfg.eta} psi={cfg.psi}")

    def save(agent):
        import os
        os.makedirs("artifacts", exist_ok=True)
        # classifier is frozen; save agent only (classifier reloaded from its own file)
        torch.save({**agent.state_dicts(), "model": args.model, "variant": "prob"},
                   f"artifacts/scenario{args.scenario}_{args.model}_prob.pt")
        print(f"saved artifacts/scenario{args.scenario}_{args.model}_prob.pt")

    # ---------------- PPO: on-policy episodic rollouts ----------------
    if args.model == "ppo":
        N_UPD = getattr(cfg, "ppo_updates", 300)
        agent = build_agent("ppo", state_dim, cfg.action_dim, cfg.d_NN, cfg.l_NN,
                            cfg.I_max, cfg.gamma, tau=None,
                            lr=getattr(cfg, "ppo_lr", 3e-4), N=N_UPD)

        def bstate(S, t, inv):
            w = torch.tensor(S[t-cfg.W:t+1], dtype=torch.float32).reshape(1, -1)
            st = torch.tensor([[S[t]]], dtype=torch.float32)
            phi = clf.phi(w)                    # frozen posterior feature
            return torch.cat([normalize_state_features(st), inv, phi], dim=1)

        for upd in range(N_UPD):
            S, _ = simulate_path(cfg.n, rng, cfg.regimes, cfg.A, cfg.dt,
                                 kappa=kappa, sigma=sigma, s0=1.0, **ou_kw)
            inv = torch.zeros(1, 1)
            states, acts, logps, rews, vals = [], [], [], [], []
            for t in range(cfg.W, cfg.n - 1):
                with torch.no_grad():
                    s = bstate(S, t, inv)
                    a, lp = agent.actor.act(s); v = agent.critic(s)
                r = step_reward(inv, a, S[t], S[t+1], cfg.lam, eta=cfg.eta, psi=cfg.psi)
                states.append(s); acts.append(a); logps.append(lp)
                rews.append(float(r)); vals.append(v)
                inv = a.detach()
            with torch.no_grad():
                nv = agent.critic(bstate(S, cfg.n - 1, inv)).item()
            states = torch.cat(states); acts = torch.cat(acts); olp = torch.cat(logps)
            vt = torch.cat(vals).squeeze(-1); rt = torch.tensor(rews, dtype=torch.float32)
            adv, ret = compute_gae(rt, vt, nv, cfg.gamma, agent.gae_lambda)
            agent.update(states, acts, olp, adv.unsqueeze(-1), ret.unsqueeze(-1))
            agent.step_scheduler()
            if upd % 20 == 0:
                print(f"update {upd:4d} | rollout reward {rt.sum().item():8.2f} | "
                      f"lr {agent.current_lr():.6f} | log_std {agent.actor.log_std.item():.3f}")
        save(agent)

    # ---------------- DDPG / TD3: off-policy random batches ----------------
    else:
        agent = build_agent(args.model, state_dim, cfg.action_dim, cfg.d_NN, cfg.l_NN,
                            cfg.I_max, cfg.gamma, cfg.tau, cfg.lr, cfg.N)
        eps = 1.0
        for m in range(cfg.N):
            b = sample_batch(cfg.batch_size, cfg.W, rng, cfg.regimes, cfg.A,
                             kappa, sigma, cfg.dt, cfg.I_max, **ou_kw)
            wt = torch.tensor(b["windows"], dtype=torch.float32)
            nwt = torch.tensor(b["next_windows"], dtype=torch.float32)
            St = torch.tensor(b["S_t"], dtype=torch.float32).reshape(-1, 1)
            It = torch.tensor(b["I_t"], dtype=torch.float32).reshape(-1, 1)
            Sn = torch.tensor(b["S_next"], dtype=torch.float32).reshape(-1, 1)
            with torch.no_grad():
                phi = clf.phi(wt); phi_n = clf.phi(nwt)
            state = torch.cat([normalize_state_features(St), It, phi], dim=1)
            action = agent.select_action(state, eps)
            reward = step_reward(It, action, St, Sn, cfg.lam, eta=cfg.eta, psi=cfg.psi)
            nstate = torch.cat([normalize_state_features(Sn), action, phi_n], dim=1)
            for _ in range(cfg.ell):
                cl = agent.update_critic(state, action, reward, nstate); agent.update_targets()
            for _ in range(cfg.l):
                al = agent.update_actor(state); agent.soft_update(agent.actor_target, agent.actor)
            agent.step_schedulers()
            if m % cfg.log_every == 0:
                print(f"iter {m:5d} | critic {cl:.4f} | actor {al:.4f} | eps {eps:.3f}")
            eps = max(cfg.eps_a / (cfg.eps_a + (m + 1)), cfg.eps_min)
        save(agent)


if __name__ == "__main__":
    main()