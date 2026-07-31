"""
Unified training: any scenario, any model.

    python3 -m src.train.train --scenario 1 --model ppo
    python3 -m src.train.train --scenario 3 --model ddpg --seed 42

Replaces the per-scenario train scripts. Scenario only selects the config file;
everything else flows from cfg. Branches ddpg/td3 (off-policy) vs ppo (on-policy).
"""
import argparse
import numpy as np
import torch

from src.data.ou_simulator import simulate_path
from src.env.trading_env import sample_batch, step_reward, normalize_state_features
from src.models.gru import GRUEncoder
from src.models.agents import build_agent
from src.models.ppo import compute_gae
from src.utils.config import load_config
from src.utils.ou_args import ou_args_from_cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", type=int, required=True, choices=[1, 2, 3])
    p.add_argument("--model", default="ddpg", choices=["ddpg", "td3", "ppo"])
    p.add_argument("--seed", type=int, default=None, help="override cfg.train_seed")
    args = p.parse_args()

    cfg = load_config(f"configs/scenario{args.scenario}_hid.yaml")
    kappa, sigma, ou_kw = ou_args_from_cfg(cfg)

    seed = args.seed if args.seed is not None else (
        cfg.train_seed if cfg.train_seed is not None else np.random.randint(0, 1_000_000))
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    print(f"scenario {args.scenario} | model {args.model} | seed {seed}")
    print(f"reward penalties: eta={cfg.eta} psi={cfg.psi}")

    encoder = GRUEncoder(cfg.d_h, cfg.d_l, enc_dim=cfg.enc_dim)
    gru_opt = torch.optim.AdamW(encoder.parameters(), lr=cfg.lr, weight_decay=1e-5)

    def save(agent):
        import os
        os.makedirs("artifacts", exist_ok=True)
        ckpt = {**agent.state_dicts(), "encoder": encoder.state_dict(), "model": args.model}
        path = f"artifacts/scenario{args.scenario}_{args.model}_hid.pt"
        torch.save(ckpt, path)
        print(f"saved {path}")

    # ---------------- PPO: on-policy episodic rollouts ----------------
    if args.model == "ppo":
        N_UPD = getattr(cfg, "ppo_updates", 300)
        gsch = torch.optim.lr_scheduler.CosineAnnealingLR(gru_opt, T_max=N_UPD)
        agent = build_agent("ppo", cfg.state_dim, cfg.action_dim, cfg.d_NN, cfg.l_NN,
                            cfg.I_max, cfg.gamma, tau=None,
                            lr=getattr(cfg, "ppo_lr", 3e-4), N=N_UPD)

        def bstate(S, t, inv):
            w = torch.tensor(S[t-cfg.W:t+1], dtype=torch.float32).reshape(1, -1)
            st = torch.tensor([[S[t]]], dtype=torch.float32)
            o, _ = encoder(w)
            return torch.cat([normalize_state_features(st), inv, o], dim=1)

        for upd in range(N_UPD):
            S, _ = simulate_path(cfg.n, rng, cfg.regimes, cfg.A, cfg.dt,
                                 kappa=kappa, sigma=sigma, s0=1.0, **ou_kw)
            inv = torch.zeros(1, 1)
            states, acts, logps, rews, vals, gw, gt = [], [], [], [], [], [], []
            for t in range(cfg.W, cfg.n - 1):
                with torch.no_grad():
                    s = bstate(S, t, inv)
                    a, lp = agent.actor.act(s)
                    v = agent.critic(s)
                r = step_reward(inv, a, S[t], S[t+1], cfg.lam, eta=cfg.eta, psi=cfg.psi)
                states.append(s); acts.append(a); logps.append(lp)
                rews.append(float(r)); vals.append(v)
                gw.append(S[t-cfg.W:t+1]); gt.append(S[t+1])
                inv = a.detach()
            with torch.no_grad():
                nv = agent.critic(bstate(S, cfg.n - 1, inv)).item()
            states = torch.cat(states); acts = torch.cat(acts); olp = torch.cat(logps)
            vt = torch.cat(vals).squeeze(-1); rt = torch.tensor(rews, dtype=torch.float32)
            adv, ret = compute_gae(rt, vt, nv, cfg.gamma, agent.gae_lambda)
            agent.update(states, acts, olp, adv.unsqueeze(-1), ret.unsqueeze(-1))
            agent.step_scheduler()

            gwt = torch.tensor(np.array(gw), dtype=torch.float32)
            gtt = torch.tensor(gt, dtype=torch.float32).reshape(-1, 1)
            _, sp = encoder(gwt)
            gl = torch.nn.functional.mse_loss(sp, gtt)
            gru_opt.zero_grad(); gl.backward(); gru_opt.step(); gsch.step()

            if upd % 20 == 0:
                print(f"update {upd:4d} | rollout reward {rt.sum().item():8.2f} | "
                      f"gru {gl.item():.4f} | lr {agent.current_lr():.6f} | "
                      f"log_std {agent.actor.log_std.item():.3f}")
        save(agent)

    # ---------------- DDPG / TD3: off-policy random batches ----------------
    else:
        gsch = torch.optim.lr_scheduler.CosineAnnealingLR(gru_opt, T_max=cfg.N)
        agent = build_agent(args.model, cfg.state_dim, cfg.action_dim, cfg.d_NN, cfg.l_NN,
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

            o, sp = encoder(wt)
            gl = torch.nn.functional.mse_loss(sp, Sn)
            gru_opt.zero_grad(); gl.backward(); gru_opt.step()
            with torch.no_grad():
                o, _ = encoder(wt); on, _ = encoder(nwt)

            state = torch.cat([normalize_state_features(St), It, o], dim=1)
            action = agent.select_action(state, eps)
            reward = step_reward(It, action, St, Sn, cfg.lam, eta=cfg.eta, psi=cfg.psi)
            nstate = torch.cat([normalize_state_features(Sn), action, on], dim=1)

            for _ in range(cfg.ell):
                cl = agent.update_critic(state, action, reward, nstate)
                agent.update_targets()
            for _ in range(cfg.l):
                al = agent.update_actor(state)
                agent.soft_update(agent.actor_target, agent.actor)
            agent.step_schedulers(); gsch.step()

            if m % cfg.log_every == 0:
                print(f"iter {m:5d} | critic {cl:.4f} | actor {al:.4f} | "
                      f"gru {gl.item():.4f} | eps {eps:.3f} | lr {agent.current_lr():.5f}")
            eps = max(cfg.eps_a / (cfg.eps_a + (m + 1)), cfg.eps_min)
        save(agent)


if __name__ == "__main__":
    main()