import numpy as np
import torch
import matplotlib.pyplot as plt

from src.data.ou_simulator import simulate_path
from src.env.trading_env import step_reward, normalize_state_features
from src.models.ddpg import DDPG
from src.models.gru import GRUEncoder
from src.utils.config import load_config
from src.utils.ou_args import ou_args_from_cfg
from src.eval.plot_rewards import plot_reward_histogram


cfg = load_config("configs/scenario3_hid.yaml")
kappa, sigma, ou_kw = ou_args_from_cfg(cfg)


seed = cfg.train_seed if cfg.train_seed is not None else np.random.randint(0, 1_000_000)
rng = np.random.default_rng(seed)
print(f"seed: {seed}")


encoder = GRUEncoder(cfg.d_h, cfg.d_l, enc_dim=cfg.enc_dim)
ddpg = DDPG(cfg.state_dim, cfg.action_dim, cfg.d_NN, cfg.l_NN, cfg.I_max, cfg.gamma, cfg.tau, cfg.lr)   # no encoder arg

ckpt = torch.load("artifacts/backup/scenario3_hid_backup.pt")
ddpg.actor.load_state_dict(ckpt["actor"])
ddpg.critic.load_state_dict(ckpt["critic"])
encoder.load_state_dict(ckpt["encoder"])
ddpg.actor.eval(); ddpg.critic.eval(); encoder.eval()

episode_rewards = []

signals = []
invs = []

with torch.no_grad():
    for i in range(cfg.M):
        # TEST phase: fixed start S_0 = 1 (paper), inventory I_0 = 0.
        S, _ = simulate_path(cfg.n, rng, cfg.regimes, cfg.A, cfg.dt, kappa=kappa, sigma=sigma, s0=1.0, **ou_kw)
        inventory = torch.zeros(1, 1)
        total_reward = 0.0

        for t in range(cfg.W, cfg.n - 1):
            window = torch.tensor(S[t-cfg.W:t+1], dtype=torch.float32).reshape(1, -1)
            S_t_col = torch.tensor([[S[t]]], dtype=torch.float32)

            o_t, _ = encoder(window)                          # unpack: encoding head only
            S_t_norm = normalize_state_features(S_t_col)
            state = torch.cat([S_t_norm, inventory, o_t], dim=1)

            action = ddpg.actor(state)                        # I_{t+1} (deterministic, no noise)
            r = step_reward(inventory, action, S[t], S[t+1], cfg.lam)
            total_reward += r.item()

            if i == 0:                                        # record first episode for the scatter
                signals.append(S[t])
                invs.append(action.item())

            inventory = action                                # roll forward: new inventory = action

        episode_rewards.append(total_reward)

# --- scatter: chosen inventory vs signal (acceptance test vs paper Fig 13a) ---
plt.figure(figsize=(7, 5))
plt.scatter(signals, invs, s=4, alpha=0.4)
plt.axvline(1.0, color="gray", ls="--", lw=0.8)
plt.xlabel("signal $S_t$")
plt.ylabel("chosen inventory $I_{t+1}$")
plt.title("Policy: inventory vs signal (compare to paper Fig 13a)")
plt.tight_layout()
plt.savefig("policy_scatter3.png", dpi=130)
plt.close()

episode_rewards = np.array(episode_rewards)
print(f"mean {episode_rewards.mean():.2f} +/- {episode_rewards.std():.2f}")

# --- save rewards + Figure 6 histogram (scenario 2) ---
import os
os.makedirs("artifacts", exist_ok=True)
os.makedirs("figures", exist_ok=True)
np.save(f"artifacts/rewards_scenario{cfg.scenario}_hid.npy", episode_rewards)
plot_reward_histogram(episode_rewards, cfg.scenario, bins=10)