import numpy as np
import torch
import matplotlib.pyplot as plt

from src.data.ou_simulator import simulate_path
from src.env.trading_env import step_reward, normalize_state_features
from src.models.ddpg import DDPG
from src.models.regressor import GRURegressor
from src.utils.config import load_config
from src.utils.ou_args import ou_args_from_cfg


cfg = load_config("configs/scenario1_reg.yaml")
kappa, sigma, ou_kw = ou_args_from_cfg(cfg)


seed = cfg.train_seed if cfg.train_seed is not None else np.random.randint(0, 1_000_000)
rng = np.random.default_rng(seed)
print(f"seed: {seed}")

regressor = GRURegressor(
    hidden_dim=cfg.reg_hidden_dim,
    num_layers=cfg.reg_layers,
    ffn_layers=cfg.reg_ffn_layers,
    ffn_hidden=cfg.reg_ffn_hidden,
)

ddpg = DDPG(
    cfg.state_dim,
    cfg.action_dim,
    cfg.d_NN,
    cfg.l_NN,
    cfg.I_max,
    cfg.gamma,
    cfg.tau,
    cfg.lr,
)

ckpt = torch.load(
    "artifacts/scenario1_reg.pt",
    map_location="cpu",
)

ddpg.actor.load_state_dict(ckpt["actor"])
ddpg.critic.load_state_dict(ckpt["critic"])
regressor.load_state_dict(ckpt["regressor"])

ddpg.actor.eval()
ddpg.critic.eval()
regressor.eval()

print("checkpoint loaded successfully")
print("checkpoint train seed:", ckpt.get("train_seed"))
print("state dimension:", cfg.state_dim)

episode_rewards = []

signals = []
invs = []

with torch.no_grad():
    for i in range(cfg.M):
        S, _ = simulate_path(
            cfg.n,
            rng,
            cfg.regimes,
            cfg.A,
            cfg.dt,
            kappa=kappa,
            sigma=sigma,
            s0=1.0,
            **ou_kw,
        )

        inventory = torch.zeros(1, 1)
        total_reward = 0.0

        for t in range(cfg.W, cfg.n - 1):
            window = torch.tensor(
                S[t - cfg.W:t + 1],
                dtype=torch.float32,
            ).reshape(1, -1)

            S_t_col = torch.tensor(
                [[S[t]]],
                dtype=torch.float32,
            )

            predicted_S_next = regressor(window)

            S_t_norm = normalize_state_features(S_t_col)

            state = torch.cat(
                [S_t_norm, inventory, predicted_S_next],
                dim=1,
            )

            action = ddpg.actor(state)

            reward = step_reward(
                inventory,
                action,
                S[t],
                S[t + 1],
                cfg.lam,
            )

            total_reward += reward.item()

            if i == 0:
                signals.append(S[t])
                invs.append(action.item())

            inventory = action

        episode_rewards.append(total_reward)

episode_rewards = np.array(episode_rewards)

print("episode rewards:", episode_rewards)
print(
    f"mean {episode_rewards.mean():.2f} "
    f"+/- {episode_rewards.std():.2f}"
)
