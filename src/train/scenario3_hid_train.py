import numpy as np
import torch

from src.env.trading_env import sample_batch, step_reward, normalize_state_features
from src.models.gru import GRUEncoder
from src.models.ddpg import DDPG
from src.utils.config import load_config
from src.utils.ou_args import ou_args_from_cfg


cfg = load_config("configs/scenario3_hid.yaml")
kappa, sigma, ou_kw = ou_args_from_cfg(cfg)

# --- reproducibility (numpy drives the simulator; torch drives net init + noise) ---
seed = cfg.train_seed if cfg.train_seed is not None else np.random.randint(0, 1_000_000)
rng = np.random.default_rng(seed)
torch.manual_seed(seed)
print(f"seed: {seed}")

epsilon = 1.0

# --- networks: encoder is OWNED HERE (trained separately from the RL step) ---
encoder = GRUEncoder(cfg.d_h, cfg.d_l, enc_dim=cfg.enc_dim)
gru_opt = torch.optim.AdamW(encoder.parameters(), lr=cfg.lr, weight_decay=1e-5)   # GRU's own optimizer

ddpg = DDPG(cfg.state_dim, cfg.action_dim, cfg.d_NN, cfg.l_NN, cfg.I_max, cfg.gamma, cfg.tau, cfg.lr)          # no encoder in DDPG

for m in range(cfg.N):
    # --- obtain ONE batch per iteration (Alg.1 line 4); both procedures reuse it ---
    batch = sample_batch(cfg.batch_size, cfg.W, rng, cfg.regimes, cfg.A, kappa, sigma, cfg.dt, cfg.I_max, **ou_kw)
    windows_tensor = torch.tensor(batch["windows"], dtype=torch.float32)
    next_windows_tensor = torch.tensor(batch["next_windows"], dtype=torch.float32)
    S_t_col = torch.tensor(batch["S_t"], dtype=torch.float32).reshape(-1, 1)
    I_t_col = torch.tensor(batch["I_t"], dtype=torch.float32).reshape(-1, 1)
    S_next_col = torch.tensor(batch["S_next"], dtype=torch.float32).reshape(-1, 1)

    # === GRU procedure (Alg.1 lines 6-8): train the GRU FIRST, before Actor/Critic ===
    # aux task: predict S_{t+1} from the window via the LeakyReLU head; MSE loss.
    o_t, s_pred = encoder(windows_tensor)
    gru_loss = torch.nn.functional.mse_loss(s_pred, S_next_col)
    gru_opt.zero_grad()
    gru_loss.backward()
    gru_opt.step()

    # re-extract encodings DETACHED (the RL step treats o_t as a fixed feature;
    # the GRU is trained only by its aux loss above, not by RL gradients).
    with torch.no_grad():
        o_t, _ = encoder(windows_tensor)
        o_next, _ = encoder(next_windows_tensor)

    # --- normalize features (S, I -> [0,1]); windows/reward stay raw ---
    S_t_norm = normalize_state_features(S_t_col)

    # --- assemble current state G_t = (S_t, I_t, o_t) ---
    state = torch.cat([S_t_norm, I_t_col, o_t], dim=1)          # (b, 3)

    # --- executed action I_{t+1} = pi(G_t) + N(0,eps), clamped (this IS the new inventory) ---
    action = ddpg.select_action(state, epsilon)                 # (b, 1), detached+clamped

    # --- reward (Eq. 4), RAW signal/inventory; action is the new inventory I_{t+1} ---
    reward = step_reward(I_t_col, action, S_t_col, S_next_col, cfg.lam)

    # --- assemble next state G'_{t+1} = (S_{t+1}, I_{t+1}, o_{t+1}) ---
    S_next_norm = normalize_state_features(S_next_col)
    next_state = torch.cat([S_next_norm, action, o_next], dim=1)   # (b, 3)

    # === Update Critic (Alg.1 lines 9-23): ell gradient steps on this batch ===
    for _ in range(cfg.ell):
        critic_loss = ddpg.update_critic(state, action, reward, next_state)
        ddpg.soft_update(ddpg.critic_target, ddpg.critic)

    # === Update Actor (Alg.1 lines 24-34): l gradient steps on this batch ===
    for _ in range(cfg.l):
        actor_loss = ddpg.update_actor(state)
        ddpg.soft_update(ddpg.actor_target, ddpg.actor)

    if m % cfg.log_every == 0:
        print(f"iter {m:5d} | critic {critic_loss:.4f} | actor {actor_loss:.4f} "
              f"| gru {gru_loss.item():.4f} | eps {epsilon:.3f}")
        
    epsilon = max(cfg.eps_a / (cfg.eps_a + (m + 1)), cfg.eps_min)

# --- save trained weights (actor + critic + encoder) ---
import os
os.makedirs("artifacts", exist_ok=True)
torch.save({"actor": ddpg.actor.state_dict(),
            "critic": ddpg.critic.state_dict(),
            "encoder": encoder.state_dict()}, "artifacts/scenario3_hid.pt")