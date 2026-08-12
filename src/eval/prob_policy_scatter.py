"""
prob-DDPG policy scatter (reproduce paper Fig 13c / Fig 14 qualitatively).

Trains one prob-DDPG (classifier + DDPG) and rolls out one long test path,
recording (S_t, I_{t+1}, posterior). Produces a two-panel figure:

  Left  : inventory vs signal, colored by the most-likely theta regime (argmax
          posterior). If the policy conditions on the regime, points at the SAME
          signal split by color -- the paper's key interpretability claim.
  Right : inventory vs signal, colored by P(theta = 0.9) (the low regime), a
          continuous view of the same effect.

Uses S2 (theta, kappa switching) with a moderate seed so positions are not pinned
at the corner (which would hide the structure).

Usage: python -m src.eval.prob_policy_scatter
"""

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.data.ou_simulator import simulate_path
from src.env.trading_env import normalize_state_features
from src.utils.config import load_config
from src.utils.ou_args import ou_args_from_cfg
from src.eval.sweep_prob import pretrain_classifier, train_ddpg

SCEN = 2
SEED = 4        # moderate seed (reward ~26, mean|I| ~2.5) -> shows structure

cfg = load_config(f"configs/scenario{SCEN}_prob.yaml")
kappa, sigma, ou_kw = ou_args_from_cfg(cfg)
num_classes = len(cfg.regimes)

print("pretraining classifier...", flush=True)
clf, acc = pretrain_classifier(cfg, kappa, sigma, ou_kw, cfg.clf_pretrain_steps, num_classes)
print(f"classifier acc {acc:.3f}; training DDPG (seed {SEED})...", flush=True)
ddpg = train_ddpg(cfg, kappa, sigma, ou_kw, clf, SEED, cfg.N)

# --- roll out several long test paths, record policy + posterior ---
rng = np.random.default_rng(999)
S_list, I_list, post_list = [], [], []
with torch.no_grad():
    for _ in range(6):
        S, _ = simulate_path(cfg.n, rng, cfg.regimes, cfg.A, cfg.dt,
                             kappa=kappa, sigma=sigma, s0=1.0, **ou_kw)
        inv = torch.zeros(1, 1)
        for t in range(cfg.W, cfg.n - 1):
            w = torch.tensor(S[t - cfg.W:t + 1], dtype=torch.float32).reshape(1, -1)
            phi = clf.posterior(w)                       # (1, K)
            st = normalize_state_features(torch.tensor([[S[t]]], dtype=torch.float32))
            a = ddpg.actor(torch.cat([st, inv, phi], dim=1))
            S_list.append(S[t]); I_list.append(a.item()); post_list.append(phi.numpy().ravel())
            inv = a

S_arr = np.array(S_list); I_arr = np.array(I_list); P = np.array(post_list)
argmax_reg = P.argmax(1)
regime_names = [f"θ={r}" for r in cfg.regimes]
reg_colors = ["#2a78d6", "#eda100", "#e34948"]   # low / mid / high theta

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

# panel 1: colored by argmax regime
ax = axes[0]
for k in range(num_classes):
    m = argmax_reg == k
    ax.scatter(S_arr[m], I_arr[m], s=10, alpha=0.4, color=reg_colors[k], label=regime_names[k])
ax.axvline(1.0, color="gray", ls="--", lw=0.8)
ax.axhline(0.0, color="gray", ls=":", lw=0.8)
ax.set_xlabel("signal $S_t$"); ax.set_ylabel("chosen inventory $I_{t+1}$")
ax.set_title("prob-DDPG policy, colored by most-likely regime")
ax.legend(title="argmax posterior", fontsize=9)

# panel 2: continuous P(theta=0.9)
ax = axes[1]
sc = ax.scatter(S_arr, I_arr, s=10, alpha=0.5, c=P[:, 0], cmap="viridis", vmin=0, vmax=1)
ax.axvline(1.0, color="gray", ls="--", lw=0.8)
ax.axhline(0.0, color="gray", ls=":", lw=0.8)
ax.set_xlabel("signal $S_t$"); ax.set_ylabel("chosen inventory $I_{t+1}$")
ax.set_title("colored by posterior $P(\\theta=0.9)$")
fig.colorbar(sc, ax=ax, label="$P(\\theta=0.9)$")

fig.suptitle(f"prob-DDPG (scenario {SCEN}, θ,κ) — actions condition on the regime posterior",
             y=1.02, fontsize=12)
plt.tight_layout()
import os
os.makedirs("figures", exist_ok=True)
plt.savefig("figures/prob_policy_scatter.png", dpi=150, bbox_inches="tight")
print("saved figures/prob_policy_scatter.png", flush=True)
print(f"points={len(S_arr)}  mean|I|={np.abs(I_arr).mean():.2f}  "
      f"regime split={[int((argmax_reg==k).sum()) for k in range(num_classes)]}", flush=True)
