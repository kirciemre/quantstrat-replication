"""
Seed-distribution strip plot + robust stats for the three-method sweep.

Reads the saved per-seed reward arrays and, for each (method, scenario) cell,
computes the per-seed mean (one point per seed). Plots those points as a strip
plot (3 scenario panels x 3 methods) with the mean-of-means marked, and prints
mean AND median across seeds (median is robust to the outlier seeds).

Outputs: figures/seed_distribution.png  + a printed table (incl. JS array for the deck).
"""

import os
import numpy as np
import matplotlib.pyplot as plt

METHODS = ["reg", "hid", "prob"]
COL = {"reg": "#1baf7a", "hid": "#eb6834", "prob": "#2a78d6"}
PAPER = {
    "reg":  {1: 8.69, 2: 2.95, 3: -0.07},
    "hid":  {1: 15.70, 2: 8.08, 3: 1.29},
    "prob": {1: 25.65, 2: 15.59, 3: 4.51},
}


def files(method, scen):
    if method == "reg":
        base = f"artifacts/sweep_s{scen}_seed"
    elif method == "hid":
        base = f"artifacts/sweep_hid_s{scen}_baseline_seed"
    else:
        base = f"artifacts/sweep_prob_s{scen}_seed"
    out = []
    for k in range(5):
        p = f"{base}{k}.npy"
        if os.path.exists(p):
            out.append(np.load(p))
    return out


seed_means = {}   # (method, scen) -> list of per-seed means
for m in METHODS:
    for s in (1, 2, 3):
        arrs = files(m, s)
        seed_means[(m, s)] = [float(a.mean()) for a in arrs]

# --- print table + JS array for the HTML deck ---
print("method  scen  n  mean-of-means  median  seeds")
js = {}
for m in METHODS:
    js[m] = {}
    for s in (1, 2, 3):
        v = np.array(seed_means[(m, s)])
        js[m][s] = [round(x, 2) for x in v]
        print(f"{m:5s}  S{s}  {len(v)}  {v.mean():7.2f}      {np.median(v):7.2f}  "
              f"{[round(x,1) for x in v]}")
print("\nJS_SEED_MEANS =", str(js).replace("'", ""))

# --- strip plot: 3 panels, x = method, y = per-seed mean ---
fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=False)
titles = {1: r"S1: $\theta$", 2: r"S2: $\theta,\kappa$", 3: r"S3: $\theta,\kappa,\sigma$"}
rng = np.random.default_rng(0)
for ax, s in zip(axes, (1, 2, 3)):
    for xi, m in enumerate(METHODS):
        v = np.array(seed_means[(m, s)])
        jitter = (rng.random(len(v)) - 0.5) * 0.22
        ax.scatter(np.full(len(v), xi) + jitter, v, s=70, color=COL[m],
                   alpha=0.8, edgecolor="white", linewidth=1.2, zorder=3)
        ax.hlines(v.mean(), xi - 0.28, xi + 0.28, color=COL[m], linewidth=2.5, zorder=4)
        ax.plot(xi, PAPER[m][s], marker="_", markersize=20, color="black",
                markeredgewidth=2.2, zorder=5)
    ax.set_title(titles[s], fontsize=12)
    ax.set_xticks(range(len(METHODS)))
    ax.set_xticklabels(METHODS)
    ax.axhline(0, color="#bbb", lw=0.8, ls="--")
    ax.grid(axis="y", alpha=0.25)
    ax.set_xlim(-0.6, len(METHODS) - 0.4)
axes[0].set_ylabel("cumulative reward (per-seed mean)")
# legend proxy
from matplotlib.lines import Line2D
handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=COL[m],
                  markersize=10, label=m) for m in METHODS]
handles.append(Line2D([0], [0], marker="_", color="black", markeredgewidth=2.2,
                      markersize=14, linestyle="None", label="paper (Table 4)"))
fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False,
           bbox_to_anchor=(0.5, 1.04), fontsize=10)
plt.tight_layout(rect=[0, 0, 1, 0.92])
os.makedirs("figures", exist_ok=True)
plt.savefig("figures/seed_distribution.png", dpi=150, bbox_inches="tight")
print("\nsaved figures/seed_distribution.png")
