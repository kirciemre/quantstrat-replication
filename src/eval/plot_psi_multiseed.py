"""
Multi-seed psi sweep figure (the reliable version): reads _local_backup/task1_full.csv
(3 scenarios x 5 psi x 3 variants x 3 seeds) and plots reward vs psi, one panel per
scenario, with error bars (std over seeds). Shows prob on top in every panel/friction.
"""
import csv
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

COL = {"reg": "#1baf7a", "hid": "#eb6834", "prob": "#2a78d6"}
PSIS = [0.0, 0.0002, 0.0005, 0.001, 0.002]
XL = ["0", "2e-4", "5e-4\n(default)", "1e-3", "2e-3"]

agg = defaultdict(list)
for r in csv.DictReader(open("_local_backup/task1_full.csv")):
    if r.get("mean"):
        agg[(int(r["scenario"]), float(r["psi"]), r["variant"])].append(float(r["mean"]))

fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), sharex=True)
titles = {1: r"S1: $\theta$", 2: r"S2: $\theta,\kappa$", 3: r"S3: $\theta,\kappa,\sigma$"}
x = np.arange(len(PSIS))
for ax, sc in zip(axes, (1, 2, 3)):
    for v in ("reg", "hid", "prob"):
        m = np.array([np.mean(agg[(sc, p, v)]) for p in PSIS])
        s = np.array([np.std(agg[(sc, p, v)]) for p in PSIS])
        ax.errorbar(x, m, yerr=s, marker="o", ms=6, lw=2.2, color=COL[v],
                    capsize=3, label=v, zorder=3 if v == "prob" else 2)
    ax.axvline(2, color="#888", ls="--", lw=1.1)          # default psi
    ax.axvspan(-0.3, 0.5, color="#2a78d6", alpha=0.05)
    ax.set_title(titles[sc], fontsize=12)
    ax.set_xticks(x); ax.set_xticklabels(XL, fontsize=8.5)
    ax.set_xlabel("inventory penalty  psi  (eta=0.03)")
    ax.grid(axis="y", alpha=0.25); ax.legend(fontsize=9.5, loc="upper right")
axes[0].set_ylabel("eval reward (paper Eq.4)")
fig.suptitle("Multi-seed psi sweep (3 seeds ± std) — prob (blue) is on top in every scenario "
             "at every friction level; friction only compresses & swaps 2nd/3rd",
             fontsize=12, y=1.02, color="#333")
plt.tight_layout(rect=[0, 0, 1, 0.94])
import os
os.makedirs("figures", exist_ok=True)
plt.savefig("figures/psi_sweep_multiseed.png", dpi=150, bbox_inches="tight")
print("saved figures/psi_sweep_multiseed.png")
# print the ranking table
for sc in (1, 2, 3):
    print(f"S{sc}:", "  ".join(
        f"psi{p}:" + ">".join(sorted(("prob","hid","reg"), key=lambda v:-np.mean(agg[(sc,p,v)])))
        for p in PSIS))
