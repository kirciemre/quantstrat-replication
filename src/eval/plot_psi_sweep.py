"""
psi sweep figure (S1): how the inventory penalty psi erodes each variant's reward.
Reads _local_backup/psi_{reg,hid,prob}.csv. eta held at default 0.03; only psi varies.

Left  : eval reward vs psi (3 variants). The aggressive methods (reg, prob) fall
        steeply; hid (already moderate) is flatter. Default psi=0.0005 marked.
Right : mean|I| vs psi -- positions collapse as psi grows (why reward falls).
"""
import csv
import numpy as np
import matplotlib.pyplot as plt

VAR = {"reg": "#1baf7a", "hid": "#eb6834", "prob": "#2a78d6"}


def load(v):
    rows = list(csv.DictReader(open(f"_local_backup/psi_{v}.csv")))
    psi = [float(r["psi"]) for r in rows]
    mean = [float(r["mean"]) for r in rows]
    mi = [float(r["mean_I"]) for r in rows]
    return psi, mean, mi


data = {v: load(v) for v in VAR}
psis = data["reg"][0]
x = np.arange(len(psis))
xlabels = ["0", "2e-4", "5e-4\n(default)", "1e-3", "2e-3"]
default_idx = psis.index(0.0005)

fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.5, 4.8))

for v, c in VAR.items():
    _, mean, mi = data[v]
    axL.plot(x, mean, "-o", color=c, lw=2.2, ms=7, label=v)
    axR.plot(x, mi, "-o", color=c, lw=2.2, ms=7, label=v)
    for xi, m in zip(x, mean):
        axL.annotate(f"{m:.1f}", (xi, m), textcoords="offset points", xytext=(0, 8),
                     ha="center", fontsize=8, color=c)

for ax in (axL, axR):
    ax.axvspan(-0.3, 0.7, color="#2a78d6", alpha=0.06)          # low-friction region
    ax.axvline(default_idx, color="#888", ls="--", lw=1.2)
    ax.set_xticks(x); ax.set_xticklabels(xlabels, fontsize=9)
    ax.set_xlabel("inventory penalty  psi   (eta fixed at 0.03)")
    ax.grid(axis="y", alpha=0.25); ax.legend(fontsize=10)
    ax.set_xlim(-0.35, len(psis) - 0.65)

axL.set_ylabel("eval reward (paper Eq.4)")
axL.set_title("psi erodes reward — aggressive methods (reg/prob) fall fastest", fontsize=12, fontweight="bold")
axR.set_ylabel("mean |I|  (position size)")
axR.set_title("positions collapse as psi grows", fontsize=12, fontweight="bold")
axL.text(0.2, axL.get_ylim()[1] * 0.93, "friction ↓\nranking spreads", fontsize=8.5, color="#2a78d6", ha="center")

fig.suptitle("Scenario 1 — convexity penalty psi flattens the method ranking "
             "(reg≈prob≫hid at psi=0  →  all compressed at default psi)",
             fontsize=12, y=1.02, color="#333")
plt.tight_layout(rect=[0, 0, 1, 0.96])
import os
os.makedirs("figures", exist_ok=True)
plt.savefig("figures/psi_sweep_s1.png", dpi=150, bbox_inches="tight")
print("saved figures/psi_sweep_s1.png")
for v in VAR:
    _, mean, _ = data[v]
    print(f"{v:>4}: " + "  ".join(f"{p}:{m:.1f}" for p, m in zip(psis, mean)))
