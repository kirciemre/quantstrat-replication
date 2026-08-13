"""
Slide-4 figure: variance collapse under H1 stabilization.

The headline of H1 is that the CROSS-SEED *relative* variance (std/mean) collapses
from 96% (baseline) to 11% (H1), matching the paper's 10% -- even though H1's
absolute mean is much higher. A naive std bar chart would mislead (H1's absolute
std is larger), so we show:

  Left  : relative variance (CV = std/mean, %). The headline: 96% -> 11%.
  Right : per-seed values normalized to each group's own mean. Baseline dots
          scatter wildly (0.04-2.8x); H1 dots sit tight around 1.0 -> "the seeds
          agree now". Paper's +/-10% band shown for reference.

Mean values are annotated so nobody mistakes "11% ~ 10%" for "matched the paper"
(H1 mean 76 vs paper 3).

Usage: python -m src.eval.plot_variance_collapse
"""

import numpy as np
import matplotlib.pyplot as plt

def seed_means(prefix):
    out = []
    for k in range(5):
        p = f"artifacts/{prefix}{k}.npy"
        try:
            out.append(float(np.load(p).mean()))
        except FileNotFoundError:
            pass
    return np.array(out)

base = seed_means("sweep_s2_seed")            # baseline S2 reg
h1 = seed_means("sweep_s2_stableH1_seed")     # H1 stabilized S2
paper_mean, paper_std = 2.95, 0.30

groups = ["baseline", "H1 stabilized", "paper"]
means = [base.mean(), h1.mean(), paper_mean]
stds = [base.std(), h1.std(), paper_std]
cv = [100 * s / m for s, m in zip(stds, means)]
colors = ["#d13c3a", "#128a5b", "#7c8494"]    # red=problem, green=fixed, gray=ref

fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.5, 5))

# --- Panel A: relative variance (CV %) ---
bars = axA.bar(range(3), cv, color=colors, width=0.62, edgecolor="white", linewidth=1.5)
for i, (b, c) in enumerate(zip(bars, cv)):
    axA.text(b.get_x() + b.get_width() / 2, c + 2, f"{c:.0f}%",
             ha="center", va="bottom", fontsize=15, fontweight="bold", color=colors[i])
axA.set_xticks(range(3))
axA.set_xticklabels([f"{g}\n(mean {m:.1f})" for g, m in zip(groups, means)], fontsize=11)
axA.set_ylabel("relative variance  std / mean  (%)", fontsize=12)
axA.set_title(f"Cross-seed variance collapses:  {cv[0]:.0f}% → {cv[1]:.0f}%", fontsize=13, fontweight="bold")
axA.set_ylim(0, 112)
axA.grid(axis="y", alpha=0.25)
axA.annotate("", xy=(1, 20), xytext=(0, 92),
             arrowprops=dict(arrowstyle="->", color="#444", lw=1.8,
                             connectionstyle="arc3,rad=-0.25"))

# --- Panel B: per-seed values normalized to each group's mean ---
rng = np.random.default_rng(0)
for xi, (vals, col) in enumerate([(base, colors[0]), (h1, colors[1])]):
    norm = vals / vals.mean()
    jit = (rng.random(len(norm)) - 0.5) * 0.18
    axB.scatter(np.full(len(norm), xi) + jit, norm, s=90, color=col,
                alpha=0.85, edgecolor="white", linewidth=1.4, zorder=3)
axB.axhline(1.0, color="#444", lw=1.2, ls="-", zorder=1)
axB.axhspan(1 - paper_std / paper_mean, 1 + paper_std / paper_mean,
            color="#7c8494", alpha=0.18, zorder=0, label="paper ±10% band")
axB.set_xticks([0, 1])
axB.set_xticklabels(["baseline\n(seeds scatter)", "H1\n(seeds agree)"], fontsize=11)
axB.set_ylabel("per-seed reward / group mean", fontsize=12)
axB.set_title("Same seeds, normalized: baseline flies apart, H1 clusters", fontsize=13, fontweight="bold")
axB.set_xlim(-0.5, 1.5)
axB.grid(axis="y", alpha=0.25)
axB.legend(fontsize=10, loc="upper right")

fig.suptitle("H1 stabilization — variance is a training-instability artifact "
             "(but H1 converges high, not to the paper)", fontsize=12.5, y=1.02, color="#333")
plt.tight_layout(rect=[0, 0, 1, 0.96])
import os
os.makedirs("figures", exist_ok=True)
plt.savefig("figures/variance_collapse.png", dpi=150, bbox_inches="tight")
print(f"baseline: mean {base.mean():.2f} std {base.std():.2f} CV {cv[0]:.0f}%  seeds {np.round(base,1)}")
print(f"H1:       mean {h1.mean():.2f} std {h1.std():.2f} CV {cv[1]:.0f}%  seeds {np.round(h1,1)}")
print("saved figures/variance_collapse.png")
