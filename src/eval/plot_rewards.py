"""
Reward histograms for the hid/prob/reg agents (paper Figures 5, 6, 7 style).

Each figure is the single-method reward histogram over the M test episodes with
the mean marked. Saves to:

    figures/rewards_scenario{n}_{variant}_{model}.png

so hid/prob/reg and ddpg/td3/ppo runs don't overwrite each other.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_reward_histogram(episode_rewards, scenario, variant="hid", model="ppo",
                          out_path=None, bins=50, color="green"):
    """
    Histogram of cumulative rewards over the M test episodes.

    Args:
      episode_rewards : (M,) array of cumulative rewards, one per test episode.
      scenario        : 1, 2, or 3 (used in title + filename).
      variant         : "hid" | "prob" | "reg"  (filename + legend).
      model           : "ddpg" | "td3" | "ppo"  (filename + legend).
      out_path        : override; default figures/rewards_scenario{n}_{variant}_{model}.png
      bins, color     : styling.
    """
    episode_rewards = np.asarray(episode_rewards)
    mean = episode_rewards.mean()
    std = episode_rewards.std()

    mc_desc = {
        1: r"$\theta$ modelled as a MC",
        2: r"$\theta, \kappa$ modelled as a MC",
        3: r"$\theta, \kappa, \sigma$ modelled as a MC",
    }[scenario]

    label = f"{variant}-{model}"
    plt.figure(figsize=(7, 5))
    plt.hist(episode_rewards, bins=bins, alpha=0.6, color=color,
             label=f"{label} (Mean: {mean:.2f})")
    plt.axvline(mean, color=color, linestyle="dashed", linewidth=1.8)
    plt.title(f"Rewards over {len(episode_rewards)} episodes, {mc_desc}")
    plt.xlabel("Cumulative rewards")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if out_path is None:
        out_path = f"figures/rewards_scenario{scenario}_{variant}_{model}.png"
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"saved {out_path}")
    print(f"  mean {mean:.2f} +/- {std:.2f}  (n={len(episode_rewards)} episodes)")
    return mean, std