"""
Reward histograms for hid-DDPG (paper Figures 5, 6, 7 -- single-method version).

Each of the paper's Figs 5/6/7 is a 3x3 grid comparing the three algorithms for
one scenario. For hid-DDPG alone you want the single "hid-DDPG" panel: a
histogram of the M=500 test-episode cumulative rewards with the mean marked.

Two ways to use this:

  1. Call plot_reward_histogram(...) at the end of your existing eval script,
     passing the episode_rewards array it already computes.

  2. Run this file directly to plot from a saved .npy of episode rewards
     (if you save them from eval).

Save episode_rewards from eval with:
    np.save(f"artifacts/rewards_scenario{cfg.scenario}_hid.npy", episode_rewards)
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_reward_histogram(episode_rewards, scenario, out_path=None,
                          bins=50, color="green", label="hid-DDPG"):
    """
    Histogram of cumulative rewards over the M test episodes (paper Fig 5/6/7 style).

    Args:
      episode_rewards : (M,) array of cumulative rewards, one per test episode.
      scenario        : 1, 2, or 3 (used in the title).
      out_path        : where to save the png; defaults to a scenario-based name.
      bins, color     : histogram styling (paper uses green for hid-DDPG).
    """
    episode_rewards = np.asarray(episode_rewards)
    mean = episode_rewards.mean()
    std = episode_rewards.std()

    # Scenario -> the paper's description of the data-generating process
    mc_desc = {
        1: r"$\theta$ modelled as a MC",
        2: r"$\theta, \kappa$ modelled as a MC",
        3: r"$\theta, \kappa, \sigma$ modelled as a MC",
    }[scenario]

    plt.figure(figsize=(7, 5))
    plt.hist(episode_rewards, bins=bins, alpha=0.6, color=color,
             label=f"{label} (Mean: {mean:.2f})")
    plt.axvline(mean, color=color, linestyle="dashed", linewidth=1.8,
                label=f"Mean ({label})")
    plt.title(f"Histogram of rewards for {len(episode_rewards)} episodes "
              f"with {mc_desc}")
    plt.xlabel("Cumulative rewards")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if out_path is None:
        out_path = f"figures/rewards_scenario{scenario}_hid.png"
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"saved {out_path}")
    print(f"  mean {mean:.2f} +/- {std:.2f}  (n={len(episode_rewards)} episodes)")
    return mean, std


if __name__ == "__main__":
    # Plot from saved reward arrays, if you saved them in eval.
    import os
    os.makedirs("figures", exist_ok=True)

    for scenario in (1, 2, 3):
        path = f"artifacts/rewards_scenario{scenario}_hid.npy"
        if os.path.exists(path):
            rewards = np.load(path)
            plot_reward_histogram(rewards, scenario)
        else:
            print(f"skip scenario {scenario}: {path} not found "
                  f"(save episode_rewards from eval first)")