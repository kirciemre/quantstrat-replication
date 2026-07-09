"""
Batched training-data factory for the synthetic RL experiments.

This is the "environment" the DDPG agent trains against, but it is deliberately
NOT a Gym-style reset/step environment that walks one agent through a trajectory.
Per Section 3.1 of Macri, Jaimungal & Lillo (2025), training uses fresh batches
of independent random (signal-window, inventory) situations each iteration --
there is no replay buffer and no path dependence, because trades have no market
impact. So this module is a batch FACTORY, not a stateful environment.

  sample_batch : produces a batch of independent training samples
  step_reward  : applies the per-step reward (Eq. 4) across a batch

IMPORTANT (training vs. evaluation): the windows here are short, independent,
non-contiguous paths with RANDOMLY sampled inventories -- correct for training.
Evaluation is the opposite (one long contiguous path, inventory accumulated
forward from I_0 = 0) and will live in eval/evaluate.py. Do NOT evaluate with
sample_batch.
"""

from src.data.ou_simulator import simulate_path
from src.env.reward import reward

import numpy as np


# TODO: batch-vectorize paths if profiling shows this is hot
def sample_batch(batch_size, W, rng, regimes, A, kappa, sigma, dt, I_max):
    """
    Build one training batch of independent samples.

    Args:
      batch_size : number of samples in the batch.
      W          : look-back window length; the GRU reads W+1 signal values.
      rng        : np.random.Generator (from np.random.default_rng(seed)).
      regimes, A, kappa, sigma, dt : OU / regime-chain parameters (see ou_simulator).
      I_max      : inventory bound; training inventories are drawn U[-I_max, I_max].

    Returns a dict of stacked arrays:
      windows     : (batch_size, W+1)  the GRU look-back, S_{t-W} .. S_t
      next_windows: (batch_size, W+1)  the GRU look-back for the next step, S_{t-W+1} .. S_{t+1}
      S_t         : (batch_size,)      current signal (== windows[:, -1])
      S_next      : (batch_size,)      next signal S_{t+1}, for the reward
      regime      : (batch_size,)      active regime index at t (prob-DDPG label)
      I_t         : (batch_size,)      randomly sampled current inventory
    """
    windows = np.zeros((batch_size, W+1))
    next_windows = np.zeros((batch_size, W+1))
    S_t = np.zeros(batch_size)
    S_next = np.zeros(batch_size)
    I_t = np.zeros(batch_size)
    regime = np.zeros(batch_size, dtype=int)

    for i in range(batch_size):
        # Each sample is an independent fresh path of length W+2:
        # indices 0..W are the GRU window (index W = "now"), index W+1 = next step.
        S, regime_index = simulate_path(W+2, rng, regimes, A, kappa, sigma, dt)

        windows[i] = S[:W+1]                       # look-back window (W+1 values)
        next_windows[i] = S[1:W+2]                 # indices 1..W+1 (ends at t+1)
        S_t[i] = S[W]                              # "now" = last element of the window
        S_next[i] = S[W+1]                         # one step ahead, for the reward
        regime[i] = regime_index[W]                # regime active at "now"
        I_t[i] = rng.uniform(-I_max, I_max)        # random inventory (NOT accumulated)

    return {"windows": windows, "next_windows":next_windows, "S_t": S_t, "S_next": S_next, "regime": regime, "I_t": I_t}


def step_reward(I_t, I_next, S_t, S_next, lam):
    """
    Apply the per-step reward (Eq. 4) across a batch.

    Thin wrapper over env.reward.reward; exists as the named "environment step"
    the training loop calls. I_next is the agent's action (new inventory).
    All inputs are batched arrays; returns a (batch_size,) array of rewards.
    """
    return reward(I_t, I_next, S_t, S_next, lam)


if __name__ == "__main__":
    # Quick manual check: print the shape of each field in one batch.
    b = sample_batch(8, 10, np.random.default_rng(0),
                     np.array([0.9, 1.0, 1.1]),
                     np.array([[-0.1, 0.05, 0.05], [0.05, -0.1, 0.05], [0.05, 0.05, -0.1]]),
                     5, 0.2, 0.2, 10)
    print({k: v.shape for k, v in b.items()})