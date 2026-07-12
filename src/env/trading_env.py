"""
Vectorized batch simulation for TRAINING.

The per-sample `simulate_path` loop was 87% of training time (profiled): it called
rng.choice ~338k times and recomputed expm(A*dt) once per sample (a constant!).
This module simulates the WHOLE batch at once with array ops:

  - transition matrices expm(A*dt) computed ONCE, not per sample
  - all `batch_size` Markov chains advanced in one vectorized step
  - all `batch_size` OU updates done in one vectorized step

Interface matches the old sample_batch (same returned dict), so nothing
downstream changes. Eval still uses the scalar simulate_path (one long contiguous
path, only M times -- not a bottleneck).
"""

import numpy as np
from scipy.linalg import expm

from src.env.reward import reward


# --- feature normalization constants (single source of truth) ---
S_MIN = 0.5
S_MAX = 1.5


def normalize_state_features(S, s_min=S_MIN, s_max=S_MAX):
    """Normalize the SIGNAL to [0,1] (clipped). Inventory stays RAW."""
    S_norm = (S - s_min) / (s_max - s_min)
    if hasattr(S_norm, "clamp"):
        S_norm = S_norm.clamp(0.0, 1.0)
    else:
        S_norm = np.clip(S_norm, 0.0, 1.0)
    return S_norm


def _step_chains(rows, P, rng):
    """
    Advance a whole batch of Markov chains one step, vectorized.

    rows : (batch,) int array of current regime indices
    P    : (n_states, n_states) one-step transition matrix
    returns : (batch,) int array of next regime indices
    """
    cum = P[rows].cumsum(axis=1)                    # (batch, n_states)
    u = rng.random((rows.shape[0], 1))              # (batch, 1)
    return (u > cum).sum(axis=1).clip(0, P.shape[1] - 1)


def sample_batch(batch_size, W, rng, regimes, A, kappa, sigma, dt, I_max, **ou_kwargs):
    """
    Build one training batch of independent samples -- VECTORIZED.

    Same signature and same returned dict as the original scalar version.
    Scenario is implied by which OU params are given (see simulate_path):
      Scenario 1: kappa=5, sigma=0.2
      Scenario 2: kappa=None, sigma=0.2, regimes_kappa=..., A_kappa=...
      Scenario 3: kappa=None, sigma=None, + regimes_sigma=..., A_sigma=...

    Returns dict of stacked arrays:
      windows      : (batch_size, W+1)   S_{t-W} .. S_t
      next_windows : (batch_size, W+1)   S_{t-W+1} .. S_{t+1}
      S_t          : (batch_size,)
      S_next       : (batch_size,)
      regime       : (batch_size,)       theta regime index at t
      I_t          : (batch_size,)       random inventory U[-I_max, I_max]
    """
    regimes_kappa = ou_kwargs.get("regimes_kappa")
    A_kappa = ou_kwargs.get("A_kappa")
    regimes_sigma = ou_kwargs.get("regimes_sigma")
    A_sigma = ou_kwargs.get("A_sigma")

    # --- validate: exactly one form per parameter (mirrors simulate_path) ---
    assert (kappa is None) != (regimes_kappa is None), \
        "provide exactly one of kappa (constant) or regimes_kappa (switching)"
    assert (sigma is None) != (regimes_sigma is None), \
        "provide exactly one of sigma (constant) or regimes_sigma (switching)"
    if regimes_kappa is not None:
        assert A_kappa is not None, "regimes_kappa requires A_kappa"
    if regimes_sigma is not None:
        assert A_sigma is not None, "regimes_sigma requires A_sigma"

    kappa_switches = regimes_kappa is not None
    sigma_switches = regimes_sigma is not None

    n = W + 2                                    # path length per sample

    # --- transition matrices: computed ONCE (was 512x redundant before) ---
    P_theta = expm(A * dt)
    P_kappa = expm(A_kappa * dt) if kappa_switches else None
    P_sigma = expm(A_sigma * dt) if sigma_switches else None

    # --- init uses MINIMUM kappa/sigma across regimes (paper's rule) ---
    kappa_min = float(np.min(regimes_kappa)) if kappa_switches else kappa
    sigma_min = float(np.min(regimes_sigma)) if sigma_switches else sigma
    sigma_inv = sigma_min / np.sqrt(2 * kappa_min)

    S = np.zeros((batch_size, n))
    theta_idx = np.zeros((batch_size, n), dtype=int)

    # --- step 0: random regimes + N(mu_inv, 3*sigma_inv) signal, all at once ---
    th_rows = rng.integers(len(regimes), size=batch_size)
    k_rows = rng.integers(len(regimes_kappa), size=batch_size) if kappa_switches else None
    s_rows = rng.integers(len(regimes_sigma), size=batch_size) if sigma_switches else None
    theta_idx[:, 0] = th_rows
    S[:, 0] = rng.normal(1.0, 3 * sigma_inv, size=batch_size)

    # --- evolve the whole batch, one timestep at a time (vectorized across batch) ---
    for t in range(1, n):
        th_rows = _step_chains(th_rows, P_theta, rng)
        theta_t = regimes[th_rows]                          # (batch,)

        if kappa_switches:
            k_rows = _step_chains(k_rows, P_kappa, rng)
            kappa_t = regimes_kappa[k_rows]                 # (batch,)
        else:
            kappa_t = kappa                                 # scalar

        if sigma_switches:
            s_rows = _step_chains(s_rows, P_sigma, rng)
            sigma_t = regimes_sigma[s_rows]                 # (batch,)
        else:
            sigma_t = sigma                                 # scalar

        # OU discretization constants for THIS step (arrays if switching)
        a = np.exp(-kappa_t * dt)
        b = sigma_t * np.sqrt((1 - np.exp(-2 * kappa_t * dt)) / (2 * kappa_t))

        S[:, t] = theta_t + (S[:, t-1] - theta_t) * a + b * rng.standard_normal(batch_size)
        theta_idx[:, t] = th_rows

    return {
        "windows":      S[:, :W+1],              # (batch, W+1)
        "next_windows": S[:, 1:W+2],             # (batch, W+1)
        "S_t":          S[:, W],                 # (batch,)
        "S_next":       S[:, W+1],               # (batch,)
        "regime":       theta_idx[:, W],         # (batch,)
        "I_t":          rng.uniform(-I_max, I_max, size=batch_size),
    }


def step_reward(I_t, I_next, S_t, S_next, lam):
    """Per-step reward (Eq. 4) across a batch. RAW signal/inventory (real P&L)."""
    return reward(I_t, I_next, S_t, S_next, lam)