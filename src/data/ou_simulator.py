"""
Regime-switching Ornstein-Uhlenbeck (OU) signal simulator.

Reproduces the synthetic data-generating process from Section 2 of
Macri, Jaimungal & Lillo (2025), "Deep reinforcement learning for optimal
trading with partial information" -- the simplest setting where only the
mean-reversion level theta switches between regimes (Table 1, row 1).

The signal follows the OU SDE (Eq. 1):
    dS_t = kappa * (theta_t - S_t) dt + sigma * dW_t

theta_t switches among discrete regimes (e.g. {0.9, 1.0, 1.1}) via a
continuous-time Markov chain. kappa, sigma constant in this first stage.

Initialization (paper): the signal starts at S_{t-W} ~ N(mu_inv, 3*sigma_inv),
where sigma_inv = sigma / sqrt(2*kappa) is the OU invariant volatility and
mu_inv is the invariant mean. For the TEST phase the paper instead fixes
S_0 = 1, so simulate_path accepts an optional s0 to override the random draw.
"""

import numpy as np
from scipy.linalg import expm


def simulate_path(n: int, rng: np.random.Generator, regimes: np.ndarray,
                  A: np.ndarray, kappa: float, sigma: float, dt: float,
                  mu_inv: float = 1.0, s0: float = None):
  """
  Simulate one path of the regime-switching OU signal.

  Args:
    n       : number of time steps.
    rng     : NumPy Generator (seed controlled at the call site).
    regimes : possible theta values, e.g. np.array([0.9, 1.0, 1.1]).
    A       : Markov generator (rate) matrix for regime switching.
    kappa, sigma, dt : OU parameters.
    mu_inv  : invariant mean for the training init (paper: 1.0).
    s0      : if given, start the signal at this fixed value (TEST phase, S_0=1);
              if None, draw S[0] ~ N(mu_inv, 3*sigma_inv) (TRAIN phase).

  Returns:
    S          : (n,) signal values.
    regime_idx : (n,) active regime index at each step.
  """
  # One-step regime transition probabilities (Eq. 5): matrix-exponentiate the
  # rate matrix over dt. Must be scipy.linalg.expm, NOT np.exp.
  P = expm(A * dt)

  # Exact OU discretization constants (no error at any step size).
  a = np.exp(-kappa * dt)
  b = sigma * np.sqrt((1 - np.exp(-2 * kappa * dt)) / (2 * kappa))

  S = np.zeros(n)
  regime_idx = np.zeros(n, dtype=int)

  # Regime at step 0: no previous regime to transition from, so pick uniformly.
  row = rng.integers(len(regimes))
  regime_idx[0] = row

  # Signal at step 0: paper init N(mu_inv, 3*sigma_inv) for training,
  # or the fixed s0 (=1) for the test phase.
  sigma_inv = sigma / np.sqrt(2 * kappa)      # OU invariant volatility
  S[0] = rng.normal(mu_inv, 3 * sigma_inv) if s0 is None else s0

  for i in range(1, n):
    # Advance the Markov chain; `row` is the state, so write the result back.
    row = rng.choice(len(regimes), p=P[row])
    theta = regimes[row]

    S[i] = theta + (S[i-1] - theta) * a + b * rng.standard_normal()
    regime_idx[i] = row

  return S, regime_idx