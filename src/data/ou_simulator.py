"""
Regime-switching Ornstein-Uhlenbeck (OU) signal simulator.

Reproduces the synthetic data-generating process from Section 2 of
Macri, Jaimungal & Lillo (2025), "Deep reinforcement learning for optimal
trading with partial information" -- specifically the simplest setting where
only the mean-reversion level theta switches between regimes (Table 1, row 1).

The signal follows the OU stochastic differential equation (paper Eq. 1):

    dS_t = kappa * (theta_t - S_t) dt + sigma * dW_t

  - kappa  : speed of mean reversion (how hard S is pulled toward theta)
  - theta  : the long-run mean S reverts to; here it JUMPS between regimes
  - sigma  : volatility (size of the random shocks)
  - dW_t   : Brownian increment (the randomness)

theta_t switches among a small set of discrete regimes (e.g. {0.9, 1.0, 1.1})
according to a continuous-time Markov chain. kappa and sigma are constant in
this first stage (later stages let them switch too).

This module is the ENVIRONMENT the RL agent trades against in the synthetic
experiments (paper Sections 3-4). Nothing downstream -- the GRU, the DDPG
agent, the reward -- can run without a signal to trade, so this is built first.
"""

import numpy as np
from scipy.linalg import expm


def simulate_path(n: int, rng: np.random.Generator, regimes: np.ndarray,
                  A: np.ndarray, kappa: float, sigma: float, dt: float):
  """
  Simulate one path of the regime-switching OU signal.

  Args:
    n       : number of time steps to simulate.
    rng     : a NumPy Generator (np.random.default_rng(seed)). Passed in by the
              caller so the seed is controlled at the call site, not hidden here.
    regimes : 1-D array of the possible theta values, e.g. np.array([0.9, 1.0, 1.1]).
    A       : the Markov chain GENERATOR (rate) matrix for regime switching.
              Off-diagonal A[i,j] = rate of jumping i -> j; each row sums to 0.
    kappa   : mean-reversion speed.
    sigma   : volatility.
    dt      : time step size.

  Returns:
    S          : (n,) array of signal values.
    regime_idx : (n,) integer array of the active regime index at each step.
                 These labels are free in synthetic data and become the
                 supervised targets for the prob-DDPG classifier later on.
  """
  # One-step regime transition probabilities (paper Eq. 5): exponentiating the
  # rate matrix over dt turns continuous-time rates into probabilities.
  # Must be the MATRIX exponential (scipy.linalg.expm), NOT np.exp.
  P = expm(A * dt)

  # Exact OU discretization constants (no error at any step size):
  #   a = fraction of the gap to theta retained per step (1 - a is closed)
  #   b = per-step noise std, calibrated to the true stationary width
  a = np.exp(-kappa * dt)
  b = sigma * np.sqrt((1 - np.exp(-2 * kappa * dt)) / (2 * kappa))

  S = np.zeros(n)
  regime_idx = np.zeros(n, dtype=int)

  # Step 0 uses rng.integers (no previous regime to transition from).
  row = rng.integers(len(regimes))
  S[0] = regimes[row]
  regime_idx[0] = row

  for i in range(1, n):
    # Advance the Markov chain; `row` is the state, so write the result back.
    row = rng.choice(len(regimes), p=P[row])
    theta = regimes[row]

    S[i] = theta + (S[i-1] - theta) * a + b * rng.standard_normal()
    regime_idx[i] = row

  return S, regime_idx