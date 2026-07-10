"""
Regime-switching Ornstein-Uhlenbeck (OU) signal simulator.

Reproduces the synthetic data-generating process from Section 2 of
Macri, Jaimungal & Lillo (2025). The signal follows the OU SDE (Eq. 1):

    dS_t = kappa_t * (theta_t - S_t) dt + sigma_t * dW_t

theta ALWAYS switches via a Markov chain. kappa and sigma may be either
constant or switching, which selects the scenario:

  - Scenario 1 (theta):          kappa, sigma constant.
  - Scenario 2 (theta, kappa):   kappa switches, sigma constant.
  - Scenario 3 (theta,kappa,sigma): kappa and sigma both switch.

Each parameter that switches has its OWN independent Markov chain. The scenario
is implied by which arguments are given (constant value vs. regime set), and the
asserts enforce that exactly one form is supplied per parameter.

Initialization (paper): S_{t-W} ~ N(mu_inv, 3*sigma_inv), where
sigma_inv = sigma_min / sqrt(2*kappa_min) is the OU invariant volatility using
the MINIMUM kappa/sigma across regimes (paper's rule for the switching cases).
For the TEST phase the paper fixes S_0 = 1 (pass s0=1.0).

Note the OU discretization constants a, b are computed INSIDE the loop from the
current step's kappa_t, sigma_t (they are only constant in Scenario 1). This is
one unified code path for all scenarios; verified to reproduce Scenario 1.
"""

import numpy as np
from scipy.linalg import expm


def simulate_path(n: int, rng: np.random.Generator, regimes: np.ndarray,
                  A: np.ndarray, dt: float,
                  kappa: float = None, regimes_kappa: np.ndarray = None, A_kappa: np.ndarray = None,
                  sigma: float = None, regimes_sigma: np.ndarray = None, A_sigma: np.ndarray = None,
                  mu_inv: float = 1.0, s0: float = None):
  """
  Simulate one path of the regime-switching OU signal.

  theta is always a switching chain (regimes, A required). kappa and sigma are
  each EITHER constant (pass kappa=/sigma=) OR switching (pass regimes_kappa +
  A_kappa / regimes_sigma + A_sigma). Exactly one form per parameter.

  Returns:
    S          : (n,) signal values.
    regime_idx : (n,) active THETA regime index at each step (prob-DDPG label).
  """
  # --- validate: exactly one of (constant, regimes) per parameter ---
  assert (kappa is None) != (regimes_kappa is None), \
      "provide exactly one of kappa (constant) or regimes_kappa (switching)"
  
  assert (sigma is None) != (regimes_sigma is None), \
      "provide exactly one of sigma (constant) or regimes_sigma (switching)"
  
  if regimes_kappa is not None:
    assert A_kappa is not None, "regimes_kappa requires A_kappa (its transition matrix)"
    
  if regimes_sigma is not None:
    assert A_sigma is not None, "regimes_sigma requires A_sigma (its transition matrix)"

  kappa_switches = regimes_kappa is not None
  sigma_switches = regimes_sigma is not None

  # --- one-step transition matrices (Eq. 5) for each ACTIVE chain ---
  P_theta = expm(A * dt)
  P_kappa = expm(A_kappa * dt) if kappa_switches else None
  P_sigma = expm(A_sigma * dt) if sigma_switches else None

  # --- init uses the MINIMUM kappa / sigma across regimes (paper's rule) ---
  kappa_min = float(np.min(regimes_kappa)) if kappa_switches else kappa
  sigma_min = float(np.min(regimes_sigma)) if sigma_switches else sigma

  S = np.zeros(n)
  regime_idx = np.zeros(n, dtype=int)          # theta regime index (the label)

  # --- step 0: pick each active chain's regime uniformly (no prior state) ---
  theta_row = rng.integers(len(regimes))
  kappa_row = rng.integers(len(regimes_kappa)) if kappa_switches else None
  sigma_row = rng.integers(len(regimes_sigma)) if sigma_switches else None
  regime_idx[0] = theta_row

  # --- signal at step 0: N(mu_inv, 3*sigma_inv) using min params, or fixed s0 ---
  sigma_inv = sigma_min / np.sqrt(2 * kappa_min)
  S[0] = rng.normal(mu_inv, 3 * sigma_inv) if s0 is None else s0

  for i in range(1, n):
    # advance each active chain independently
    theta_row = rng.choice(len(regimes), p=P_theta[theta_row])
    theta_t = regimes[theta_row]

    if kappa_switches:
      kappa_row = rng.choice(len(regimes_kappa), p=P_kappa[kappa_row])
      kappa_t = regimes_kappa[kappa_row]
    else:
      kappa_t = kappa

    if sigma_switches:
      sigma_row = rng.choice(len(regimes_sigma), p=P_sigma[sigma_row])
      sigma_t = regimes_sigma[sigma_row]
    else:
      sigma_t = sigma

    # OU discretization constants for THIS step (constant only in Scenario 1)
    a = np.exp(-kappa_t * dt)
    b = sigma_t * np.sqrt((1 - np.exp(-2 * kappa_t * dt)) / (2 * kappa_t))

    S[i] = theta_t + (S[i-1] - theta_t) * a + b * rng.standard_normal()
    regime_idx[i] = theta_row

  return S, regime_idx