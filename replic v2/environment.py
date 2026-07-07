"""
environment.py
---------------
Simulates the regime-switching Ornstein-Uhlenbeck trading signal of
Section 2.1 (Eq. 1) with Markov-chain-driven parameters (Eq. 5), and
implements the trading reward of Eq. (4).

Layout
======
  RegimeProcess     - one latent parameter (theta, kappa or sigma): either a
                      constant, or an n-state continuous-time Markov chain
                      sampled at the discretisation step dt via
                      P(dt) = expm(A * dt)  (Eq. 5).
  simulate_ou_batch - draws a batch of signal paths given three
                      RegimeProcess objects (theta, kappa, sigma).
  reward            - Eq. (4): r_t = I_{t+1}(S_{t+1}-S_t) - lambda|I_{t+1}-I_t|
  TrainingBatchGenerator - produces the length-(W+2) signal windows and
                      random inventories used every training iteration
                      (Section 3.1: "we simulate batches of size b of time
                      series ... of length W+2 and inventories I_t").
  TestEpisodeRoller - rolls a *trained* policy forward for n steps across M
                      parallel test episodes, in the rolling-window fashion
                      described in Section 3.1, starting at S0=1, I0=0.
"""
from __future__ import annotations
import numpy as np
from scipy.linalg import expm

import config as C


# --------------------------------------------------------------- regimes
def stationary_distribution(A):
    """Stationary distribution pi of a continuous-time Markov chain with
    generator A (pi A = 0, sum(pi) = 1), via the left null-vector of A."""
    vals, vecs = np.linalg.eig(A.T)
    idx = int(np.argmin(np.abs(vals)))
    vec = np.real(vecs[:, idx])
    vec = np.clip(vec, 0, None)
    vec = vec / vec.sum()
    return vec


class RegimeProcess:
    """A single latent parameter: constant, or a discrete-state Markov chain."""

    def __init__(self, regimes=None, constant=None, A=None, dt=C.DT):
        self.is_constant = regimes is None
        self.dt = dt
        if self.is_constant:
            if constant is None:
                raise ValueError("Provide `constant` when `regimes` is None.")
            self.constant = float(constant)
        else:
            self.regimes = np.asarray(regimes, dtype=float)
            self.n_states = len(self.regimes)
            self.A = np.asarray(A, dtype=float)
            self.P = expm(self.A * dt)
            self.cumP = np.cumsum(self.P, axis=1)
            self.stationary = stationary_distribution(self.A)

    def min_value(self):
        return self.constant if self.is_constant else float(self.regimes.min())

    def sample_path(self, batch, length, rng):
        """Returns (values, idx), both shape (batch, length)."""
        if self.is_constant:
            values = np.full((batch, length), self.constant, dtype=float)
            idx = np.zeros((batch, length), dtype=np.int64)
            return values, idx

        idx = np.zeros((batch, length), dtype=np.int64)
        idx[:, 0] = rng.choice(self.n_states, size=batch, p=self.stationary)
        for i in range(1, length):
            u = rng.uniform(size=batch)
            cum_rows = self.cumP[idx[:, i - 1]]
            idx[:, i] = (u[:, None] < cum_rows).argmax(axis=1)
        values = self.regimes[idx]
        return values, idx

    def continue_path(self, last_idx, extra_len, rng):
        """Continue `extra_len` more steps given the current state indices
        `last_idx` (shape (batch,)). Returns (values, idx) of shape
        (batch, extra_len), NOT including the given last state."""
        batch = last_idx.shape[0]
        if self.is_constant:
            values = np.full((batch, extra_len), self.constant, dtype=float)
            idx = np.zeros((batch, extra_len), dtype=np.int64)
            return values, idx
        idx = np.zeros((batch, extra_len), dtype=np.int64)
        cur = last_idx.copy()
        for i in range(extra_len):
            u = rng.uniform(size=batch)
            cum_rows = self.cumP[cur]
            nxt = (u[:, None] < cum_rows).argmax(axis=1)
            idx[:, i] = nxt
            cur = nxt
        values = self.regimes[idx]
        return values, idx


def build_regime_processes(scenario_cfg, dt=C.DT):
    sc = scenario_cfg
    theta = RegimeProcess(regimes=sc["theta_regimes"], A=C.A_THETA, dt=dt)
    if sc["kappa_regimes"] is None:
        kappa = RegimeProcess(constant=sc["kappa"], dt=dt)
    else:
        kappa = RegimeProcess(regimes=sc["kappa_regimes"], A=C.A_KAPPA, dt=dt)
    if sc["sigma_regimes"] is None:
        sigma = RegimeProcess(constant=sc["sigma"], dt=dt)
    else:
        sigma = RegimeProcess(regimes=sc["sigma_regimes"], A=C.A_SIGMA, dt=dt)
    return theta, kappa, sigma


# --------------------------------------------------------------- OU signal
def invariant_std(kappa_val, sigma_val, use_textbook=C.USE_TEXTBOOK_OU_STD):
    """sigma_inv, per the paper's literal formula sigma/(2*kappa) by default,
    or the textbook OU stationary std sigma/sqrt(2*kappa) if requested."""
    if use_textbook:
        return sigma_val / np.sqrt(2.0 * kappa_val)
    return sigma_val / (2.0 * kappa_val)


def simulate_ou_batch(theta_proc, kappa_proc, sigma_proc, batch, length, dt, rng,
                       s0=None, mu_inv=C.MU_INV, use_textbook_std=C.USE_TEXTBOOK_OU_STD):
    """
    Draws `batch` independent signal paths of `length` time points using the
    discretised Ornstein-Uhlenbeck recursion (Eq. 1-2):
        S_{i+1} = S_i + kappa_i (theta_i - S_i) dt + sigma_i sqrt(dt) Z_i
    with regime values sampled from the three RegimeProcess objects.

    If `s0` is None, S_0 ~ N(mu_inv, 3*sigma_inv) (Section 3.1), using the
    *minimum* regime value of kappa/sigma when they are time-varying
    (footnote 4). Otherwise S_0 is fixed to `s0` for every path (used for
    the testing rollout, which always starts at S0 = 1).

    Returns a dict with S (batch,length), theta/kappa/sigma values and
    indices (batch,length), useful for prob-DDPG's classification target
    and for debugging/inspection.
    """
    theta_vals, theta_idx = theta_proc.sample_path(batch, length, rng)
    kappa_vals, kappa_idx = kappa_proc.sample_path(batch, length, rng)
    sigma_vals, sigma_idx = sigma_proc.sample_path(batch, length, rng)

    if s0 is None:
        kappa_for_std = kappa_proc.min_value()
        sigma_for_std = sigma_proc.min_value()
        sigma_inv = invariant_std(kappa_for_std, sigma_for_std, use_textbook_std)
        S0 = rng.normal(loc=mu_inv, scale=3.0 * sigma_inv, size=batch)
    else:
        S0 = np.full(batch, float(s0))

    S = np.zeros((batch, length))
    S[:, 0] = S0
    Z = rng.normal(size=(batch, length - 1))
    for i in range(length - 1):
        S[:, i + 1] = S[:, i] + kappa_vals[:, i] * (theta_vals[:, i] - S[:, i]) * dt \
            + sigma_vals[:, i] * np.sqrt(dt) * Z[:, i]

    return dict(S=S, theta_vals=theta_vals, kappa_vals=kappa_vals, sigma_vals=sigma_vals,
                theta_idx=theta_idx, kappa_idx=kappa_idx, sigma_idx=sigma_idx)


# --------------------------------------------------------------------- reward
def reward(I_next, S_t, S_next, I_curr, lam=C.LAMBDA_COST):
    """Eq. (4): r_t = I_{t+1}(S_{t+1}-S_t) - lambda|I_{t+1}-I_t|"""
    return I_next * (S_next - S_t) - lam * np.abs(I_next - I_curr)


# ------------------------------------------------------------ training batches
class TrainingBatchGenerator:
    """
    Produces one training batch per call to `sample()`:
      S        : (batch, W+2)  signal windows {S_u}_{t-W}^{t+1}
      I_t      : (batch,)      current inventory,  ~ U[Imin, Imax]
      theta_idx_t : (batch,)   TRUE regime index of theta at time t (for
                                pretraining prob-DDPG's classifier)
      S_tp1_target: (batch,)   TRUE next signal value S_{t+1} (for
                                pretraining reg-DDPG's regressor, and for
                                hid-DDPG's own auxiliary GRU loss)
    W is the look-back window (so the window has W+1 points, t-W..t) and the
    array carries one extra point (t+1) needed for the reward and for the
    "next state" window G'_{t+1} used by the critic's TD target.
    """

    def __init__(self, scenario_cfg, W, dt=C.DT, batch_size=C.BATCH_SIZE,
                 i_min=C.I_MIN, i_max=C.I_MAX, seed=None):
        self.scenario_cfg = scenario_cfg
        self.W = W
        self.dt = dt
        self.batch_size = batch_size
        self.i_min, self.i_max = i_min, i_max
        self.theta_proc, self.kappa_proc, self.sigma_proc = build_regime_processes(scenario_cfg, dt)
        self.rng = np.random.default_rng(seed)

    def sample(self, batch_size=None):
        b = batch_size or self.batch_size
        length = self.W + 2  # indices 0..W+1 <-> times t-W..t+1
        out = simulate_ou_batch(self.theta_proc, self.kappa_proc, self.sigma_proc,
                                 b, length, self.dt, self.rng)
        S = out["S"]
        I_t = self.rng.uniform(self.i_min, self.i_max, size=b)
        theta_idx_t = out["theta_idx"][:, self.W]     # regime AT time t (index W)
        S_tp1_target = S[:, self.W + 1]                 # true S_{t+1}
        return dict(S=S, I_t=I_t, theta_idx_t=theta_idx_t, S_tp1_target=S_tp1_target)


# --------------------------------------------------------------- test rollout
class TestEpisodeRoller:
    """
    Rolls a trained policy forward across M parallel test episodes for n
    steps, in the rolling-window fashion of Section 3.1: at each time t the
    agent sees {S_u}_{t-W}^{t} and I_t, picks I_{t+1}, collects the reward,
    then the window slides forward by one step.

    Testing episodes always start at S0 = 1, I0 = 0 (Section 3.1). To give
    the GRU/classifier a full W-length history for its very first decision,
    a W-step "pre-history" is simulated from the invariant distribution and
    then the value at time 0 is overridden to exactly 1.0 (documented in
    the README as a modelling choice, since the paper does not specify how
    the pre-history before a fixed S0 is generated).
    """

    def __init__(self, scenario_cfg, W, dt=C.DT, i_min=C.I_MIN, i_max=C.I_MAX, seed=None):
        self.scenario_cfg = scenario_cfg
        self.W = W
        self.dt = dt
        self.i_min, self.i_max = i_min, i_max
        self.theta_proc, self.kappa_proc, self.sigma_proc = build_regime_processes(scenario_cfg, dt)
        self.rng = np.random.default_rng(seed)

    def run(self, policy_fn, M=C.TEST_EPISODES, n=C.TEST_STEPS, s0=1.0, i0=0.0):
        """
        policy_fn(window_S) -> I_next (numpy array, shape (M,))
            window_S: numpy array (M, W+1), the current {S_u}_{t-W}^{t}.
            policy_fn is expected to be *deterministic* (no exploration
            noise) at test time, exactly as in Eq. (10): I*_{t+1} = pi(G_t | mu*_pi).

        Returns dict with:
          cumulative_reward : (M,) final R_n for each test episode
          reward_path        : (M, n) per-step reward r_t (for Figure 12-style diagnostics)
          inventory_path      : (M, n+1)
          signal_path          : (M, n+1)  (S_0 .. S_n, S_0 = s0)
        """
        # simulate a W-length pre-history ending just before t=0, then splice S0=s0 on.
        pre = simulate_ou_batch(self.theta_proc, self.kappa_proc, self.sigma_proc,
                                 M, self.W + 1, self.dt, self.rng)
        window = pre["S"].copy()          # (M, W+1) representing times -W..0
        window[:, -1] = s0                 # override S_0 = 1 exactly, per Section 3.1
        last_theta_idx = pre["theta_idx"][:, -1]
        last_kappa_idx = pre["kappa_idx"][:, -1]
        last_sigma_idx = pre["sigma_idx"][:, -1]

        I = np.full(M, float(i0))
        cum_reward = np.zeros(M)
        reward_path = np.zeros((M, n))
        inventory_path = np.zeros((M, n + 1))
        inventory_path[:, 0] = I
        signal_path = np.zeros((M, n + 1))
        signal_path[:, 0] = window[:, -1]

        for t in range(n):
            I_next = policy_fn(window)
            I_next = np.clip(I_next, self.i_min, self.i_max)

            theta_next, last_theta_idx = self.theta_proc.continue_path(last_theta_idx, 1, self.rng)
            kappa_next, last_kappa_idx = self.kappa_proc.continue_path(last_kappa_idx, 1, self.rng)
            sigma_next, last_sigma_idx = self.sigma_proc.continue_path(last_sigma_idx, 1, self.rng)
            last_theta_idx = last_theta_idx[:, 0]
            last_kappa_idx = last_kappa_idx[:, 0]
            last_sigma_idx = last_sigma_idx[:, 0]
            theta_next, kappa_next, sigma_next = theta_next[:, 0], kappa_next[:, 0], sigma_next[:, 0]

            S_t = window[:, -1]
            Z = self.rng.normal(size=M)
            S_next = S_t + kappa_next * (theta_next - S_t) * self.dt + sigma_next * np.sqrt(self.dt) * Z

            r = reward(I_next, S_t, S_next, I)
            cum_reward += r
            reward_path[:, t] = r

            window = np.concatenate([window[:, 1:], S_next[:, None]], axis=1)
            I = I_next
            inventory_path[:, t + 1] = I
            signal_path[:, t + 1] = S_next

        return dict(cumulative_reward=cum_reward, reward_path=reward_path,
                    inventory_path=inventory_path, signal_path=signal_path)
