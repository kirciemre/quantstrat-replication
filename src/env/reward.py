"""
Per-step trading reward.

Paper (Eq. 4):
    r_t = I_{t+1} (S_{t+1} - S_t)  -  lam * |q_t|,     q_t = I_{t+1} - I_t

Both terms are linear/piecewise-linear in the action, so the optimum sits at a
CORNER (+/- I_max) -- the bang-bang behaviour observed in training. Adding a
convex (superlinear) cost makes the objective strictly concave in the action and
produces a unique interior optimum:

    r_t = I_{t+1} (S_{t+1} - S_t)
          - lam * |q_t|          linear cost      (spread / fees)   [paper]
          - eta * q_t^2          temporary impact (fill quality falls with size)
          - psi * I_{t+1}^2      holding cost     (risk / margin / exit risk)

eta = psi = 0 recovers Eq. (4) exactly, so the paper-faithful reward is nested.

d^2 r / da^2 = -2(eta + psi) < 0 whenever eta + psi > 0  ->  strictly concave.

IMPORTANT (comparability): train with the penalties, but EVALUATE with the
paper's reward (eta = psi = 0). Otherwise reported cumulative rewards are on a
different scale than Table 4 and cannot be compared to 15.70 / 8.08 / 1.29.
"""

import numpy as np


def reward(I_t, I_next, S_t, S_next, lam, eta=0.0, psi=0.0):
    """
    Per-step reward. Works on scalars or batched arrays/tensors.

    Args:
      I_t    : inventory held coming into the step.
      I_next : the action -- new inventory I_{t+1}.
      S_t, S_next : raw (UN-normalized) signal now and next step.
      lam    : linear transaction cost (paper, Eq. 4).
      eta    : quadratic trade cost (temporary market impact). 0 = paper.
      psi    : quadratic inventory (holding) cost.            0 = paper.

    All quantities are RAW units -- this is real P&L, never normalized inputs.
    """
    q = I_next - I_t
    pnl = I_next * (S_next - S_t)

    r = pnl - lam * abs(q)

    if eta:
        r = r - eta * q ** 2
    if psi:
        r = r - psi * I_next ** 2

    return r