import numpy as np
import numpy.typing as npt


def reward(I_t: npt.ArrayLike, I_next: npt.ArrayLike,
           S_t: npt.ArrayLike, S_next: npt.ArrayLike, lam: float) -> npt.ArrayLike:
    """
    Per-step trading reward (Macri, Jaimungal & Lillo 2025, Eq. 4):

        r_t = I_next * (S_next - S_t) - lam * |I_next - I_t|

      - I_next * (S_next - S_t) : P&L from holding the NEW inventory across the step
      - lam * |I_next - I_t|    : transaction cost on the volume traded (lam = 0.05 in the paper)

    Pure function: does not enforce the inventory bound [-Imax, Imax] -- that is the
    Actor's responsibility. Inputs may be scalars or NumPy arrays (broadcasts over a batch).
    """
    q_t = I_next - I_t
    return I_next * (S_next - S_t) - lam * np.abs(q_t)