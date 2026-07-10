"""
Minimal config loader for the DDPG replication.

Reads a YAML config (see configs/*.yaml) into a plain namespace so both
train_agent.py and evaluate.py draw from ONE source of truth -- no more
parameters drifting between the two files.

Usage:
    from src.config import load_config
    cfg = load_config("configs/scenario1_hid.yaml")
    print(cfg.gamma, cfg.state_dim)          # attribute access
    print(cfg.regimes)                        # numpy array

Derived values (state_dim, A/regimes as numpy) are computed here so callers
never recompute them inconsistently.
"""

import numpy as np
import yaml
from types import SimpleNamespace


def load_config(path):
    with open(path) as f:
        raw = yaml.safe_load(f)

    # Convert list-valued params to numpy arrays (simulator expects arrays).
    for key in ("regimes", "A", "regimes_kappa", "regimes_sigma",
                "A_kappa", "A_sigma"):
        if key in raw and raw[key] is not None:
            raw[key] = np.array(raw[key], dtype=float)

    # Derived: state_dim = S_t (1) + I_t (1) + o_t (enc_dim).
    raw["state_dim"] = 2 + raw["enc_dim"]

    return SimpleNamespace(**raw)


if __name__ == "__main__":
    cfg = load_config("configs/scenario1_hid.yaml")
    print("algorithm :", cfg.algorithm)
    print("gamma     :", cfg.gamma)
    print("state_dim :", cfg.state_dim)
    print("regimes   :", cfg.regimes, type(cfg.regimes))
    print("A shape   :", cfg.A.shape)