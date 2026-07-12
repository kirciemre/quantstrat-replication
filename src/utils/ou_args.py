"""
Config -> OU-parameter routing for the simulator.

Turns a loaded scenario config into the (kappa, sigma, ou_kwargs) triple that
simulate_path / sample_batch expect, so train and eval scripts don't hardcode
scenario-specific calls. Works for all three scenarios:

  Scenario 1: cfg has kappa, sigma            -> (kappa, sigma, {})
  Scenario 2: cfg has regimes_kappa, A_kappa  -> (None, sigma, {regimes_kappa, A_kappa})
  Scenario 3: + regimes_sigma, A_sigma        -> (None, None, {all four})

For a parameter that SWITCHES, its constant is returned as None and its regime
chain goes into ou_kwargs -- matching simulate_path's "exactly one form" rule.

Usage (train and eval alike):
    kappa, sigma, ou_kw = ou_args_from_cfg(cfg)
    simulate_path(n, rng, cfg.regimes, cfg.A, cfg.dt,
                  kappa=kappa, sigma=sigma, **ou_kw)          # + s0=1.0 in eval
    sample_batch(cfg.batch_size, cfg.W, rng, cfg.regimes, cfg.A,
                 kappa, sigma, cfg.dt, cfg.I_max, **ou_kw)
"""


def ou_args_from_cfg(cfg):
    """
    Return (kappa, sigma, ou_kwargs) for the OU simulator from a scenario config.

    A parameter that switches (regimes_* present in the config) is returned as a
    None constant, with its regime chain placed in ou_kwargs. A constant
    parameter is returned as its value with nothing added to ou_kwargs.
    """
    ou_kwargs = {}

    # --- kappa: switching if regimes_kappa present, else constant ---
    if getattr(cfg, "regimes_kappa", None) is not None:
        ou_kwargs["regimes_kappa"] = cfg.regimes_kappa
        ou_kwargs["A_kappa"] = cfg.A_kappa
        kappa = None
    else:
        kappa = cfg.kappa

    # --- sigma: switching if regimes_sigma present, else constant ---
    if getattr(cfg, "regimes_sigma", None) is not None:
        ou_kwargs["regimes_sigma"] = cfg.regimes_sigma
        ou_kwargs["A_sigma"] = cfg.A_sigma
        sigma = None
    else:
        sigma = cfg.sigma

    return kappa, sigma, ou_kwargs