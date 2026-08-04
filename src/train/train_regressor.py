"""
Stage 1 (offline): train the GRU regressor to predict S~_{t+1}, then FREEZE it.

    python3 -m src.train.train_regressor --scenario 3

MSE loss between the prediction and the actual next signal. Saves frozen weights
to artifacts/scenario{N}_regressor.pt for Stage 2 (train_reg) to load.
"""
import argparse
import numpy as np
import torch

from src.env.trading_env import sample_batch
from src.models.regressor import GRURegressor
from src.utils.config import load_config
from src.utils.ou_args import ou_args_from_cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", type=int, required=True, choices=[1, 2, 3])
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--iters", type=int, default=None,
                   help="override cfg.reg_pretrain_steps (paper: 10000)")
    args = p.parse_args()

    cfg = load_config(f"configs/scenario{args.scenario}_reg.yaml")
    kappa, sigma, ou_kw = ou_args_from_cfg(cfg)
    iters = args.iters if args.iters is not None else getattr(cfg, "reg_pretrain_steps", 10000)

    seed = args.seed if args.seed is not None else (
        cfg.train_seed if cfg.train_seed is not None else np.random.randint(0, 1_000_000))
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    print(f"scenario {args.scenario} | regressor pretrain | iters {iters} | seed {seed}")

    reg = GRURegressor(hidden_dim=cfg.reg_hidden_dim, num_layers=cfg.reg_layers,
                       ffn_layers=cfg.reg_ffn_layers, ffn_hidden=cfg.reg_ffn_hidden)
    opt = torch.optim.AdamW(reg.parameters(), lr=cfg.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters)

    for m in range(iters):
        b = sample_batch(cfg.batch_size, cfg.W, rng, cfg.regimes, cfg.A,
                         kappa, sigma, cfg.dt, cfg.I_max, **ou_kw)
        win = torch.tensor(b["windows"], dtype=torch.float32)
        target = torch.tensor(b["S_next"], dtype=torch.float32).reshape(-1, 1)
        pred = reg(win)
        loss = torch.nn.functional.mse_loss(pred, target)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if m % 500 == 0:
            print(f"iter {m:5d} | mse {loss.item():.5f} | lr {opt.param_groups[0]['lr']:.5f}")

    import os
    os.makedirs("artifacts", exist_ok=True)
    torch.save({"state_dict": reg.state_dict(),
                "reg_hidden_dim": cfg.reg_hidden_dim, "reg_layers": cfg.reg_layers,
                "reg_ffn_layers": cfg.reg_ffn_layers, "reg_ffn_hidden": cfg.reg_ffn_hidden},
               f"artifacts/scenario{args.scenario}_regressor.pt")
    print(f"saved artifacts/scenario{args.scenario}_regressor.pt")


if __name__ == "__main__":
    main()