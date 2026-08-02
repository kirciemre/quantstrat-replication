"""
Stage 1 (offline): train the GRU regime classifier, then FREEZE it.

    python3 -m src.train.train_classifier --scenario 3

Generates labeled batches via sample_batch (which now returns regime,
regime_kappa, regime_sigma), trains separate softmax heads by cross-entropy
against the true regime indices, and saves the frozen weights to
artifacts/scenario{N}_classifier.pt for Stage 2 (prob RL training) to load.
"""
import argparse
import numpy as np
import torch

from src.env.trading_env import sample_batch
from src.models.gru_classifier import GRUClassifier
from src.utils.config import load_config
from src.utils.ou_args import ou_args_from_cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", type=int, required=True, choices=[1, 2, 3])
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--theta-only", action="store_true",
                   help="classify ONLY theta (disable kappa/sigma heads) even on Sc2/Sc3")
    p.add_argument("--iters", type=int, default=None,
                   help="override cfg.classifier_iters (paper: 10000)")
    args = p.parse_args()

    cfg = load_config(f"configs/scenario{args.scenario}_prob.yaml")
    kappa, sigma, ou_kw = ou_args_from_cfg(cfg)
    iters = args.iters if args.iters is not None else getattr(cfg, "classifier_iters", 10000)

    seed = args.seed if args.seed is not None else (
        cfg.train_seed if cfg.train_seed is not None else np.random.randint(0, 1_000_000))
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    # number of regimes per switching param (0 = constant, head disabled)
    n_theta = len(cfg.regimes)
    if args.theta_only:
        n_kappa = n_sigma = 0                 # theta-only ablation
    else:
        n_kappa = len(ou_kw["regimes_kappa"]) if "regimes_kappa" in ou_kw else 0
        n_sigma = len(ou_kw["regimes_sigma"]) if "regimes_sigma" in ou_kw else 0
    print(f"scenario {args.scenario} | classifier heads: "
          f"theta={n_theta} kappa={n_kappa or '-'} sigma={n_sigma or '-'} | seed {seed}")

    clf = GRUClassifier(cfg.d_h, cfg.d_l, n_theta=n_theta, n_kappa=n_kappa, n_sigma=n_sigma)
    opt = torch.optim.AdamW(clf.parameters(), lr=cfg.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters)
    ce = torch.nn.functional.cross_entropy

    for m in range(iters):
        b = sample_batch(cfg.batch_size, cfg.W, rng, cfg.regimes, cfg.A,
                         kappa, sigma, cfg.dt, cfg.I_max, **ou_kw)
        win = torch.tensor(b["windows"], dtype=torch.float32)
        lg = clf.logits(win)

        loss = ce(lg["theta"], torch.tensor(b["regime"], dtype=torch.long))
        if n_kappa:
            loss = loss + ce(lg["kappa"], torch.tensor(b["regime_kappa"], dtype=torch.long))
        if n_sigma:
            loss = loss + ce(lg["sigma"], torch.tensor(b["regime_sigma"], dtype=torch.long))

        opt.zero_grad(); loss.backward(); opt.step(); sched.step()

        if m % 300 == 0:
            with torch.no_grad():
                acc = {}
                acc["theta"] = (lg["theta"].argmax(-1).numpy() == b["regime"]).mean()
                if n_kappa:
                    acc["kappa"] = (lg["kappa"].argmax(-1).numpy() == b["regime_kappa"]).mean()
                if n_sigma:
                    acc["sigma"] = (lg["sigma"].argmax(-1).numpy() == b["regime_sigma"]).mean()
            accs = " ".join(f"{k}={v:.3f}" for k, v in acc.items())
            print(f"iter {m:5d} | loss {loss.item():.4f} | train-acc {accs} | lr {opt.param_groups[0]['lr']:.5f}")

    import os
    os.makedirs("artifacts", exist_ok=True)
    torch.save({"state_dict": clf.state_dict(),
                "n_theta": n_theta, "n_kappa": n_kappa, "n_sigma": n_sigma},
               (f"artifacts/scenario{args.scenario}_classifier_thetaonly.pt" if args.theta_only
                else f"artifacts/scenario{args.scenario}_classifier.pt"))
    tag = "_thetaonly" if args.theta_only else ""
    print(f"saved artifacts/scenario{args.scenario}_classifier{tag}.pt (phi_dim={clf.phi_dim})")


if __name__ == "__main__":
    main()