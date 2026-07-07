"""
main.py
-------
Orchestrates the full Section 4 replication: three scenarios × three agents.

Usage
=====
Quick smoke test (seconds):
  python main.py

Custom steps:
  python main.py --ddpg_steps 500 --pretrain_steps 500 --test_episodes 50 --test_steps 200

Full paper parameters (hours on the numpy autodiff engine):
  python main.py --full

Key paper parameters reproduced here
=====================================
gamma     = 0.999          (config.py, note 3)
sigma_inv = sigma/(2*kappa) (config.py, USE_TEXTBOOK_OU_STD=False)
GRU gates = tanh everywhere (nn.py GRUCell, config note 4)
W_prob    = 10 (or 20 for theta+kappa+sigma, footnote 6)
W_reg     = 50  (Table 2, all scenarios)
W_hid     = 10  (Table 3)
"""

import argparse
import sys
import numpy as np

from autodiff import Tensor
import config as C
from environment import TrainingBatchGenerator, TestEpisodeRoller
from agents import ProbDDPGAgent, RegDDPGAgent, HidDDPGAgent
from ddpg_core import train_iteration


# ─────────────────────────────── gradient clipping ──────────────────────────

def _clip_grad_norm(params, max_norm: float = 1.0) -> None:
    """In-place gradient norm clipping (same as torch.nn.utils.clip_grad_norm_)."""
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += float(np.sum(p.grad ** 2))
    total = total ** 0.5
    if total > max_norm:
        scale = max_norm / (total + 1e-8)
        for p in params:
            if p.grad is not None:
                p.grad *= scale


# ─────────────────────────────── pre-training loops ─────────────────────────

def pretrain_classifier(agent: ProbDDPGAgent, batch_gen: TrainingBatchGenerator,
                         n_steps: int) -> None:
    """
    Cross-entropy pre-training for the GRU theta-classifier (prob-DDPG Step 1).
    The label is the true theta regime index at time t (index W of the path).
    """
    print(f"    Classifier pre-train: {n_steps} steps ...")
    n_cls        = agent._num_classes
    report_every = max(1, n_steps // 5)

    for step in range(1, n_steps + 1):
        batch     = batch_gen.sample()
        S         = batch["S"]                   # (B, W+2)
        theta_idx = batch["theta_idx_t"]         # (B,)  int, true regime at t
        W1        = S.shape[1] - 1               # W+1
        S_win     = S[:, :W1]                    # (B, W+1)
        B         = S_win.shape[0]

        # autodiff forward (GRU + head are requires_grad=True from init)
        seq    = [Tensor(S_win[:, k : k + 1], requires_grad=False) for k in range(W1)]
        hidden = agent.gru.forward(seq)           # Tensor(B, gru_H)
        probs  = agent.head(hidden)               # Tensor(B, n_cls), softmax

        # cross-entropy: -mean_i  sum_c  one_hot[i,c] * log(probs[i,c])
        one_hot = np.zeros((B, n_cls))
        one_hot[np.arange(B), theta_idx] = 1.0
        loss = -(Tensor(one_hot) * probs.log()).sum(axis=-1).mean()

        agent.opt_pretrain.zero_grad()
        loss.backward()
        _clip_grad_norm(agent.gru.parameters() + agent.head.parameters(), max_norm=5.0)
        agent.opt_pretrain.step()

        if step % report_every == 0:
            print(f"      step {step:>6}/{n_steps}   CE loss = {loss.item():.4f}")



def pretrain_regressor(agent: RegDDPGAgent, batch_gen: TrainingBatchGenerator,
                        n_steps: int) -> None:
    """
    MSE pre-training for the GRU next-signal regressor (reg-DDPG Step 1).
    Target is the true S_{t+1} stored in the batch.
    """
    print(f"    Regressor pre-train: {n_steps} steps ...")
    report_every = max(1, n_steps // 5)

    for step in range(1, n_steps + 1):
        batch  = batch_gen.sample()
        S      = batch["S"]                       # (B, W+2)
        target = batch["S_tp1_target"]            # (B,)  true S_{t+1}
        W1     = S.shape[1] - 1
        S_win  = S[:, :W1]                        # (B, W+1)

        seq    = [Tensor(S_win[:, k : k + 1], requires_grad=False) for k in range(W1)]
        hidden = agent.gru.forward(seq)            # Tensor(B, gru_H)
        pred   = agent.head(hidden)                # Tensor(B, 1), SiLU output
        tgt    = Tensor(target[:, np.newaxis], requires_grad=False)
        loss   = ((pred - tgt) ** 2).mean()

        agent.opt_pretrain.zero_grad()
        loss.backward()
        _clip_grad_norm(agent.gru.parameters() + agent.head.parameters(), max_norm=5.0)
        agent.opt_pretrain.step()

        if step % report_every == 0:
            print(f"      step {step:>6}/{n_steps}   MSE loss = {loss.item():.6f}")



# ─────────────────────────────── DDPG training ──────────────────────────────

def train_ddpg(agent, batch_gen: TrainingBatchGenerator, n_steps: int) -> None:
    """Runs n_steps of Algorithm 1 via ddpg_core.train_iteration."""
    rng          = np.random.default_rng(0)
    report_every = max(1, n_steps // 5)
    print(f"    DDPG training: {n_steps} steps ...")

    for m in range(1, n_steps + 1):
        info = train_iteration(agent, batch_gen, m, rng)
        if m % report_every == 0:
            al = info["actor_loss"]
            ep = info["eps"]
            tag = f"actor_loss={al:+.4f}" if al is not None else "actor_loss=n/a"
            print(f"      step {m:>6}/{n_steps}   {tag}   eps={ep:.4f}")


# ─────────────────────────────── evaluation ─────────────────────────────────

class StatefulPolicy:
    """
    Wraps an agent into the callable signature expected by TestEpisodeRoller:
        policy_fn(window_S) -> I_next   (numpy, shape (M,))
    
    The roller's API doesn't pass the current inventory, so we track it
    internally. Since the roller also tracks I (for the reward calculation)
    and both start at i0=0 and are updated identically, they stay in sync.
    """

    def __init__(self, agent, M: int, i0: float = 0.0):
        self.agent = agent
        self.I     = np.full(M, float(i0))

    def __call__(self, window: np.ndarray) -> np.ndarray:
        """window: (M, W+1) numpy signal history."""
        G      = self.agent.build_state(window, self.I, grad=False)
        I_next = np.clip(self.agent.actor(G).data[:, 0], C.I_MIN, C.I_MAX)
        self.I = I_next
        return I_next


def evaluate(agent, scenario_cfg: dict, W: int, M: int, n: int):
    """
    Rolls the policy for M independent episodes of n steps.
    Returns (rewards array, mean, std).
    """
    roller = TestEpisodeRoller(scenario_cfg, W, seed=99)
    policy = StatefulPolicy(agent, M, i0=0.0)
    result = roller.run(policy, M=M, n=n)
    r = result["cumulative_reward"]
    return r, float(r.mean()), float(r.std())


# ─────────────────────────────── per-scenario run ───────────────────────────

def run_scenario(sc_name: str, sc_cfg: dict, N: int, N_pre: int,
                 M: int, n: int) -> dict:
    """Train and evaluate all three agents for one scenario. Returns results dict."""
    print(f"\n{'─' * 62}")
    print(f"  SCENARIO: {sc_cfg['label']}")
    print(f"{'─' * 62}")
    res = {}

    # ── prob-DDPG ─────────────────────────────────────────────────────────
    W_p = sc_cfg["lookback_prob"]
    print(f"\n  [prob-DDPG]  W={W_p}")
    agent_p = ProbDDPGAgent(sc_cfg, W_p, np.random.default_rng(1))
    bgen_p  = TrainingBatchGenerator(sc_cfg, W_p, seed=10)
    pretrain_classifier(agent_p, bgen_p, N_pre)
    train_ddpg(agent_p, bgen_p, N)
    _, mu_p, sd_p = evaluate(agent_p, sc_cfg, W_p, M, n)
    res["prob-DDPG"] = (mu_p, sd_p)
    print(f"    ➜  prob-DDPG  {mu_p:+.2f} ± {sd_p:.2f}")

    # ── hid-DDPG ──────────────────────────────────────────────────────────
    W_h = sc_cfg["lookback_hid"]
    print(f"\n  [hid-DDPG]   W={W_h}")
    agent_h = HidDDPGAgent(sc_cfg, W_h, np.random.default_rng(2))
    bgen_h  = TrainingBatchGenerator(sc_cfg, W_h, seed=20)
    train_ddpg(agent_h, bgen_h, N)
    _, mu_h, sd_h = evaluate(agent_h, sc_cfg, W_h, M, n)
    res["hid-DDPG"] = (mu_h, sd_h)
    print(f"    ➜  hid-DDPG  {mu_h:+.2f} ± {sd_h:.2f}")

    # ── reg-DDPG ──────────────────────────────────────────────────────────
    W_r = sc_cfg["lookback_reg"]
    print(f"\n  [reg-DDPG]   W={W_r}")
    agent_r = RegDDPGAgent(sc_cfg, W_r, np.random.default_rng(3))
    bgen_r  = TrainingBatchGenerator(sc_cfg, W_r, seed=30)
    pretrain_regressor(agent_r, bgen_r, N_pre)
    train_ddpg(agent_r, bgen_r, N)
    _, mu_r, sd_r = evaluate(agent_r, sc_cfg, W_r, M, n)
    res["reg-DDPG"] = (mu_r, sd_r)
    print(f"    ➜  reg-DDPG  {mu_r:+.2f} ± {sd_r:.2f}")

    return res


# ─────────────────────────────── summary table ──────────────────────────────

def print_table(all_results: dict) -> None:
    sc_keys  = list(C.SCENARIOS.keys())
    models   = ["prob-DDPG", "hid-DDPG", "reg-DDPG"]
    col_w    = 24
    sep      = "─"

    print(f"\n{'=' * 75}")
    print("TABLE 4 REPLICATION  (v2: numpy autodiff engine, gamma=0.999)")
    print(f"{'=' * 75}")

    # header
    hdr = f"{'Model':<12}"
    for k in sc_keys:
        hdr += f"  {C.SCENARIOS[k]['label'][:col_w]:<{col_w}}"
    print(hdr)
    print(sep * len(hdr))

    for model in models:
        row = f"{model:<12}"
        for k in sc_keys:
            mu, sd = all_results[k][model]
            cell   = f"{mu:+8.2f} ± {sd:5.2f}"
            row   += f"  {cell:<{col_w}}"
        print(row)

    print("=" * len(hdr))
    print("\nExpected paper ordering: prob-DDPG ≥ hid-DDPG ≥ reg-DDPG")
    print("(Requires sufficient training steps – use --full for paper parameters)")


# ─────────────────────────────── entry point ────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="replic v2 – Table 4 replication (Section 4, numpy autodiff)"
    )
    parser.add_argument("--full", action="store_true",
                        help="Use paper's N=10000, M=500, n=2000 (slow on CPU)")
    parser.add_argument("--ddpg_steps",     type=int, default=None,
                        help="DDPG training iterations per agent")
    parser.add_argument("--pretrain_steps", type=int, default=None,
                        help="Offline pre-training steps for prob/reg filters")
    parser.add_argument("--test_episodes",  type=int, default=None,
                        help="Number of test episodes M")
    parser.add_argument("--test_steps",     type=int, default=None,
                        help="Steps per test episode n")
    args = parser.parse_args()

    N     = C.TRAIN_EPISODES if args.full else (args.ddpg_steps     or 50)
    N_pre = C.TRAIN_EPISODES if args.full else (args.pretrain_steps or 50)
    M     = C.TEST_EPISODES  if args.full else (args.test_episodes   or 10)
    n     = C.TEST_STEPS     if args.full else (args.test_steps      or 50)

    print("=" * 75)
    print("replic v2 – Deep RL for Optimal Trading (numpy autodiff engine)")
    print(f"gamma={C.GAMMA}  sigma_inv=sigma/(2*kappa)  GRU gates=tanh (paper literal)")
    print(f"N_ddpg={N}  N_pretrain={N_pre}  M={M}  n={n}")
    if not args.full:
        print("(quick mode – for full paper results add --full)")
    print("=" * 75)

    np.random.seed(42)

    all_results = {}
    for sc_name, sc_cfg in C.SCENARIOS.items():
        all_results[sc_name] = run_scenario(sc_name, sc_cfg, N, N_pre, M, n)

    print_table(all_results)


if __name__ == "__main__":
    main()
