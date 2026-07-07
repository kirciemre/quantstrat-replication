"""
agents.py
---------
The three DDPG agent wrappers that satisfy the interface expected by
ddpg_core.train_iteration():

    agent.hid_gru_update(batch_gen)          – no-op for prob/reg, live for hid
    agent.build_state(S_window, I, grad=True) -> Tensor(batch, state_dim)
    agent.actor           : FeedForward
    agent.critic          : FeedForward
    agent.critic_target   : FeedForward
    agent.opt_actor       : Adam
    agent.opt_critic      : Adam

Design choices that differ from v1
====================================
* GRU gate activation follows the paper's literal Eq.(6)-(9): tanh everywhere
  (documented in config.py note 4). v1 used PyTorch's default sigmoid gates.
* gamma = 0.999 (config.py), not 0.99.
* sigma_inv = sigma/(2*kappa) per paper's literal formula (config.USE_TEXTBOOK_OU_STD=False).
* Initial regime drawn from stationary distribution (environment.py).
* Lookback windows from config.SCENARIOS; W=20 for prob-DDPG on scenario 3
  (Section 4.3 footnote 6), W=50 for reg-DDPG (Table 2).

Speed note
==========
build_state uses pure-numpy inference (_gru_numpy / _ffn_numpy) so that the
frozen first-step networks of prob-DDPG and reg-DDPG are NOT wired into the
autodiff graph during DDPG training. This cuts graph size dramatically compared
with calling gru.forward() (autodiff Tensors) on every critic/actor batch.
hid-DDPG's hid_gru_update() DOES use autodiff (it must backprop through the GRU
to update the GRU weights), but only runs once per training iteration on a
single fresh batch.
"""

import numpy as np
from nn import GRUStack, FeedForward, Linear, Adam, hard_copy
from autodiff import Tensor
import config as C
from ddpg_core import FeatureNormalizer, feature_range_for_scenario


# ──────────────────────────────── numpy inference helpers ────────────────────

def _silu(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -30.0, 30.0)   # prevent exp overflow in very early training
    return x / (1.0 + np.exp(-x))


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _gru_numpy(gru_stack: GRUStack, S_window: np.ndarray) -> np.ndarray:
    """
    Pure-numpy forward pass through a GRUStack, using the paper's tanh gate
    activation (Eq. 6-9). Returns the final hidden state h of shape (batch, H).

    Uses the weight arrays (.data) of each GRUCell directly so the autodiff
    graph is never entered; safe to call from build_state during DDPG training.
    """
    batch, seq_len = S_window.shape
    layer_input = [S_window[:, k : k + 1] for k in range(seq_len)]  # list of (B,1)

    h = None
    for cell in gru_stack.cells:
        H = cell.hidden_dim
        h = np.zeros((batch, H))
        outputs = []
        for x_k in layer_input:
            p    = np.tanh(x_k @ cell.Hp.data + h @ cell.Up.data + cell.bp.data)
            z    = np.tanh(x_k @ cell.Hz.data + h @ cell.Uz.data + cell.bz.data)
            cand = np.tanh(x_k @ cell.Hh.data + (p * h) @ cell.Uh.data + cell.bh.data)
            h    = (1.0 - z) * h + z * cand
            outputs.append(h)
        layer_input = outputs

    return h   # (batch, H)


def _ffn_numpy(ffn: FeedForward, x: np.ndarray, out_act: str = "identity") -> np.ndarray:
    """
    Pure-numpy forward pass through a FeedForward (SiLU hidden layers).
    out_act: "identity" | "silu" | "softmax" | "tanh"
    """
    h = x
    for layer in ffn.layers:
        h = _silu(h @ layer.W.data + layer.b.data)
    h = h @ ffn.out_layer.W.data + ffn.out_layer.b.data
    if out_act == "softmax":
        return _softmax(h)
    if out_act == "silu":
        return _silu(h)
    if out_act == "tanh":
        return np.tanh(h)
    return h   # identity


# ──────────────────────────────── ProbDDPGAgent ──────────────────────────────

class ProbDDPGAgent:
    """
    Two-step agent (Section 3.3.1):
      Step 1 (offline): GRU + softmax FFN classifies the theta regime.
      Step 2 (DDPG):    State = [norm(S_t), norm(I_t), pi_hat_0, pi_hat_1, pi_hat_2]
    """

    def __init__(self, scenario_cfg: dict, W: int, rng):
        sc      = scenario_cfg
        gru_L   = sc["gru_layers_two_step"]
        gru_H   = sc["gru_hidden_two_step"]
        n_cls   = len(sc["theta_regimes"])   # 3 theta states
        ac      = C.ACTOR_CRITIC_SIZES["prob-DDPG"]
        s_dim   = 2 + n_cls                  # norm_S + norm_I + n_cls probs

        s_lo, s_hi      = feature_range_for_scenario(sc)
        self.norm        = FeatureNormalizer(C.I_MIN, C.I_MAX, s_lo, s_hi)
        self.W           = W
        self._num_classes = n_cls

        # ── first-step network (pre-trained, frozen during DDPG) ──────────
        self.gru  = GRUStack(gru_L, input_dim=1, hidden_dim=gru_H, rng=rng)
        self.head = FeedForward(gru_H, C.HEAD_HIDDEN_DIM, C.HEAD_N_LAYERS,
                                n_cls, rng, hidden_act="silu", out_act="softmax")
        self.opt_pretrain = Adam(
            self.gru.parameters() + self.head.parameters(),
            lr=C.LEARNING_RATE, weight_decay=1e-5,
        )

        # ── actor / critic ────────────────────────────────────────────────
        self.actor  = FeedForward(s_dim, ac["hidden_dim"], ac["n_layers"],
                                   1, rng, hidden_act="silu",
                                   out_act="tanh", out_scale=C.I_MAX)
        self.critic = FeedForward(s_dim + 1, ac["hidden_dim"], ac["n_layers"],
                                   1, rng, hidden_act="silu", out_act="identity")
        self.critic_target = FeedForward(s_dim + 1, ac["hidden_dim"], ac["n_layers"],
                                          1, rng, hidden_act="silu", out_act="identity")
        hard_copy(self.critic_target.parameters(), self.critic.parameters())

        self.opt_actor  = Adam(self.actor.parameters(),  lr=C.LEARNING_RATE, weight_decay=1e-5)
        self.opt_critic = Adam(self.critic.parameters(), lr=C.LEARNING_RATE, weight_decay=1e-5)

    def build_state(self, S_window: np.ndarray, I: np.ndarray, grad: bool = True) -> Tensor:
        """
        S_window : (batch, W+1)  numpy – signal history {S_{t-W},...,S_t}
        I        : (batch,)      numpy – current inventory
        """
        h      = _gru_numpy(self.gru, S_window)             # (batch, gru_H)
        probs  = _ffn_numpy(self.head, h, "softmax")         # (batch, n_cls)
        S_t    = self.norm.norm_S(S_window[:, -1:])          # (batch, 1)
        I_n    = self.norm.norm_I(I[:, np.newaxis])          # (batch, 1)
        G_np   = np.concatenate([S_t, I_n, probs], axis=1)  # (batch, 2+n_cls)
        return Tensor(G_np, requires_grad=grad)

    def hid_gru_update(self, batch_gen):
        pass   # no-op: GRU is frozen after pre-training


# ──────────────────────────────── RegDDPGAgent ───────────────────────────────

class RegDDPGAgent:
    """
    Two-step agent (Section 3.3.2):
      Step 1 (offline): GRU + SiLU FFN predicts S̃_{t+1}.
      Step 2 (DDPG):    State = [norm(S_t), norm(I_t), norm(S̃_{t+1})]
    """

    def __init__(self, scenario_cfg: dict, W: int, rng):
        sc      = scenario_cfg
        gru_L   = sc["gru_layers_two_step"]
        gru_H   = sc["gru_hidden_two_step"]
        ac      = C.ACTOR_CRITIC_SIZES["reg-DDPG"]
        s_dim   = 3   # norm_S + norm_I + norm_pred_S

        s_lo, s_hi = feature_range_for_scenario(sc)
        self.norm   = FeatureNormalizer(C.I_MIN, C.I_MAX, s_lo, s_hi)
        self.W      = W

        # ── first-step network ────────────────────────────────────────────
        self.gru  = GRUStack(gru_L, input_dim=1, hidden_dim=gru_H, rng=rng)
        self.head = FeedForward(gru_H, C.HEAD_HIDDEN_DIM, C.HEAD_N_LAYERS,
                                1, rng, hidden_act="silu", out_act="silu")
        self.opt_pretrain = Adam(
            self.gru.parameters() + self.head.parameters(),
            lr=C.LEARNING_RATE, weight_decay=1e-5,
        )

        # ── actor / critic ────────────────────────────────────────────────
        self.actor  = FeedForward(s_dim, ac["hidden_dim"], ac["n_layers"],
                                   1, rng, hidden_act="silu",
                                   out_act="tanh", out_scale=C.I_MAX)
        self.critic = FeedForward(s_dim + 1, ac["hidden_dim"], ac["n_layers"],
                                   1, rng, hidden_act="silu", out_act="identity")
        self.critic_target = FeedForward(s_dim + 1, ac["hidden_dim"], ac["n_layers"],
                                          1, rng, hidden_act="silu", out_act="identity")
        hard_copy(self.critic_target.parameters(), self.critic.parameters())

        self.opt_actor  = Adam(self.actor.parameters(),  lr=C.LEARNING_RATE, weight_decay=1e-5)
        self.opt_critic = Adam(self.critic.parameters(), lr=C.LEARNING_RATE, weight_decay=1e-5)

    def build_state(self, S_window: np.ndarray, I: np.ndarray, grad: bool = True) -> Tensor:
        h      = _gru_numpy(self.gru, S_window)             # (batch, gru_H)
        pred   = _ffn_numpy(self.head, h, "silu")           # (batch, 1)
        S_t    = self.norm.norm_S(S_window[:, -1:])
        I_n    = self.norm.norm_I(I[:, np.newaxis])
        pred_n = self.norm.norm_S(pred)                     # normalize prediction
        G_np   = np.concatenate([S_t, I_n, pred_n], axis=1)
        return Tensor(G_np, requires_grad=grad)

    def hid_gru_update(self, batch_gen):
        pass


# ──────────────────────────────── HidDDPGAgent ───────────────────────────────

class HidDDPGAgent:
    """
    One-step agent (Section 3.2.1):
      GRU encoder trained *online* each iteration via an auxiliary next-step
      MSE loss before the critic and actor updates (Algorithm 1, line 12-13).
      State = [norm(S_t), norm(I_t), (h_t + 1)/2]   dim = 2 + gru_H
    """

    def __init__(self, scenario_cfg: dict, W: int, rng):
        sc      = scenario_cfg
        gru_L   = sc["gru_layers_hid"]     # 1 (or 2 for theta+kappa+sigma)
        gru_H   = sc["gru_hidden_hid"]     # 10
        ac      = C.ACTOR_CRITIC_SIZES["hid-DDPG"]
        s_dim   = 2 + gru_H

        s_lo, s_hi = feature_range_for_scenario(sc)
        self.norm   = FeatureNormalizer(C.I_MIN, C.I_MAX, s_lo, s_hi)
        self.W      = W

        # ── GRU encoder + auxiliary prediction head (online training) ─────
        self.gru      = GRUStack(gru_L, input_dim=1, hidden_dim=gru_H, rng=rng)
        self.aux_head = Linear(gru_H, 1, rng)        # single linear layer
        self.opt_gru  = Adam(
            self.gru.parameters() + self.aux_head.parameters(),
            lr=C.LEARNING_RATE, weight_decay=1e-5,
        )

        # ── actor / critic ────────────────────────────────────────────────
        self.actor  = FeedForward(s_dim, ac["hidden_dim"], ac["n_layers"],
                                   1, rng, hidden_act="silu",
                                   out_act="tanh", out_scale=C.I_MAX)
        self.critic = FeedForward(s_dim + 1, ac["hidden_dim"], ac["n_layers"],
                                   1, rng, hidden_act="silu", out_act="identity")
        self.critic_target = FeedForward(s_dim + 1, ac["hidden_dim"], ac["n_layers"],
                                          1, rng, hidden_act="silu", out_act="identity")
        hard_copy(self.critic_target.parameters(), self.critic.parameters())

        self.opt_actor  = Adam(self.actor.parameters(),  lr=C.LEARNING_RATE, weight_decay=1e-5)
        self.opt_critic = Adam(self.critic.parameters(), lr=C.LEARNING_RATE, weight_decay=1e-5)

    def hid_gru_update(self, batch_gen):
        """
        One auxiliary update of the GRU encoder via next-step prediction loss.
        Called inside train_iteration() *before* the critic/actor steps so that
        the features used by build_state() reflect the freshest GRU weights
        (those weights are read as .data in the numpy forward pass).
        """
        batch  = batch_gen.sample()
        S      = batch["S"]
        target = batch["S_tp1_target"]   # (B,)  true S_{t+1}
        W1     = S.shape[1] - 1         # W + 1 signal points
        S_win  = S[:, :W1]              # (B, W+1)

        # autodiff forward through GRU (builds graph for GRU weight gradients)
        seq    = [Tensor(S_win[:, k : k + 1], requires_grad=False) for k in range(W1)]
        hidden = self.gru.forward(seq)  # Tensor(B, gru_H)

        # auxiliary head: Linear + LeakyReLU
        pred = self.aux_head(hidden).leaky_relu(0.01)   # Tensor(B, 1)
        tgt  = Tensor(target[:, np.newaxis], requires_grad=False)
        loss = ((pred - tgt) ** 2).mean()

        self.opt_gru.zero_grad()
        loss.backward()
        self.opt_gru.step()

    def build_state(self, S_window: np.ndarray, I: np.ndarray, grad: bool = True) -> Tensor:
        """
        numpy inference through the GRU (reads current .data of weights).
        The tanh-bounded hidden state is mapped to (0,1) before concatenation.
        """
        h    = _gru_numpy(self.gru, S_window)             # (B, gru_H) in (-1,1)
        h_n  = FeatureNormalizer.hidden_tanh_to_unit(h)  # → (0,1)
        S_t  = self.norm.norm_S(S_window[:, -1:])
        I_n  = self.norm.norm_I(I[:, np.newaxis])
        G_np = np.concatenate([S_t, I_n, h_n], axis=1)  # (B, 2+gru_H)
        return Tensor(G_np, requires_grad=grad)
