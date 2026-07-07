"""
nn.py
-----
Neural-network building blocks used throughout the replication, all built
on top of the tiny `autodiff.Tensor` engine (see autodiff.py):

  * Linear            - a single affine layer y = xW + b
  * FeedForward       - the Actor / Critic / classifier / regressor MLPs
  * GRUCell / GRUStack - the recurrent network of Section 3.2.1, Eq. (6)-(9)
  * Adam              - the "Weighted ADAM optimiser" mentioned in the paper
  * soft_update        - the target-network Polyak/soft update (tau = 0.001)

Paper -> code mapping
======================
Eq. (6)-(9) define the GRU with reset gate p_k, update gate z_k and
candidate state h~_k, and state *explicitly* that "sigma is the hyperbolic
tangent (tanh) activation function" for all three. That is a non-standard
GRU: the usual formulation (Cho et al., 2014, cited by the paper) uses a
*sigmoid* for the reset/update gates (so they act as values in [0,1], i.e.
a true convex blend in Eq. 9) and *tanh* only for the candidate state. We
implement the paper's literal statement (`gate_activation="tanh"`,
the default) but expose `gate_activation="sigmoid_tanh"` to switch to the
textbook GRU if you want to compare the two.

The paper's per-symbol matrix shapes for the GRU (e.g. Hp in R^{dh x b},
mixing the *batch size* b into a weight matrix) are not internally
consistent for a properly batched, multi-layer network (they would tie the
hidden state to a fixed batch size rather than letting it vary per
sample). We use the standard batched convention instead - weights
in R^{input_dim x hidden_dim} / R^{hidden_dim x hidden_dim}, state in
R^{batch x hidden_dim} - which is what every practical GRU implementation
(including PyTorch's) uses, while keeping the paper's stated gate
equations and activation choice.
"""
from __future__ import annotations
import numpy as np
from autodiff import Tensor

ACTIVATIONS = {
    "silu": lambda t: t.silu(),
    "tanh": lambda t: t.tanh(),
    "sigmoid": lambda t: t.sigmoid(),
    "leaky_relu": lambda t: t.leaky_relu(0.01),
    "identity": lambda t: t,
    "softmax": lambda t: t.softmax(axis=-1),
}


def _glorot(fan_in, fan_out, rng):
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, size=(fan_in, fan_out))


class Linear:
    def __init__(self, in_dim, out_dim, rng, bias=True):
        self.W = Tensor(_glorot(in_dim, out_dim, rng), requires_grad=True, name="W")
        self.b = Tensor(np.zeros((1, out_dim)), requires_grad=True, name="b") if bias else None

    def __call__(self, x):
        out = x.matmul(self.W)
        if self.b is not None:
            out = out + self.b
        return out

    def parameters(self):
        return [self.W] + ([self.b] if self.b is not None else [])


class FeedForward:
    """MLP: `n_layers` hidden layers of `hidden_dim` units with `hidden_act`,
    followed by one linear output layer of size `out_dim` with `out_act`,
    optionally rescaled by `out_scale` (used by the Actor: tanh output * Imax).
    """

    def __init__(self, in_dim, hidden_dim, n_layers, out_dim, rng,
                 hidden_act="silu", out_act="identity", out_scale=1.0):
        self.layers = []
        d = in_dim
        for _ in range(n_layers):
            self.layers.append(Linear(d, hidden_dim, rng))
            d = hidden_dim
        self.out_layer = Linear(d, out_dim, rng)
        self.hidden_act = ACTIVATIONS[hidden_act]
        self.out_act = ACTIVATIONS[out_act]
        self.out_scale = out_scale

    def __call__(self, x):
        h = x
        for layer in self.layers:
            h = self.hidden_act(layer(h))
        o = self.out_act(self.out_layer(h))
        if self.out_scale != 1.0:
            o = o * self.out_scale
        return o

    def parameters(self):
        params = []
        for layer in self.layers:
            params += layer.parameters()
        params += self.out_layer.parameters()
        return params


class GRUCell:
    """One GRU layer, implementing Eq. (6)-(9) of the paper."""

    def __init__(self, input_dim, hidden_dim, rng, gate_activation="tanh"):
        self.Hp = Tensor(_glorot(input_dim, hidden_dim, rng), requires_grad=True)
        self.Hz = Tensor(_glorot(input_dim, hidden_dim, rng), requires_grad=True)
        self.Hh = Tensor(_glorot(input_dim, hidden_dim, rng), requires_grad=True)
        self.Up = Tensor(_glorot(hidden_dim, hidden_dim, rng), requires_grad=True)
        self.Uz = Tensor(_glorot(hidden_dim, hidden_dim, rng), requires_grad=True)
        self.Uh = Tensor(_glorot(hidden_dim, hidden_dim, rng), requires_grad=True)
        self.bp = Tensor(np.zeros((1, hidden_dim)), requires_grad=True)
        self.bz = Tensor(np.zeros((1, hidden_dim)), requires_grad=True)
        self.bh = Tensor(np.zeros((1, hidden_dim)), requires_grad=True)
        self.hidden_dim = hidden_dim
        if gate_activation == "tanh":
            self.gate_act = ACTIVATIONS["tanh"]
            self.cand_act = ACTIVATIONS["tanh"]
        elif gate_activation == "sigmoid_tanh":
            self.gate_act = ACTIVATIONS["sigmoid"]
            self.cand_act = ACTIVATIONS["tanh"]
        else:
            raise ValueError("gate_activation must be 'tanh' or 'sigmoid_tanh'")

    def step(self, x_k, h_prev):
        p = self.gate_act(x_k.matmul(self.Hp) + h_prev.matmul(self.Up) + self.bp)
        z = self.gate_act(x_k.matmul(self.Hz) + h_prev.matmul(self.Uz) + self.bz)
        cand = self.cand_act(x_k.matmul(self.Hh) + (p * h_prev).matmul(self.Uh) + self.bh)
        h = (Tensor(1.0) - z) * h_prev + z * cand
        return h

    def parameters(self):
        return [self.Hp, self.Hz, self.Hh, self.Up, self.Uz, self.Uh, self.bp, self.bz, self.bh]


class GRUStack:
    """Stack of `n_layers` GRUCells. `forward` consumes a length-(W+1) sequence
    of (batch, 1) Tensors (the past+current signal window {S_u}_{u=t-W}^{t})
    and returns the hidden state h_W of the *last* layer, exactly as
    described in Section 3.2.1 ("we take as output the hidden state for
    k = W of the last layer").
    """

    def __init__(self, n_layers, input_dim, hidden_dim, rng, gate_activation="tanh"):
        self.cells = []
        d = input_dim
        for _ in range(n_layers):
            self.cells.append(GRUCell(d, hidden_dim, rng, gate_activation))
            d = hidden_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

    def forward(self, x_seq):
        batch = x_seq[0].shape[0]
        layer_input = x_seq
        final_h = None
        for cell in self.cells:
            h = Tensor(np.zeros((batch, cell.hidden_dim)))
            outputs = []
            for x_k in layer_input:
                h = cell.step(x_k, h)
                outputs.append(h)
            layer_input = outputs
            final_h = h
        return final_h

    def parameters(self):
        params = []
        for c in self.cells:
            params += c.parameters()
        return params


class Adam:
    """Standard Adam optimiser (the paper: 'Weighted ADAM optimiser with a
    scheduler'; the scheduler itself is not fully specified, so a simple
    step-decay helper is provided separately via `StepScheduler`).
    """

    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        self.params = list(params)
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.wd = weight_decay
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]
        self.t = 0

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad
            if self.wd:
                g = g + self.wd * p.data
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g * g)
            mhat = self.m[i] / (1 - self.b1 ** self.t)
            vhat = self.v[i] / (1 - self.b2 ** self.t)
            p.data -= self.lr * mhat / (np.sqrt(vhat) + self.eps)

    def set_lr(self, lr):
        self.lr = lr


class StepScheduler:
    """Simple step-decay scheduler: multiply lr by `gamma` every `step_size` calls."""

    def __init__(self, optimizer, step_size=2000, gamma=0.7):
        self.opt = optimizer
        self.step_size = step_size
        self.gamma = gamma
        self.base_lr = optimizer.lr
        self.calls = 0

    def step(self):
        self.calls += 1
        if self.calls % self.step_size == 0:
            self.opt.set_lr(self.opt.lr * self.gamma)


def soft_update(target_params, source_params, tau):
    """Polyak/soft update: target <- (1-tau)*target + tau*source. tau=0.001 in the paper."""
    for tp, sp in zip(target_params, source_params):
        tp.data = (1.0 - tau) * tp.data + tau * sp.data


def hard_copy(target_params, source_params):
    for tp, sp in zip(target_params, source_params):
        tp.data = sp.data.copy()
