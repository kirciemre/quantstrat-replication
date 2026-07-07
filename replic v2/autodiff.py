"""
autodiff.py
-----------
A tiny reverse-mode automatic-differentiation engine operating on NumPy
arrays.  It exists for one reason: PyTorch / TensorFlow / JAX cannot be
installed in this environment (the package registry blocks them), but the
paper's algorithms (GRU + Actor-Critic DDPG, trained with gradient descent)
still need real gradients, not hand-derived ones.

The design follows the well-known "micrograd" pattern (Karpathy, 2020):
every operation builds a small graph node (`Tensor`) that remembers how it
was produced (`_prev`) and how to push a gradient back into its inputs
(`_backward`). Calling `.backward()` on a scalar (or a tensor with an
explicit seed gradient) walks the graph in reverse topological order and
accumulates `.grad` on every node that requested one.

Only the operations actually needed by the GRU / feed-forward nets in this
project are implemented: elementwise +,-,*,/, matmul, sum/mean, tanh,
sigmoid, SiLU, LeakyReLU, softmax, concatenation and power. Broadcasting is
supported the same way NumPy supports it (bias vectors added to batches),
via `_unbroadcast`, which sums a gradient back down to its original shape.

This module is deliberately small and is unit-tested against numerical
(finite-difference) gradients in `tests/test_autodiff.py` — do not trust it
blindly, but it has been checked.
"""
from __future__ import annotations
import numpy as np


class Tensor:
    __slots__ = ("data", "grad", "requires_grad", "_backward", "_prev", "_op", "name")

    def __init__(self, data, requires_grad=False, _children=(), _op="", name=""):
        self.data = np.asarray(data, dtype=np.float64)
        self.requires_grad = requires_grad
        self.grad = None
        self._backward = lambda: None
        self._prev = tuple(_children)
        self._op = _op
        self.name = name

    # ---------------------------------------------------------- utilities
    @staticmethod
    def _wrap(x):
        return x if isinstance(x, Tensor) else Tensor(x)

    def _ensure_grad(self):
        if self.grad is None:
            self.grad = np.zeros_like(self.data, dtype=np.float64)

    @staticmethod
    def _unbroadcast(g, shape):
        """Sum-reduce gradient `g` down to `shape`, undoing NumPy broadcasting."""
        g = np.asarray(g, dtype=np.float64)
        while g.ndim > len(shape):
            g = g.sum(axis=0)
        for i, s in enumerate(shape):
            if s == 1 and g.shape[i] != 1:
                g = g.sum(axis=i, keepdims=True)
        return g.reshape(shape)

    @property
    def shape(self):
        return self.data.shape

    def zero_grad(self):
        self.grad = None

    def item(self):
        return float(self.data.reshape(-1)[0])

    def __repr__(self):
        return f"Tensor(shape={self.data.shape}, op={self._op!r}, name={self.name!r})"

    # ------------------------------------------------------------- add/sub
    def __add__(self, other):
        other = Tensor._wrap(other)
        out = Tensor(self.data + other.data,
                      requires_grad=(self.requires_grad or other.requires_grad),
                      _children=(self, other), _op="add")

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                self.grad += Tensor._unbroadcast(out.grad, self.data.shape)
            if other.requires_grad:
                other._ensure_grad()
                other.grad += Tensor._unbroadcast(out.grad, other.data.shape)
        out._backward = _backward
        return out

    __radd__ = __add__

    def __neg__(self):
        out = Tensor(-self.data, requires_grad=self.requires_grad, _children=(self,), _op="neg")

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                self.grad += -out.grad
        out._backward = _backward
        return out

    def __sub__(self, other):
        return self + (-Tensor._wrap(other))

    def __rsub__(self, other):
        return Tensor._wrap(other) + (-self)

    # ------------------------------------------------------------- mul/div
    def __mul__(self, other):
        other = Tensor._wrap(other)
        out = Tensor(self.data * other.data,
                      requires_grad=(self.requires_grad or other.requires_grad),
                      _children=(self, other), _op="mul")

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                self.grad += Tensor._unbroadcast(out.grad * other.data, self.data.shape)
            if other.requires_grad:
                other._ensure_grad()
                other.grad += Tensor._unbroadcast(out.grad * self.data, other.data.shape)
        out._backward = _backward
        return out

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = Tensor._wrap(other)
        out = Tensor(self.data / other.data,
                      requires_grad=(self.requires_grad or other.requires_grad),
                      _children=(self, other), _op="div")

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                self.grad += Tensor._unbroadcast(out.grad / other.data, self.data.shape)
            if other.requires_grad:
                other._ensure_grad()
                other.grad += Tensor._unbroadcast(-out.grad * self.data / (other.data ** 2), other.data.shape)
        out._backward = _backward
        return out

    def __pow__(self, p):
        out = Tensor(self.data ** p, requires_grad=self.requires_grad, _children=(self,), _op=f"pow{p}")

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                self.grad += (p * self.data ** (p - 1)) * out.grad
        out._backward = _backward
        return out

    # ------------------------------------------------------------- matmul
    def matmul(self, other):
        other = Tensor._wrap(other)
        out = Tensor(self.data @ other.data,
                      requires_grad=(self.requires_grad or other.requires_grad),
                      _children=(self, other), _op="matmul")

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                self.grad += out.grad @ other.data.T
            if other.requires_grad:
                other._ensure_grad()
                other.grad += self.data.T @ out.grad
        out._backward = _backward
        return out

    def __matmul__(self, other):
        return self.matmul(other)

    # ------------------------------------------------------------- reduce
    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims),
                      requires_grad=self.requires_grad, _children=(self,), _op="sum")

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                g = out.grad
                if axis is not None and not keepdims:
                    g = np.expand_dims(g, axis)
                self.grad += np.ones_like(self.data) * g
        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        n = self.data.size if axis is None else self.data.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

    # ---------------------------------------------------------- activations
    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, requires_grad=self.requires_grad, _children=(self,), _op="tanh")

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                self.grad += (1.0 - t ** 2) * out.grad
        out._backward = _backward
        return out

    def sigmoid(self):
        clipped = np.clip(self.data, -30.0, 30.0)
        sg = 1.0 / (1.0 + np.exp(-clipped))
        out = Tensor(sg, requires_grad=self.requires_grad, _children=(self,), _op="sigmoid")

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                self.grad += sg * (1.0 - sg) * out.grad
        out._backward = _backward
        return out

    def silu(self):
        """SiLU / Sigmoid Linear Unit: x * sigmoid(x). Used in Actor/Critic hidden layers (paper: 'SiLu activation')."""
        clipped = np.clip(self.data, -30.0, 30.0)
        sg = 1.0 / (1.0 + np.exp(-clipped))
        val = self.data * sg
        out = Tensor(val, requires_grad=self.requires_grad, _children=(self,), _op="silu")

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                self.grad += (sg + clipped * sg * (1.0 - sg)) * out.grad
        out._backward = _backward
        return out

    def leaky_relu(self, negative_slope=0.01):
        mask = (self.data > 0).astype(np.float64)
        val = np.where(self.data > 0, self.data, negative_slope * self.data)
        out = Tensor(val, requires_grad=self.requires_grad, _children=(self,), _op="leaky_relu")

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                self.grad += (mask + (1 - mask) * negative_slope) * out.grad
        out._backward = _backward
        return out

    def log(self, eps=1e-9):
        val = np.log(self.data + eps)
        out = Tensor(val, requires_grad=self.requires_grad, _children=(self,), _op="log")

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                self.grad += out.grad / (self.data + eps)
        out._backward = _backward
        return out

    def softmax(self, axis=-1):
        z = self.data - np.max(self.data, axis=axis, keepdims=True)
        e = np.exp(z)
        sm = e / np.sum(e, axis=axis, keepdims=True)
        out = Tensor(sm, requires_grad=self.requires_grad, _children=(self,), _op="softmax")

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                g = out.grad
                dot = np.sum(g * sm, axis=axis, keepdims=True)
                self.grad += sm * (g - dot)
        out._backward = _backward
        return out

    # ------------------------------------------------------------- concat
    @staticmethod
    def cat(tensors, axis=-1):
        tensors = list(tensors)
        arrs = [t.data for t in tensors]
        out_data = np.concatenate(arrs, axis=axis)
        req = any(t.requires_grad for t in tensors)
        out = Tensor(out_data, requires_grad=req, _children=tuple(tensors), _op="cat")
        sizes = [a.shape[axis] for a in arrs]

        def _backward():
            if not req:
                return
            idx = 0
            g = out.grad
            for t, size in zip(tensors, sizes):
                sl = [slice(None)] * g.ndim
                sl[axis] = slice(idx, idx + size)
                if t.requires_grad:
                    t._ensure_grad()
                    t.grad += g[tuple(sl)]
                idx += size
        out._backward = _backward
        return out

    # ------------------------------------------------------------- backward
    def backward(self, grad=None):
        topo = []
        visited = set()

        def build(v):
            if id(v) not in visited:
                visited.add(id(v))
                for p in v._prev:
                    build(p)
                topo.append(v)
        build(self)

        self.grad = np.ones_like(self.data) if grad is None else np.array(grad, dtype=np.float64)
        for v in reversed(topo):
            v._backward()


def zeros(shape):
    return Tensor(np.zeros(shape))


def constant(x):
    return Tensor(x, requires_grad=False)
