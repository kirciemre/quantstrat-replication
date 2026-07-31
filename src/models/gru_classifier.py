"""
Offline GRU regime classifier for prob-DDPG / prob-PPO.

The "prob" variant replaces hid's aux-trained encoding with the POSTERIOR over
the non-stationary regimes. The classifier is trained OFFLINE on labeled paths
(cross-entropy against the true regime indices), then FROZEN; the RL agent then
trains on states (S_t, I_t, Phi) where Phi is the concatenated softmax outputs.

Separate head per switching parameter:
  Scenario 1 (theta):            head_theta (3-way)                  -> Phi dim 3
  Scenario 2 (theta, kappa):     + head_kappa (2-way)                -> Phi dim 5
  Scenario 3 (theta, kappa, sig):+ head_sigma (2-way)                -> Phi dim 7

Which heads are active is set by n_kappa / n_sigma being > 0. Phi is the
concatenation of the active heads' full posterior vectors (they each sum to 1).
"""

import torch
import torch.nn as nn


class GRUClassifier(nn.Module):
    def __init__(self, d_h, d_l, n_theta=3, n_kappa=0, n_sigma=0, input_size=1):
        """
        d_h, d_l : GRU hidden size and layers (match the hid encoder's capacity).
        n_theta  : number of theta regimes (always active, >=2).
        n_kappa  : number of kappa regimes, or 0 if kappa is constant (Sc1).
        n_sigma  : number of sigma regimes, or 0 if sigma is constant (Sc1/Sc2).
        """
        super().__init__()
        self.n_theta = n_theta
        self.n_kappa = n_kappa
        self.n_sigma = n_sigma

        self.gru = nn.GRU(input_size, d_h, d_l, batch_first=True)
        self.head_theta = nn.Linear(d_h, n_theta)
        self.head_kappa = nn.Linear(d_h, n_kappa) if n_kappa else None
        self.head_sigma = nn.Linear(d_h, n_sigma) if n_sigma else None

    @property
    def phi_dim(self):
        """Size of the concatenated posterior Phi (state uses 2 + phi_dim)."""
        return self.n_theta + self.n_kappa + self.n_sigma

    def _last_hidden(self, window):
        # window: (batch, W+1) -> (batch, W+1, 1) -> GRU -> last step hidden
        if window.dim() == 2:
            window = window.unsqueeze(-1)
        out, _ = self.gru(window)
        return out[:, -1, :]                     # (batch, d_h)

    def logits(self, window):
        """Raw logits per active head (for cross-entropy training)."""
        h = self._last_hidden(window)
        out = {"theta": self.head_theta(h)}
        if self.head_kappa is not None:
            out["kappa"] = self.head_kappa(h)
        if self.head_sigma is not None:
            out["sigma"] = self.head_sigma(h)
        return out

    def phi(self, window):
        """
        Posterior feature Phi = concat of softmax(head) over active heads.
        Used at RL train/eval time (classifier frozen). Returns (batch, phi_dim).
        """
        h = self._last_hidden(window)
        parts = [torch.softmax(self.head_theta(h), dim=-1)]
        if self.head_kappa is not None:
            parts.append(torch.softmax(self.head_kappa(h), dim=-1))
        if self.head_sigma is not None:
            parts.append(torch.softmax(self.head_sigma(h), dim=-1))
        return torch.cat(parts, dim=-1)

    def forward(self, window):
        return self.phi(window)


if __name__ == "__main__":
    for name, nk, ns, exp in [("Sc1", 0, 0, 3), ("Sc2", 2, 0, 5), ("Sc3", 2, 2, 7)]:
        clf = GRUClassifier(d_h=10, d_l=1, n_theta=3, n_kappa=nk, n_sigma=ns)
        w = torch.randn(8, 11)          # (batch, W+1)
        lg = clf.logits(w)
        ph = clf.phi(w)
        assert clf.phi_dim == exp, (name, clf.phi_dim, exp)
        # each head's posterior sums to 1
        s = torch.softmax(lg["theta"], -1).sum(-1)
        print(f"{name}: heads={list(lg)}  phi_dim={clf.phi_dim}  phi_shape={tuple(ph.shape)}  "
              f"theta_post_sums~1: {torch.allclose(s, torch.ones(8), atol=1e-5)}")