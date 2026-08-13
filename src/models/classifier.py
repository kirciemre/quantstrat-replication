"""
GRU regime classifier (prob-DDPG, step 1).

Approximates the posterior probability that theta is in each regime given the
signal window:  Phi_{t,k} = P(theta_t = phi_k | {S_u}_{u=t-W}^t),  k = 1..K.

Architecture (Table 2): GRU (dl layers, dh hidden) -> feed-forward net (SiLU
hidden layers) -> K logits. Trained with categorical cross-entropy against the
TRUE theta regime label at time t (available from the simulator). At inference
the logits are soft-maxed to give the posterior vector fed to the DDPG state.

Mirrors GRURegressor (reg-DDPG) but outputs K class logits instead of a scalar
next-value estimate. num_classes = number of theta regimes (3 in all scenarios).
"""

import torch
import torch.nn as nn


class GRUClassifier(nn.Module):
    def __init__(self, hidden_dim=20, num_layers=5, ffn_layers=5,
                 ffn_hidden=64, num_classes=3):
        super().__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=hidden_dim,
                          num_layers=num_layers, batch_first=True)

        layers = []
        width = hidden_dim
        for _ in range(ffn_layers - 1):
            layers.append(nn.Linear(width, ffn_hidden))
            layers.append(nn.SiLU())
            width = ffn_hidden
        layers.append(nn.Linear(width, num_classes))   # logits (softmax applied by loss / at inference)
        self.ffn = nn.Sequential(*layers)

    def forward(self, windows):
        """
        windows : (batch, W+1)     signal windows
        returns : (batch, num_classes)   raw class logits (NOT soft-maxed)
        """
        x = windows.unsqueeze(-1)              # (batch, W+1, 1)
        _, hidden = self.gru(x)
        return self.ffn(hidden[-1])            # last layer's final hidden state -> logits

    def posterior(self, windows):
        """Soft-maxed posterior (batch, num_classes) for use as a DDPG state feature."""
        return torch.softmax(self.forward(windows), dim=1)


if __name__ == "__main__":
    clf = GRUClassifier(num_classes=3)
    x = torch.randn(8, 11)
    logits = clf(x)
    post = clf.posterior(x)
    assert logits.shape == (8, 3), logits.shape
    assert torch.allclose(post.sum(1), torch.ones(8), atol=1e-5)
    print("ok: logits", tuple(logits.shape), " posterior rows sum to 1")
