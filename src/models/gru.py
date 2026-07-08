"""
GRU signal encoder.

Compresses a look-back window of the signal, {S_{t-W} .. S_t}, into a single
d_h-dimensional summary vector o_t (the final hidden state). This encoding is
the feature the three algorithms diverge on:
  - hid-DDPG  feeds o_t straight to the agent,
  - prob-DDPG puts a classifier head on it (regime posteriors),
  - reg-DDPG  puts a regressor head on it (next-value forecast).

This module is the ENCODER ONLY -- a pure forward pass, no head and no opinion
about its loss. Who trains the GRU, and against what objective, differs per
algorithm and lives in the algos/train files (like reward.py, this is pure
mechanism). Heads live in models/heads.py.
"""

import torch
import torch.nn as nn


class GRUEncoder(nn.Module):
    def __init__(self, hidden_size, num_layers, input_size=1):
        # hidden_size = d_h, num_layers = d_l (paper Tables 1-3).
        # input_size = 1 because the signal is a scalar at each timestep.
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        """
        x : (batch, W+1)     a batch of signal windows (scalar per timestep)
        returns : (batch, d_h)   the last layer's final hidden state = o_t
        """
        # nn.GRU expects (batch, seq_len, input_size); our input has no feature
        # dim, so add a trailing 1 -> (batch, W+1, 1). Forgetting this is the
        # classic bug (it silently reinterprets the timesteps as features).
        x = x.unsqueeze(-1)

        # h_n is (num_layers, batch, d_h): the final hidden state of each layer.
        # We take the LAST layer's final state as the encoding (discard the
        # per-timestep `output`). h_n[-1] is (batch, d_h) regardless of depth.
        _, h_n = self.gru(x)

        return h_n[-1]


if __name__ == "__main__":
    # Shape check: output width = d_h and is independent of num_layers.
    gru = GRUEncoder(hidden_size=10, num_layers=2)
    x = torch.randn(8, 11)                 # batch 8, window W+1 = 11
    o = gru(x)
    assert o.shape == (8, 10)