"""
GRU signal encoder (hid-DDPG).

Reads a look-back window of the signal and produces TWO scalar outputs from the
final hidden state h_W (last layer, k = W):

  o_t     : the ENCODING fed into the agent's state G_t.
            A linear projection of h_W (no activation). The paper states o_t
            encodes the temporal structure and is NOT the next-value prediction.
  s_pred  : the estimate S~_{t+1}, a linear layer + LeakyReLU on h_W. Used ONLY
            for the auxiliary MSE loss that trains the GRU (before Actor/Critic
            each iteration).

Argument names follow the paper: d_h = hidden size (Table 3), d_l = number of
GRU layers.

Dimension note (paper is internally inconsistent here): Table 3 gives d_h = 10
and the GRU mechanics give h_W in R^{d_h}, yet the output is stated as o_t in R^b
(a scalar per sample) and G_t in R^{b x 3}. The reconciling reading is that o_t
is a LINEAR PROJECTION of the d_h-dim h_W down to a scalar -> state_dim = 3.
`enc_dim` keeps this switchable: enc_dim=1 -> state_dim 3 (projection reading);
set enc_dim=d_h to feed the full hidden state -> state_dim 12 (the other reading
/ the working-code variant) for empirical comparison.
"""

import torch
import torch.nn as nn


class GRUEncoder(nn.Module):
    def __init__(self, d_h, d_l, input_size=1, enc_dim=1):
        # d_h = hidden size (Table 3), d_l = number of GRU layers.
        # input_size = 1 because the signal is a scalar at each timestep.
        # enc_dim = width of o_t fed to the state (1 = scalar projection).
        super().__init__()
        self.gru = nn.GRU(input_size, d_h, d_l, batch_first=True)
        self.enc_head = nn.Linear(d_h, enc_dim)                              # o_t (encoding, NO activation)
        self.pred_head = nn.Sequential(nn.Linear(d_h, 1), nn.LeakyReLU())    # S~_{t+1} (aux prediction)

    def forward(self, x):
        """
        x : (batch, W+1)          a batch of signal windows (scalar per timestep)
        returns:
          o_t    : (batch, enc_dim)  scalar encoding for the state (enc_dim=1)
          s_pred : (batch, 1)        the next-value estimate S~_{t+1} for the aux loss
        """
        x = x.unsqueeze(-1)                # (batch, W+1, 1) for nn.GRU
        _, h_n = self.gru(x)
        h_W = h_n[-1]                      # (batch, d_h) -- final hidden state, last layer
        o_t = self.enc_head(h_W)           # (batch, enc_dim) -- linear projection, encoding
        s_pred = self.pred_head(h_W)       # (batch, 1) -- LeakyReLU prediction, aux loss
        return o_t, s_pred


if __name__ == "__main__":
    # Shape check: o_t is (batch, enc_dim), s_pred is (batch, 1).
    enc = GRUEncoder(d_h=10, d_l=1, enc_dim=1)
    x = torch.randn(8, 11)                 # batch 8, window W+1 = 11
    o_t, s_pred = enc(x)
    assert o_t.shape == (8, 1), o_t.shape
    assert s_pred.shape == (8, 1), s_pred.shape
    print("ok: o_t", tuple(o_t.shape), " s_pred", tuple(s_pred.shape))