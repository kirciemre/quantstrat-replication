"""
GRU regressor for reg-DDPG / reg-PPO (paper Section 3.3.3).

Stage 1 predicts the NEXT signal value S~_{t+1} directly from the window
{S_u}_{u=t-W}^{t}, trained by MSE against the actual S_{t+1}. Per the paper the
LAST layer has a SiLU activation. Once trained it is FROZEN; the RL agent then
uses features G_t = (S_t, I_t, S~_{t+1}) -> state_dim = 3.

Architecture (paper Table for prob/reg two-step): GRU hidden 20, 5 layers;
FFN 5 layers of width 64; W = 50.
"""

import torch
import torch.nn as nn


class GRURegressor(nn.Module):
    def __init__(self, hidden_dim=20, num_layers=5, ffn_layers=5, ffn_hidden=64):
        super().__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=hidden_dim,
                          num_layers=num_layers, batch_first=True)
        layers = []
        w = hidden_dim
        for _ in range(ffn_layers - 1):
            layers += [nn.Linear(w, ffn_hidden), nn.SiLU()]
            w = ffn_hidden
        layers.append(nn.Linear(w, 1))
        layers.append(nn.SiLU())              # paper: last layer has SiLU
        self.ffn = nn.Sequential(*layers)

    def forward(self, windows):
        # windows: (batch, W+1) -> (batch, W+1, 1)
        x = windows.unsqueeze(-1)
        _, hidden = self.gru(x)
        return self.ffn(hidden[-1])           # (batch, 1) = S~_{t+1}


# import torch
# import torch.nn as nn


# class GRURegressor(nn.Module):
#     """Predict S_{t+1} from the signal window S_{t-W:t}."""

#     def __init__(
#         self,
#         hidden_dim: int = 20,
#         num_layers: int = 5,
#         ffn_layers: int = 5,
#         ffn_hidden: int = 64,
#     ) -> None:
#         super().__init__()

#         self.gru = nn.GRU(
#             input_size=1,
#             hidden_size=hidden_dim,
#             num_layers=num_layers,
#             batch_first=True,
#         )

#         layers: list[nn.Module] = []
#         input_width = hidden_dim

#         for _ in range(ffn_layers - 1):
#             layers.append(nn.Linear(input_width, ffn_hidden))
#             layers.append(nn.SiLU())
#             input_width = ffn_hidden

#         layers.append(nn.Linear(input_width, 1))
#         layers.append(nn.SiLU())

#         self.ffn = nn.Sequential(*layers)

#     def forward(self, windows: torch.Tensor) -> torch.Tensor:
#         # windows shape: (batch_size, sequence_length)
#         x = windows.unsqueeze(-1)

#         _, hidden = self.gru(x)

#         # Last GRU layer's final hidden state
#         hidden_last = hidden[-1]

#         prediction = self.ffn(hidden_last)

#         return prediction
