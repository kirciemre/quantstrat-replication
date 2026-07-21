import torch
import torch.nn as nn


class GRURegressor(nn.Module):
    """Predict S_{t+1} from the signal window S_{t-W:t}."""

    def __init__(
        self,
        hidden_dim: int = 20,
        num_layers: int = 5,
        ffn_layers: int = 5,
        ffn_hidden: int = 64,
    ) -> None:
        super().__init__()

        self.gru = nn.GRU(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        layers: list[nn.Module] = []
        input_width = hidden_dim

        for _ in range(ffn_layers - 1):
            layers.append(nn.Linear(input_width, ffn_hidden))
            layers.append(nn.SiLU())
            input_width = ffn_hidden

        layers.append(nn.Linear(input_width, 1))
        layers.append(nn.SiLU())

        self.ffn = nn.Sequential(*layers)

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        # windows shape: (batch_size, sequence_length)
        x = windows.unsqueeze(-1)

        _, hidden = self.gru(x)

        # Last GRU layer's final hidden state
        hidden_last = hidden[-1]

        prediction = self.ffn(hidden_last)

        return prediction
