import torch
import torch.nn as nn

class GRUEncoder(nn.Module):
    """
    Used in hid-DDPG.
    Trained online using an auxiliary next-signal prediction head.
    """
    def __init__(self, input_dim=1, hidden_dim=10, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        # Auxiliary prediction head: linear layer with LeakyReLU
        self.predict_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.LeakyReLU(negative_slope=0.01)
        )

    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        out, _ = self.gru(x)
        # Extract the hidden state of the last layer at the last sequence step
        o_t = out[:, -1, :] # shape: (batch_size, hidden_dim)
        pred = self.predict_head(o_t) # shape: (batch_size, 1)
        return o_t, pred

class GRUClassifier(nn.Module):
    """
    Used in prob-DDPG.
    Trained offline to estimate posterior probabilities of regimes.
    """
    def __init__(self, input_dim=1, hidden_dim=20, num_layers=5, num_classes=3, ffn_layers=5, ffn_hidden=64):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        # FFN: 5 layers of 64 hidden nodes with SiLU activation
        layers = []
        in_dim = hidden_dim
        for _ in range(ffn_layers - 1):
            layers.append(nn.Linear(in_dim, ffn_hidden))
            layers.append(nn.SiLU())
            in_dim = ffn_hidden
        
        # Output layer
        layers.append(nn.Linear(in_dim, num_classes))
        self.ffn = nn.Sequential(*layers)

    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        out, _ = self.gru(x)
        o_t = out[:, -1, :] # shape: (batch_size, hidden_dim)
        logits = self.ffn(o_t) # shape: (batch_size, num_classes)
        probs = torch.softmax(logits, dim=-1) # shape: (batch_size, num_classes)
        return probs, logits

class GRURegressor(nn.Module):
    """
    Used in reg-DDPG.
    Trained offline to predict the next signal value S_{t+1}.
    """
    def __init__(self, input_dim=1, hidden_dim=20, num_layers=5, ffn_layers=5, ffn_hidden=64):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        # FFN: 5 layers of 64 hidden nodes with SiLU activation
        layers = []
        in_dim = hidden_dim
        for _ in range(ffn_layers - 1):
            layers.append(nn.Linear(in_dim, ffn_hidden))
            layers.append(nn.SiLU())
            in_dim = ffn_hidden
            
        # Final regression layer (minimizing squared error)
        # Note: the paper says the last layer has SiLU activation function
        layers.append(nn.Linear(in_dim, 1))
        layers.append(nn.SiLU())
        self.ffn = nn.Sequential(*layers)

    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        out, _ = self.gru(x)
        o_t = out[:, -1, :] # shape: (batch_size, hidden_dim)
        pred = self.ffn(o_t) # shape: (batch_size, 1)
        return pred
