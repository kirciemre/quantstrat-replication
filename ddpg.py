import torch
import torch.nn as nn

class Actor(nn.Module):
    """
    DDPG Actor Network.
    Takes the state representation (from GRU, classifier, or regressor) + inventory and signal,
    and outputs the target inventory level I_{t+1} in [-I_max, I_max].
    """
    def __init__(self, state_dim, num_layers=5, hidden_dim=64, max_action=10.0):
        super().__init__()
        self.max_action = max_action
        
        layers = []
        in_dim = state_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.SiLU())
            in_dim = hidden_dim
            
        # Output layer with tanh scaled by max_action
        layers.append(nn.Linear(in_dim, 1))
        layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, state):
        return self.net(state) * self.max_action

class Critic(nn.Module):
    """
    DDPG Critic Network.
    Takes the state representation and the action (I_{t+1}) and predicts the Q-value.
    """
    def __init__(self, state_dim, num_layers=5, hidden_dim=64):
        super().__init__()
        
        layers = []
        in_dim = state_dim + 1 # state + action
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.SiLU())
            in_dim = hidden_dim
            
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, state, action):
        # Concatenate state and action
        x = torch.cat([state, action], dim=-1)
        return self.net(x)

def soft_update(target_net, source_net, tau=0.001):
    """
    Soft update target network weights:
    theta_target = tau * theta_source + (1 - tau) * theta_target
    """
    for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)
