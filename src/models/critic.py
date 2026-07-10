"""
Critic (value) network for the DDPG agent.

The Critic is the value function Q: it takes a state AND an action and outputs a
single number, the Q-value -- an estimate of the expected discounted future
reward from taking that action in that state. During training the Actor learns
to produce actions the Critic scores highly, while the Critic learns to score
accurately against the Bellman target (that logic lives in ddpg.py, not here).

Paper: the Critic is a feed-forward net with (state_dim + action_dim) input
neurons -- "four input neurons" for hid-DDPG with the scalar-encoding state
(3 state features + 1 action) -- l_NN layers of d_NN hidden nodes with SiLU,
and a raw scalar output (Q-values are unbounded: no tanh, no scaling).

Arg names follow the paper: l_NN = number of hidden layers, d_NN = hidden width.
"""

import torch.nn as nn
import torch


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, d_NN, l_NN):
        super().__init__()

        # Width chain: (state_dim + action_dim) -> d_NN -> ... -> d_NN -> 1.
        # l_NN = number of hidden blocks in the loop (total Linear = l_NN + 2).
        layers = [nn.Linear(state_dim + action_dim, d_NN), nn.SiLU()]   # input block

        for _ in range(l_NN):                          # hidden blocks (d_NN -> d_NN)
            layers.append(nn.Linear(d_NN, d_NN))
            layers.append(nn.SiLU())

        layers.append(nn.Linear(d_NN, 1))              # output block: raw scalar, NO activation
        self.net = nn.Sequential(*layers)

    def forward(self, state, action):
        """
        state  : (batch, state_dim)
        action : (batch, action_dim)
        returns : (batch, 1)   the (unbounded) Q-value of each state-action pair
        """
        # Join state and action along features (dim=1): each row -> [state..., action].
        return self.net(torch.cat([state, action], dim=1))


if __name__ == "__main__":
    # hid-DDPG with scalar-encoding state: state_dim = 3 -> Critic input = 4.
    critic = Critic(state_dim=3, action_dim=1, d_NN=20, l_NN=4)
    state = torch.randn(8, 3)
    action = torch.randn(8, 1)
    q = critic(state, action)
    print(f"q.shape == {q.shape}")       # expect (8, 1)