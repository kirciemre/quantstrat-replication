"""
Critic (value) network for the DDPG agent.

The Critic is the value function Q: it takes a state AND an action and outputs a
single number, the Q-value -- an estimate of the expected discounted future
reward from taking that action in that state. It is how the agent judges how
good an action was. During training the Actor learns to produce actions the
Critic scores highly, while the Critic learns to score accurately (against the
Bellman target -- that logic lives in algos/ddpg.py, not here).

Same MLP pattern as the Actor, with three differences:
  1. input width is state_dim + action_dim (it scores a state-action PAIR),
  2. output is an UNBOUNDED scalar -- a Q-value can be any real number, so there
     is no tanh and no scaling (that was Actor-specific: actions have limits,
     values do not),
  3. forward takes (state, action) and concatenates them before the MLP.
"""

import torch.nn as nn
import torch


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, n_layers):
        super().__init__()

        # Width chain: (state_dim + action_dim) -> hidden -> ... -> hidden -> 1.
        # Same list-then-Sequential pattern as the Actor; n_layers = hidden
        # blocks in the loop (total Linear count = n_layers + 2).
        layers = [nn.Linear(state_dim + action_dim, hidden_dim), nn.SiLU()]  # input block

        for _ in range(n_layers):                              # hidden blocks (hidden -> hidden)
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())

        layers.append(nn.Linear(hidden_dim, 1))                # output block: raw scalar, NO activation
        self.net = nn.Sequential(*layers)

    def forward(self, state, action):
        """
        state  : (batch, state_dim)
        action : (batch, action_dim)
        returns : (batch, 1)   the (unbounded) Q-value of each state-action pair
        """
        # Join state and action side-by-side (dim=1 = along features) so each row
        # becomes [state_features..., action]. dim=0 would wrongly stack them as
        # extra rows; the batch dimension must stay fixed.
        return self.net(torch.cat([state, action], dim=1))


if __name__ == "__main__":
    critic = Critic(state_dim=12, action_dim=1, hidden_dim=20, n_layers=4)
    state = torch.randn(8, 12)
    action = torch.randn(8, 1)
    q = critic(state, action)

    print(f"q.shape == {q.shape}")