"""
Actor (policy) network for the DDPG agent.

The Actor is the policy pi: it maps a state to an action -- the new inventory
I_{t+1} to hold. It is a plain MLP (Linear -> SiLU blocks) ending in a tanh
scaled by I_max, so its output is STRUCTURALLY bounded to [-I_max, I_max]
(paper: "final layer has tanh activation, whose output is then scaled by Imax").

Note: this is why the inventory bound is deliberately NOT enforced in the reward
-- it lives here, baked into the architecture, so the agent physically cannot
propose an out-of-range position.

The state is a generic vector of width `state_dim`; the caller decides what goes
in it (e.g. hid-DDPG uses (S_t, I_t, o_t)). Keeping the Actor agnostic to the
state contents is what lets all three algorithm variants share it.
"""

import torch
import torch.nn as nn


class Actor(nn.Module):
    def __init__(self, state_dim, hidden_dim, n_layers, I_max):
        super().__init__()
        self.I_max = I_max                                   # needed in forward for output scaling

        # Build the MLP as a list, then unpack into nn.Sequential.
        # Width chain: state_dim -> hidden_dim -> ... -> hidden_dim -> 1.
        # Convention: n_layers = number of hidden blocks in the loop, so the
        # total Linear count is n_layers + 2 (input + loop + output). Pin this
        # down when matching the paper's layer counts (Tables 1-3).
        layers = [nn.Linear(state_dim, hidden_dim), nn.SiLU()]   # input block

        for _ in range(n_layers):                               # hidden blocks (hidden -> hidden)
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())

        layers.append(nn.Linear(hidden_dim, 1))                 # output block: NO activation here
                                                                # (tanh scaling happens in forward)
        self.net = nn.Sequential(*layers)

    def forward(self, state):
        """
        state : (batch, state_dim)
        returns : (batch, 1)   the action I_{t+1}, guaranteed in [-I_max, I_max]
        """
        # tanh squashes to (-1, 1); scaling by I_max stretches to (-I_max, I_max).
        return torch.tanh(self.net(state)) * self.I_max


if __name__ == "__main__":
    # Bound check: feed a deliberately extreme input so tanh saturates near +-1,
    # confirming the output pins near +-I_max instead of escaping the range.
    actor = Actor(12, 20, 4, 10)
    state = torch.randn(8, 12) * 1000
    a = actor(state)

    print(f"a.shape == {a.shape}")
    print(f"a.abs().max() == {a.abs().max()}")