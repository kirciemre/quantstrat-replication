"""
Actor (policy) network for the DDPG agent.

The Actor is the policy pi: it maps a state to an action -- the new INVENTORY
LEVEL I_{t+1} to hold (not a trade quantity; paper: "returns a new level of
inventory to be held I_{t+1}"). It is a plain MLP (Linear -> SiLU blocks) ending
in a tanh scaled by I_max, so the output is STRUCTURALLY bounded to
[-I_max, I_max] (paper: "final layer has tanh activation ... scaled by Imax").

This is why the inventory bound is NOT enforced in the reward -- it lives here,
baked into the architecture, so the agent cannot propose an out-of-range position.

The state is a generic vector of width state_dim; the caller decides its contents
(hid-DDPG uses (S_t, I_t, o_t)). Keeping the Actor agnostic to the state contents
is what lets all three algorithm variants share it.

Arg names follow the paper: l_NN = number of hidden layers, d_NN = hidden width.
"""

import torch
import torch.nn as nn


class Actor(nn.Module):
    def __init__(self, state_dim, d_NN, l_NN, I_max):
        super().__init__()
        self.I_max = I_max                                 # needed in forward for output scaling

        # Width chain: state_dim -> d_NN -> ... -> d_NN -> 1.
        # l_NN = number of hidden blocks in the loop (total Linear = l_NN + 2).
        layers = [nn.Linear(state_dim, d_NN), nn.SiLU()]   # input block

        for _ in range(l_NN):                              # hidden blocks (d_NN -> d_NN)
            layers.append(nn.Linear(d_NN, d_NN))
            layers.append(nn.SiLU())

        layers.append(nn.Linear(d_NN, 1))                  # output block: NO activation
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
    # Bound check: extreme input -> tanh saturates -> output pins near +-I_max.
    actor = Actor(state_dim=3, d_NN=20, l_NN=4, I_max=10)
    state = torch.randn(8, 3) * 1000        # matches state_dim=3
    a = actor(state)
    print(f"a.shape == {a.shape}")           # expect (8, 1)
    print(f"a.abs().max() == {a.abs().max()}")   # expect ~10