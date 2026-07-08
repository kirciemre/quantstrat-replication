"""
DDPG training algorithm (Deep Deterministic Policy Gradient).

This is the engine that TRAINS the Actor and Critic networks -- they are just
architecture; this class makes them learn. It implements the Actor-Critic
scheme from Section 3 of Macri, Jaimungal & Lillo (2025):

  - the Critic Q learns to predict the value of a state-action pair,
  - the Actor pi learns to output actions the Critic scores highly,

and they bootstrap each other over many iterations.

Four networks are kept: the main Actor/Critic (which learn) and frozen TARGET
copies of each. The targets exist to stabilise Critic training: the Critic's
learning target contains a Critic evaluation of the next state, so using the
same network on both sides makes the target chase the prediction ("dog chasing
its tail"). The frozen copies provide a slow-moving target instead; they are
nudged toward the mains only gradually via soft_update.

Method roles:
  update_critic  : fit the Critic to the Bellman target      (Eqs. 11-12)
  update_actor   : push the Actor toward higher-value actions (Eqs. 13-14)
  soft_update    : nudge one target network toward its main
  update_targets : soft-update both targets
  select_action  : deterministic action + exploration noise (for acting)
"""

import torch

from src.models.actor import Actor
from src.models.critic import Critic


class DDPG:
    def __init__(self, state_dim, action_dim, hidden_dim, n_layers, I_max, gamma, tau, lr):
        self.gamma = gamma      # discount factor for future rewards
        self.I_max = I_max      # inventory bound (used to clamp noisy actions)
        self.tau = tau          # soft-update rate: fraction the targets move toward the mains
        self.lr = lr

        # Main networks (these learn) + frozen target copies (stable Bellman target).
        self.actor = Actor(state_dim, hidden_dim, n_layers, I_max)
        self.actor_target = Actor(state_dim, hidden_dim, n_layers, I_max)
        self.critic = Critic(state_dim, action_dim, hidden_dim, n_layers)
        self.critic_target = Critic(state_dim, action_dim, hidden_dim, n_layers)

        # Start the targets as EXACT clones of the mains (else they'd hold
        # different random weights). Copies weight values across.
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers train the MAIN networks only. Targets are never trained by
        # gradients -- they change only via soft_update.
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)


    def update_critic(self, state, action, reward, next_state):
        """
        Fit the Critic to the Bellman target (Eqs. 11-12): its prediction
        Q(s, a) should match reward + gamma * Q_target(s', a').
        """
        # Prediction side: MAIN critic, WITH gradients (this is what we train).
        q_pred = self.critic(state, action)     # (batch, 1)

        # Target side: TARGET networks, NO gradients. no_grad freezes this into a
        # constant goal so gradients don't leak into the (frozen) target nets.
        with torch.no_grad():
            next_action = self.actor_target(next_state)
            q_next = self.critic_target(next_state, next_action)

        target = reward + self.gamma * q_next

        # Regression: pull the prediction toward the target (mean squared error).
        loss = torch.nn.functional.mse_loss(q_pred, target)

        # Standard update ritual: clear old grads -> backprop -> step the optimizer.
        # (gradients accumulate by default, so zero_grad first is mandatory.)
        self.critic_opt.zero_grad()
        loss.backward()
        self.critic_opt.step()

        return loss.item()


    def update_actor(self, state):
        """
        Push the Actor toward actions the Critic values more (Eqs. 13-14).
        """
        # Action from the MAIN actor WITH gradients, so they flow back through the
        # Critic into the Actor's weights (Critic -> action -> actor params).
        action = self.actor(state)
        q = self.critic(state, action)

        # Actor wants to MAXIMISE Q, but optimizers minimise -> minimise -Q.
        # (The minus sign is the whole trick; drop it and the Actor gets worse.)
        loss = -q.mean()

        # Only step the ACTOR optimizer: grads also land on the Critic here, but
        # actor_opt only holds actor params, so the Critic is left untouched.
        self.actor_opt.zero_grad()
        loss.backward()
        self.actor_opt.step()

        return loss.item()


    def soft_update(self, target, main):
        """Nudge one target network a fraction tau toward its main network."""
        # Manual weight assignment (not a differentiable op), hence .data / no_grad:
        #   target <- tau * main + (1 - tau) * target
        with torch.no_grad():
            for tp, mp in zip(target.parameters(), main.parameters()):
                tp.data = self.tau * mp.data + (1 - self.tau) * tp.data


    def update_targets(self):
        """Soft-update both target networks toward their mains."""
        self.soft_update(self.actor_target, self.actor)
        self.soft_update(self.critic_target, self.critic)


    def select_action(self, state, epsilon):
        """
        Deterministic action + exploration noise, for ACTING (not for updates).
        epsilon scales the noise and is decayed by the training loop over time.
        """
        with torch.no_grad():
            action = self.actor(state)
            noisy = action + epsilon * torch.randn_like(action)     # explore around the policy
            return torch.clamp(noisy, -self.I_max, self.I_max)      # keep within [-I_max, I_max]


if __name__ == "__main__":
    ddpg = DDPG(state_dim=12, action_dim=1, hidden_dim=20, n_layers=4,
                I_max=10, gamma=0.999, tau=0.001, lr=0.001)

    # Targets start identical to mains.
    a = next(ddpg.actor.parameters())
    b = next(ddpg.actor_target.parameters())
    assert torch.equal(a, b)

    state = torch.randn(8, 12)
    action = torch.randn(8, 1)
    reward = torch.randn(8, 1)
    next_state = torch.randn(8, 12)

    # Smoke test: on a fixed batch, each loss should DECREASE over repeated calls.
    print("Critic update test:")
    for _ in range(10):
        print(ddpg.update_critic(state, action, reward, next_state))

    print("Actor update test:")
    state = torch.randn(8, 12)
    for _ in range(10):
        print(ddpg.update_actor(state))

    # After training the main actor, the target has NOT moved -> no longer equal.
    print(torch.equal(a, b))    # should be False