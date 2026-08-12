"""
TD3 v2 -- fixes a confound in the first experiment (src/models/td3.py).

In v1, the training loop called update_actor() once per iteration, and TD3
internally skipped all but every policy_delay-th call. Combined with DDPG's
usual pattern of `l=5` actor updates per iteration, this meant the actor was
trained roughly 10x LESS than in the DDPG baseline (0.5 updates/iter vs 5
updates/iter) -- confounding "twin critic fixes overestimation" with "actor
undertrained", so the bad v1 results could not be attributed to either cause
cleanly.

v2 isolates the ONE variable we actually want to test (twin critics + target
policy smoothing) by keeping the actor's update frequency identical to the
DDPG baseline: cfg.l actor gradient steps happen every time the actor is due
(the caller decides when "due" is, typically every iteration, same as DDPG).

Changes from td3.py (v1):
  - update_actor() no longer skips internally -- it ALWAYS performs a step
    when called. The training script now controls cadence explicitly by
    only calling it inside an `if m % policy_delay == 0:` block, and inside
    that block calling it `cfg.l` times (matching DDPG's inner loop).
  - Everything else (twin critics, min-Q target, target policy smoothing)
    is unchanged from v1.
"""

import torch

from src.models.actor import Actor
from src.models.critic import Critic


class TD3:
    def __init__(self, state_dim, action_dim, d_NN, l_NN, I_max, gamma, tau, lr,
                 policy_noise=0.2, noise_clip=0.5):
        self.gamma = gamma
        self.I_max = I_max
        self.tau = tau
        self.lr = lr
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip

        self.actor = Actor(state_dim, d_NN, l_NN, I_max)
        self.actor_target = Actor(state_dim, d_NN, l_NN, I_max)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic1 = Critic(state_dim, action_dim, d_NN, l_NN)
        self.critic2 = Critic(state_dim, action_dim, d_NN, l_NN)
        self.critic1_target = Critic(state_dim, action_dim, d_NN, l_NN)
        self.critic2_target = Critic(state_dim, action_dim, d_NN, l_NN)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.actor_opt = torch.optim.AdamW(self.actor.parameters(), lr=lr, weight_decay=1e-5)
        self.critic1_opt = torch.optim.AdamW(self.critic1.parameters(), lr=lr, weight_decay=1e-5)
        self.critic2_opt = torch.optim.AdamW(self.critic2.parameters(), lr=lr, weight_decay=1e-5)


    def update_critic(self, state, action, reward, next_state):
        """Fit both critics to min(Q1_tgt, Q2_tgt) with smoothed target action."""
        with torch.no_grad():
            next_action = self.actor_target(next_state)
            noise = torch.clamp(
                torch.randn_like(next_action) * self.policy_noise,
                -self.noise_clip, self.noise_clip,
            )
            next_action = torch.clamp(next_action + noise, -self.I_max, self.I_max)

            q1_next = self.critic1_target(next_state, next_action)
            q2_next = self.critic2_target(next_state, next_action)
            q_next = torch.min(q1_next, q2_next)

            target = reward + self.gamma * q_next

        q1_pred = self.critic1(state, action)
        q2_pred = self.critic2(state, action)

        loss1 = torch.nn.functional.mse_loss(q1_pred, target)
        loss2 = torch.nn.functional.mse_loss(q2_pred, target)

        self.critic1_opt.zero_grad()
        loss1.backward()
        self.critic1_opt.step()

        self.critic2_opt.zero_grad()
        loss2.backward()
        self.critic2_opt.step()

        return loss1.item(), loss2.item()


    def update_actor(self, state):
        """
        ALWAYS performs one actor gradient step when called (no internal
        skipping -- v2's fix). The caller controls how often this runs.
        """
        action = self.actor(state)
        q1 = self.critic1(state, action)
        loss = -q1.mean()

        self.actor_opt.zero_grad()
        loss.backward()
        self.actor_opt.step()

        return loss.item()


    def soft_update(self, target, main):
        with torch.no_grad():
            for tp, mp in zip(target.parameters(), main.parameters()):
                tp.data = self.tau * mp.data + (1 - self.tau) * tp.data


    def select_action(self, state, epsilon):
        with torch.no_grad():
            action = self.actor(state)
            noisy = action + epsilon * torch.randn_like(action)
            return torch.clamp(noisy, -self.I_max, self.I_max)
