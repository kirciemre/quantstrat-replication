"""
TD3 (Twin Delayed DDPG) training algorithm.

This is an experimental variant of ddpg.py, aimed at diagnosing the
scenario2 reg-DDPG instability (actor loss climbing unboundedly, ~7x
inflated test reward, ~19x inflated std vs the paper). Classic DDPG uses a
SINGLE critic, which is known to systematically overestimate Q-values; the
overestimated values then mislead the actor into an aggressive, poorly-
generalizing policy -- exactly the failure mode observed in scenario2's
training logs (actor loss 0.09 -> 0.68, never converging).

TD3 (Fujimoto, Hoff & Meger 2018) fixes this with three changes relative to
DDPG, all implemented here:

  1. Twin critics: train Critic1 and Critic2 independently; use the SMALLER
     of the two target Q-values as the Bellman target. This directly
     suppresses overestimation (a single critic can only overestimate; the
     min of two independently-erring critics is a much more conservative,
     less biased estimate).

  2. Target policy smoothing: add small clipped noise to the target action
     before evaluating it, so the critic can't exploit a narrow, sharp peak
     in Q-value at one specific action (which is easy to overestimate and
     hard to generalize).

  3. Delayed policy updates: update the actor (and soft-update all targets)
     only once every `policy_delay` critic updates, giving the critics time
     to settle before the actor chases them.

Actor and Critic architectures are UNCHANGED (reused from src.models.actor /
src.models.critic) -- TD3 only changes how they are trained, not what they are.
"""

import torch
import torch.nn as nn

from src.models.actor import Actor
from src.models.critic import Critic


class TD3:
    def __init__(self, state_dim, action_dim, d_NN, l_NN, I_max, gamma, tau, lr,
                 policy_delay=2, policy_noise=0.2, noise_clip=0.5):
        self.gamma = gamma
        self.I_max = I_max
        self.tau = tau
        self.lr = lr
        self.policy_delay = policy_delay
        self.policy_noise = policy_noise   # std of smoothing noise (paper's noise scale, not I_max units)
        self.noise_clip = noise_clip       # clip range for smoothing noise

        self.actor = Actor(state_dim, d_NN, l_NN, I_max)
        self.actor_target = Actor(state_dim, d_NN, l_NN, I_max)
        self.actor_target.load_state_dict(self.actor.state_dict())

        # --- twin critics: two independent Q-estimators + their targets ---
        self.critic1 = Critic(state_dim, action_dim, d_NN, l_NN)
        self.critic2 = Critic(state_dim, action_dim, d_NN, l_NN)
        self.critic1_target = Critic(state_dim, action_dim, d_NN, l_NN)
        self.critic2_target = Critic(state_dim, action_dim, d_NN, l_NN)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.actor_opt = torch.optim.AdamW(self.actor.parameters(), lr=lr, weight_decay=1e-5)
        self.critic1_opt = torch.optim.AdamW(self.critic1.parameters(), lr=lr, weight_decay=1e-5)
        self.critic2_opt = torch.optim.AdamW(self.critic2.parameters(), lr=lr, weight_decay=1e-5)

        self._update_count = 0   # tracks calls to update_critic, drives delayed actor updates


    def update_critic(self, state, action, reward, next_state):
        """
        Fit BOTH critics to the same, conservative Bellman target:
            y = r + gamma * min(Q1_tgt(s', a'_smoothed), Q2_tgt(s', a'_smoothed))
        where a'_smoothed = clip(pi_tgt(s') + clipped_noise, -I_max, I_max).

        Returns (critic1_loss, critic2_loss) so the training loop can log both.
        """
        with torch.no_grad():
            # --- target policy smoothing: noisy, clipped target action ---
            next_action = self.actor_target(next_state)
            noise = torch.randn_like(next_action) * self.policy_noise
            noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
            next_action = torch.clamp(next_action + noise, -self.I_max, self.I_max)

            # --- twin target Q, take the MIN to suppress overestimation ---
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

        self._update_count += 1

        return loss1.item(), loss2.item()


    def update_actor(self, state):
        """
        Push the actor toward higher Q1-values (TD3 uses only critic1 for the
        actor's gradient, following the original paper). Only actually steps
        the actor every `policy_delay` calls -- earlier calls are no-ops that
        return None, so the caller's logging can skip them.
        """
        if self._update_count % self.policy_delay != 0:
            return None   # delayed: skip this round

        action = self.actor(state)
        q1 = self.critic1(state, action)
        loss = -q1.mean()

        self.actor_opt.zero_grad()
        loss.backward()
        self.actor_opt.step()

        # --- targets only move when the actor moves (paper's delayed update) ---
        self.soft_update(self.actor_target, self.actor)
        self.soft_update(self.critic1_target, self.critic1)
        self.soft_update(self.critic2_target, self.critic2)

        return loss.item()


    def soft_update(self, target, main):
        """Nudge one target network a fraction tau toward its main network."""
        with torch.no_grad():
            for tp, mp in zip(target.parameters(), main.parameters()):
                tp.data = self.tau * mp.data + (1 - self.tau) * tp.data


    def select_action(self, state, epsilon):
        """Deterministic action + exploration noise, for ACTING (not updates)."""
        with torch.no_grad():
            action = self.actor(state)
            noisy = action + epsilon * torch.randn_like(action)
            return torch.clamp(noisy, -self.I_max, self.I_max)
