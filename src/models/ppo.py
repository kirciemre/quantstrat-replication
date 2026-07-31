"""
PPO agent for the trading problem (professor-suggested alternative to DDPG).

Why PPO here: DDPG's seed-to-seed divergence traces to Q-value overestimation
(the critic ends up most confident about the worst policy). No DDPG-family fix
(twin critics, LR schedule, convex penalties, 5x iterations) reduced it. PPO is
structurally different -- on-policy, a stochastic actor, and a STATE-value critic
V(s) rather than Q(s,a) -- so it sidesteps that specific pathology, and its
clipped update is designed for low-variance, seed-robust training.

Structural differences from the DDPG code:
  - Actor outputs a Gaussian (mean = tanh*I_max network, plus a learned log-std);
    actions are SAMPLED and we track log-prob.
  - Critic is V(s) (state only), not Q(s, a).
  - On-policy: collect a rollout, compute GAE advantages, do K epochs of the
    clipped surrogate, then discard the rollout. No target nets, no soft update.

This module provides the actor, critic, and PPO update. Rollout collection (an
episodic trajectory with inventory carried forward) lives in the train script,
since it needs the encoder and environment.
"""

import numpy as np
import torch
import torch.nn as nn


class GaussianActor(nn.Module):
    """
    Policy pi(a | s) = Normal(mu(s), sigma). mu is a tanh*I_max network (same
    shape as the DDPG actor); sigma is a state-independent learned parameter
    (standard for continuous PPO). Actions are the inventory I_{t+1}.
    """
    def __init__(self, state_dim, d_NN, l_NN, I_max, init_log_std=-0.5):
        super().__init__()
        self.I_max = I_max
        layers = [nn.Linear(state_dim, d_NN), nn.SiLU()]
        for _ in range(l_NN):
            layers += [nn.Linear(d_NN, d_NN), nn.SiLU()]
        layers.append(nn.Linear(d_NN, 1))
        self.mu_net = nn.Sequential(*layers)
        # one learned log-std for the action dimension
        self.log_std = nn.Parameter(torch.full((1,), init_log_std))

    def _dist(self, state):
        mu = torch.tanh(self.mu_net(state)) * self.I_max
        std = self.log_std.exp().expand_as(mu)
        return torch.distributions.Normal(mu, std)

    def forward(self, state):
        """Return the distribution (for sampling / log-prob / entropy)."""
        return self._dist(state)

    def act(self, state):
        """Sample an action and its log-prob (rollout collection)."""
        dist = self._dist(state)
        a = dist.sample()
        a = a.clamp(-self.I_max, self.I_max)
        logp = dist.log_prob(a).sum(-1, keepdim=True)
        return a, logp

    def mean_action(self, state):
        """Deterministic action = the mean (for EVALUATION, no sampling)."""
        return torch.tanh(self.mu_net(state)) * self.I_max


class VCritic(nn.Module):
    """State-value function V(s). Same MLP shape, input is state only."""
    def __init__(self, state_dim, d_NN, l_NN):
        super().__init__()
        layers = [nn.Linear(state_dim, d_NN), nn.SiLU()]
        for _ in range(l_NN):
            layers += [nn.Linear(d_NN, d_NN), nn.SiLU()]
        layers.append(nn.Linear(d_NN, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, state):
        return self.net(state)


def compute_gae(rewards, values, next_value, gamma, lam):
    """
    Generalized Advantage Estimation.

    rewards    : (T,) tensor of step rewards
    values     : (T,) tensor V(s_t)
    next_value : scalar V(s_T) (bootstrap at the end of the rollout)
    Returns (advantages (T,), returns (T,)). Advantages are NOT yet normalized.
    """
    T = rewards.shape[0]
    adv = torch.zeros(T)
    gae = 0.0
    for t in reversed(range(T)):
        v_next = next_value if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * v_next - values[t]
        gae = delta + gamma * lam * gae
        adv[t] = gae
    returns = adv + values
    return adv, returns


class PPO:
    def __init__(self, state_dim, action_dim, d_NN, l_NN, I_max, gamma, lr,
                 gae_lambda=0.95, clip=0.2, epochs=10, minibatch=256,
                 ent_coef=0.0, vf_coef=0.5, N=None):
        self.gamma = gamma
        self.I_max = I_max
        self.gae_lambda = gae_lambda
        self.clip = clip
        self.epochs = epochs
        self.minibatch = minibatch
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef

        self.actor = GaussianActor(state_dim, d_NN, l_NN, I_max)
        self.critic = VCritic(state_dim, d_NN, l_NN)
        self.opt = torch.optim.AdamW(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=lr, weight_decay=1e-5)
        self.sched = (torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=N)
                      if N is not None else None)

    def update(self, states, actions, old_logp, advantages, returns):
        """
        One PPO update on a collected rollout: K epochs of the clipped surrogate
        over minibatches. All inputs are tensors of shape (T, ...).
        """
        # rollout tensors are FIXED targets -- detach so the K epochs don't try
        # to backprop through the graph that produced them (old policy/critic).
        states = states.detach()
        actions = actions.detach()
        old_logp = old_logp.detach()
        advantages = advantages.detach()
        returns = returns.detach()

        # normalize advantages (standard PPO stabilizer)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        T = states.shape[0]
        idx = np.arange(T)
        last = {}
        for _ in range(self.epochs):
            np.random.shuffle(idx)
            for start in range(0, T, self.minibatch):
                mb = idx[start:start + self.minibatch]
                s = states[mb]; a = actions[mb]
                olp = old_logp[mb]; adv = advantages[mb]; ret = returns[mb]

                dist = self.actor(s)
                logp = dist.log_prob(a).sum(-1, keepdim=True)
                ratio = (logp - olp).exp()

                # clipped surrogate (policy loss)
                unclipped = ratio * adv
                clipped = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv
                policy_loss = -torch.min(unclipped, clipped).mean()

                # value loss
                value = self.critic(s)
                value_loss = torch.nn.functional.mse_loss(value, ret)

                # entropy bonus (exploration; ent_coef=0 disables)
                entropy = dist.entropy().mean()

                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()), 0.5)
                self.opt.step()
                last = {"policy": policy_loss.item(), "value": value_loss.item(),
                        "entropy": entropy.item()}
        return last

    def step_scheduler(self):
        if self.sched is not None:
            self.sched.step()

    def current_lr(self):
        return self.opt.param_groups[0]["lr"]

    def state_dicts(self):
        return {"actor": self.actor.state_dict(), "critic": self.critic.state_dict()}


if __name__ == "__main__":
    # Smoke test: GAE shapes, one update runs, deterministic eval action bounded.
    torch.manual_seed(0)
    ppo = PPO(state_dim=3, action_dim=1, d_NN=20, l_NN=4, I_max=10,
              gamma=0.99, lr=3e-4, N=1000)

    T = 500
    states = torch.randn(T, 3)
    actions, logps = [], []
    for t in range(T):
        a, lp = ppo.actor.act(states[t:t+1])
        actions.append(a); logps.append(lp)
    actions = torch.cat(actions); old_logp = torch.cat(logps)

    with torch.no_grad():
        values = ppo.critic(states).squeeze(-1)
        next_value = ppo.critic(torch.randn(1, 3)).item()
    rewards = torch.randn(T)
    adv, returns = compute_gae(rewards, values, next_value, 0.99, 0.95)
    print(f"GAE: adv {tuple(adv.shape)}, returns {tuple(returns.shape)}")

    out = ppo.update(states, actions, old_logp, adv.unsqueeze(-1), returns.unsqueeze(-1))
    print(f"update ok: policy={out['policy']:.4f} value={out['value']:.4f} entropy={out['entropy']:.4f}")

    a_eval = ppo.actor.mean_action(torch.randn(5, 3) * 100)
    print(f"deterministic eval action bounded: max|a|={a_eval.abs().max().item():.2f} (<= 10)")