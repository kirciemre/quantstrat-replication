"""
DDPG agents with a selectable critic scheme.

Two variants share one base class; they differ ONLY in the critic:

  "ddpg" : single critic Q. Standard DDPG (the paper's algorithm).
  "td3"  : twin critics Q1, Q2; the Bellman target uses min(Q1, Q2) to cap the
           value-overestimation that drives DDPG's seed-to-seed divergence
           (Fujimoto et al. 2018).

Select with build_agent(model_name, ...). A --model CLI flag in the training
script maps straight onto that name, so you can A/B the two without editing code.

Everything outside the critic (actor, target nets, soft update, exploration
noise, LR schedule, pre-built states) is identical across variants and lives in
the base class.
"""

import torch

from src.models.actor import Actor
from src.models.critic import Critic
from src.models.ppo import PPO as _PPOImpl


# ---------------------------------------------------------------------------
# shared base: actor + targets + soft update + action selection + schedule
# ---------------------------------------------------------------------------
class _BaseDDPG:
    def __init__(self, state_dim, action_dim, d_NN, l_NN, I_max, gamma, tau, lr, N=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.d_NN = d_NN
        self.l_NN = l_NN
        self.gamma = gamma
        self.I_max = I_max
        self.tau = tau
        self.lr = lr

        self.actor = Actor(state_dim, d_NN, l_NN, I_max)
        self.actor_target = Actor(state_dim, d_NN, l_NN, I_max)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_opt = torch.optim.AdamW(self.actor.parameters(), lr=lr, weight_decay=1e-5)

        # subclasses build critics + self.critic_opt, then call _finish_init(N)
        self._build_critics()
        if N is not None:
            self.critic_sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.critic_opt, T_max=N)
            self.actor_sched  = torch.optim.lr_scheduler.CosineAnnealingLR(self.actor_opt,  T_max=N)
        else:
            self.critic_sched = self.actor_sched = None

    # --- subclass hooks ---
    def _build_critics(self):
        raise NotImplementedError

    def update_critic(self, state, action, reward, next_state):
        raise NotImplementedError

    def update_targets(self):
        raise NotImplementedError

    def _actor_critic(self, state):
        """The critic the actor maximises (critic1 for td3, the sole critic for ddpg)."""
        raise NotImplementedError

    # --- shared across variants ---
    def update_actor(self, state):
        action = self.actor(state)
        q = self._actor_critic(state, action)
        loss = -q.mean()
        self.actor_opt.zero_grad()
        loss.backward()
        self.actor_opt.step()
        return loss.item()

    def step_schedulers(self):
        if self.critic_sched is not None:
            self.critic_sched.step()
            self.actor_sched.step()

    def current_lr(self):
        return self.actor_opt.param_groups[0]["lr"]

    def soft_update(self, target, main):
        with torch.no_grad():
            for tp, mp in zip(target.parameters(), main.parameters()):
                tp.data = self.tau * mp.data + (1 - self.tau) * tp.data

    def act_eval(self, state):
        """Deterministic action for EVALUATION (no exploration noise)."""
        return self.actor(state)

    def select_action(self, state, epsilon):
        with torch.no_grad():
            action = self.actor(state)
            noisy = action + epsilon * torch.randn_like(action)
            return torch.clamp(noisy, -self.I_max, self.I_max)

    def state_dicts(self):
        """Checkpoint payload -- variant-specific critics + actor + targets."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# single-critic DDPG (the paper's algorithm)
# ---------------------------------------------------------------------------
class DDPG(_BaseDDPG):
    def _build_critics(self):
        self.critic = Critic(self.state_dim, self.action_dim, self.d_NN, self.l_NN)
        self.critic_target = Critic(self.state_dim, self.action_dim, self.d_NN, self.l_NN)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_opt = torch.optim.AdamW(self.critic.parameters(), lr=self.lr, weight_decay=1e-5)

    def _actor_critic(self, state, action):
        return self.critic(state, action)

    def update_critic(self, state, action, reward, next_state):
        with torch.no_grad():
            next_action = self.actor_target(next_state)
            q_next = self.critic_target(next_state, next_action)
            target = reward + self.gamma * q_next
        q_pred = self.critic(state, action)
        loss = torch.nn.functional.mse_loss(q_pred, target)
        self.critic_opt.zero_grad()
        loss.backward()
        self.critic_opt.step()
        return loss.item()

    def update_targets(self):
        self.soft_update(self.actor_target, self.actor)
        self.soft_update(self.critic_target, self.critic)

    def state_dicts(self):
        return {"actor": self.actor.state_dict(), "critic": self.critic.state_dict()}


# ---------------------------------------------------------------------------
# twin-critic TD3 (min(Q1, Q2) clipped target)
# ---------------------------------------------------------------------------
class DDPG_TD3(_BaseDDPG):
    def _build_critics(self):
        self.critic1 = Critic(self.state_dim, self.action_dim, self.d_NN, self.l_NN)
        self.critic2 = Critic(self.state_dim, self.action_dim, self.d_NN, self.l_NN)
        self.critic1_target = Critic(self.state_dim, self.action_dim, self.d_NN, self.l_NN)
        self.critic2_target = Critic(self.state_dim, self.action_dim, self.d_NN, self.l_NN)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.critic_opt = torch.optim.AdamW(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=self.lr, weight_decay=1e-5)

    def _actor_critic(self, state, action):
        return self.critic1(state, action)          # actor maximises Q1 only

    def update_critic(self, state, action, reward, next_state):
        with torch.no_grad():
            next_action = self.actor_target(next_state)
            q1n = self.critic1_target(next_state, next_action)
            q2n = self.critic2_target(next_state, next_action)
            target = reward + self.gamma * torch.min(q1n, q2n)     # clipped target
        q1 = self.critic1(state, action)
        q2 = self.critic2(state, action)
        loss = (torch.nn.functional.mse_loss(q1, target)
                + torch.nn.functional.mse_loss(q2, target))
        self.critic_opt.zero_grad()
        loss.backward()
        self.critic_opt.step()
        return loss.item()

    def update_targets(self):
        self.soft_update(self.actor_target, self.actor)
        self.soft_update(self.critic1_target, self.critic1)
        self.soft_update(self.critic2_target, self.critic2)

    def state_dicts(self):
        return {"actor": self.actor.state_dict(),
                "critic1": self.critic1.state_dict(),
                "critic2": self.critic2.state_dict()}


# ---------------------------------------------------------------------------
# factory: pick the variant by name (maps onto a --model CLI flag)
# ---------------------------------------------------------------------------
# PPO wraps the on-policy implementation but exposes act_eval for uniform eval.
class PPOAgent(_PPOImpl):
    """PPO with the same act_eval(state) interface used by eval scripts."""
    def act_eval(self, state):
        return self.actor.mean_action(state)   # deterministic mean, no sampling


_AGENTS = {"ddpg": DDPG, "td3": DDPG_TD3, "ppo": PPOAgent}


def build_agent(model_name, state_dim, action_dim, d_NN, l_NN, I_max, gamma, tau, lr, N=None):
    """
    model_name : "ddpg" (single critic) or "td3" (twin critics).
    Returns an agent exposing the same interface either way:
      update_critic, update_actor, update_targets, soft_update,
      step_schedulers, current_lr, select_action, state_dicts.
    """
    key = model_name.lower()
    if key not in _AGENTS:
        raise ValueError(f"unknown model '{model_name}'. choose from {list(_AGENTS)}")
    if key == "ppo":
        # PPO has no tau (no target nets); lr is typically lower. Callers pass the
        # PPO lr as `lr`. Extra PPO knobs use their defaults unless overridden.
        return _AGENTS[key](state_dim, action_dim, d_NN, l_NN, I_max, gamma, lr, N=N)
    return _AGENTS[key](state_dim, action_dim, d_NN, l_NN, I_max, gamma, tau, lr, N)


if __name__ == "__main__":
    for name in ("ddpg", "td3"):
        agent = build_agent(name, 3, 1, 20, 4, 10, 0.99, 0.001, 0.001, N=1000)
        s = torch.randn(8, 3); a = torch.randn(8, 1)
        r = torch.randn(8, 1); ns = torch.randn(8, 3)
        cl = agent.update_critic(s, a, r, ns)
        al = agent.update_actor(s)
        agent.update_targets(); agent.step_schedulers()
        keys = list(agent.state_dicts().keys())
        print(f"{name:>5}: critic_loss={cl:.4f}  actor_loss={al:.4f}  ckpt_keys={keys}")