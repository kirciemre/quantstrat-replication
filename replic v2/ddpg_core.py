"""
ddpg_core.py
------------
Shared building blocks for all three DDPG variants: feature normalisation
("features ... normalised in the domain [0, 1]") and the generic training
iteration of Algorithm 1 (critic inner loop, actor inner loop, soft target
update, exploration-noise decay), parametrised by an `Agent` object that
knows how to build its own state features (see agents.py).

Design note on the TD target
=============================
Eq. (12): y_t = r_t + gamma * Q_tgt(G'_{t+1}, pi(G'_{t+1}|mu_pi) | mu_Qtgt)
uses the *current* actor pi (not a target actor) together with the
*target* critic Q_tgt. We follow this literally. As in standard DDPG, y_t
is always treated as a fixed regression target for the critic's MSE loss -
no gradient is propagated back through y_t (this is also what "target" /
"y" conventionally means in the DDPG literature the paper is drawing on).

Design note on hid-DDPG's shared GRU
=====================================
Algorithm 1 lists a call "Pass {...} to GRU to train if (hid-DDPG)" inside
*both* the Critic-update and the Actor-update procedures (lines 14 and 29),
as opposed to "obtain {Phi_k}" / "obtain S~_{t+1}" for the two-step methods
(read-only use of an already-frozen network). We read this literally: for
hid-DDPG the GRU is trained jointly by three losses every iteration - its
own auxiliary next-step forecast loss (Sec. 3.2.1, "before updating the
Critic and Actor... we first optimise [the GRU's] parameters"), the critic
loss, and the actor loss - each via its own Adam optimiser instance (see
agents.HidDDPGAgent). For prob-DDPG / reg-DDPG the first-step network is
pretrained once, then frozen (no gradient at all) during DDPG training,
matching "obtain" in the algorithm listing.
"""
from __future__ import annotations
import numpy as np
from autodiff import Tensor
import config as C


class FeatureNormalizer:
    """Rescales inventory and signal level into [0, 1] for the DDPG features."""

    def __init__(self, i_min, i_max, s_low, s_high):
        self.i_min, self.i_max = float(i_min), float(i_max)
        self.s_low, self.s_high = float(s_low), float(s_high)

    def norm_I(self, I):
        return (I - self.i_min) / (self.i_max - self.i_min)

    def norm_S(self, S):
        return np.clip((S - self.s_low) / (self.s_high - self.s_low), 0.0, 1.0)

    @staticmethod
    def hidden_tanh_to_unit(h):
        """Maps a tanh-bounded value in (-1, 1) to (0, 1)."""
        return (h + 1.0) * 0.5


def feature_range_for_scenario(scenario_cfg, n_std=6.0, use_textbook_std=True):
    """
    A documented heuristic for the [s_low, s_high] normalisation range: the
    widest plausible excursion of the signal, using the (textbook) OU
    stationary standard deviation evaluated at the *slowest* mean-reversion
    speed and *highest* volatility available in the scenario (the
    combination that produces the widest invariant spread), added on top
    of the outer theta regimes. This is a modelling choice made explicit
    here since the paper does not give normalisation bounds.
    """
    theta_regimes = scenario_cfg["theta_regimes"]
    kappa_min = scenario_cfg["kappa"] if scenario_cfg["kappa_regimes"] is None else scenario_cfg["kappa_regimes"].min()
    sigma_max = scenario_cfg["sigma"] if scenario_cfg["sigma_regimes"] is None else scenario_cfg["sigma_regimes"].max()
    if use_textbook_std:
        sigma_inv = sigma_max / np.sqrt(2.0 * kappa_min)
    else:
        sigma_inv = sigma_max / (2.0 * kappa_min)
    s_low = theta_regimes.min() - n_std * sigma_inv
    s_high = theta_regimes.max() + n_std * sigma_inv
    return float(s_low), float(s_high)


def exploration_sigma(m, a=C.EPS_DECAY_A, eps_min=C.EPS_MIN):
    """eps = max(a/(a+m), eps_min): the exploration-noise std schedule (Table 1, footnote 5)."""
    return max(a / (a + m), eps_min)


def train_iteration(agent, batch_gen, m, rng):
    """
    Runs ONE full training iteration `m` (1-indexed) of Algorithm 1 for the
    given `agent` (see agents.py for the required interface):
      agent.hid_gru_update(batch)         [only meaningful for hid-DDPG; no-op otherwise]
      agent.build_state(S_window, I, grad)-> Tensor (batch, state_dim)
      agent.actor, agent.critic, agent.critic_target : FeedForward
      agent.opt_actor, agent.opt_critic
      agent.critic_extra_params / agent.actor_extra_params (e.g. GRU params for hid-DDPG)
    `batch_gen` is an environment.TrainingBatchGenerator for the scenario.
    """
    eps = exploration_sigma(m)

    # ---- (optional) auxiliary GRU update, hid-DDPG only ----------------
    agent.hid_gru_update(batch_gen)

    # ---- Critic update(s) ----------------------------------------------
    for _ in range(C.CRITIC_INNER_STEPS):
        batch = batch_gen.sample()
        S, I_t = batch["S"], batch["I_t"]
        W1 = S.shape[1] - 1  # W+1 points per window
        S_win_t = S[:, 0:W1]        # {S_u}_{t-W}^{t}
        S_win_tp1 = S[:, 1:W1 + 1]    # {S_u}_{t-W+1}^{t+1}
        S_t, S_tp1 = S[:, W1 - 1], S[:, W1]

        G_t = agent.build_state(S_win_t, I_t, grad=True)
        action = agent.actor(G_t)
        noise = rng.normal(0.0, eps, size=action.shape)
        I_tp1 = np.clip(action.data + noise, C.I_MIN, C.I_MAX)

        r = np.clip(I_tp1[:, 0], C.I_MIN, C.I_MAX)  # placeholder overwritten below
        from environment import reward as reward_fn
        r = reward_fn(I_tp1[:, 0], S_t, S_tp1, I_t, lam=C.LAMBDA_COST)

        G_tp1 = agent.build_state(S_win_tp1, I_tp1[:, 0], grad=False)
        a_tp1 = agent.actor(G_tp1)  # current (non-target) actor, per Eq. 12
        q_tgt = agent.critic_target(Tensor.cat([G_tp1, Tensor(a_tp1.data)], axis=1))
        y = r[:, None] + C.GAMMA * q_tgt.data
        y_t = Tensor(y, requires_grad=False)

        q = agent.critic(Tensor.cat([G_t, Tensor(I_tp1)], axis=1))
        critic_loss = ((q - y_t) ** 2).mean()

        agent.opt_critic.zero_grad()
        critic_loss.backward()
        agent.opt_critic.step()

    # ---- Actor update(s) -------------------------------------------------
    last_actor_loss = None
    for _ in range(C.ACTOR_INNER_STEPS):
        batch = batch_gen.sample()
        S, I_t = batch["S"], batch["I_t"]
        W1 = S.shape[1] - 1
        S_win_t = S[:, 0:W1]

        G_t = agent.build_state(S_win_t, I_t, grad=True)
        action = agent.actor(G_t)
        q = agent.critic(Tensor.cat([G_t, action], axis=1))
        actor_loss = -q.mean()

        agent.opt_actor.zero_grad()
        actor_loss.backward()
        agent.opt_actor.step()
        last_actor_loss = actor_loss.data.item()

    # ---- soft update of the target critic --------------------------------
    from nn import soft_update
    soft_update(agent.critic_target.parameters(), agent.critic.parameters(), C.SOFT_UPDATE_TAU)

    return dict(eps=eps, actor_loss=last_actor_loss)
