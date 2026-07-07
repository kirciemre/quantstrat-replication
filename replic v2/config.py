"""
config.py
---------
Every numeric hyperparameter used in Section 4 of the paper, collected in
one place with a comment pointing at its source (Table 1, Table 2, Table 3,
Eq. 5, or a footnote). Where the paper's tables are ambiguous or overlap
(see the notes below), the choice made here is documented explicitly so it
can be revisited.

Ambiguities and how they were resolved
=======================================
1. Table 1's "Simulation parameters" sub-table gives one (layers, hidden
   nodes) pair per *scenario* (5/16, 5/20, 6/20 for the theta-only,
   theta+kappa, and theta+kappa+sigma scenarios respectively), while
   Table 2 gives a single *default* GRU size (5 layers, 20 hidden nodes)
   for the prob-DDPG / reg-DDPG first-step network, and Table 3 gives a
   separate default for hid-DDPG's own GRU (1 layer, 10 hidden nodes),
   with an explicit override to 2 layers for the hardest scenario.
   We treat Table 1's per-scenario numbers as authoritative for the
   prob-DDPG/reg-DDPG first-step GRU (since they line up exactly with
   Table 2's default in the middle scenario), and Table 3's numbers
   (with its stated override) as authoritative for hid-DDPG's own GRU.
2. Section 4.3, footnote 6, states that prob-DDPG's look-back window is
   widened to W=20 specifically for the theta+kappa+sigma scenario
   (elsewhere W=10, per Table 2). This override is encoded below.
3. Table 1 states "l = 5" and "script-ell = 1"; Section 3.2.1 / Algorithm 1
   identify `l` as the number of Actor-update repetitions per training
   iteration and `script-ell` as the number of Critic-update repetitions.
   That is what is encoded below (ACTOR_INNER_STEPS=5, CRITIC_INNER_STEPS=1).
4. Eq. (6)-(9) state that "sigma is the hyperbolic tangent (tanh)
   activation function" for *all three* GRU gates (reset, update,
   candidate) - a non-standard GRU (the textbook Cho et al. 2014 GRU uses
   sigmoid for the reset/update gates). We replicate the paper's literal
   statement by default (see nn.GRUCell's `gate_activation="tanh"`).
5. The invariant volatility formula is stated in the paper as
   sigma_inv = sigma / (2*kappa). (The textbook stationary standard
   deviation of an OU process is sigma / sqrt(2*kappa).) We implement the
   paper's literal formula by default and flag the textbook alternative
   via `USE_TEXTBOOK_OU_STD` below.
"""
import numpy as np

# ----------------------------------------------------------------- generic
DT = 0.2                     # Delta t, discretisation step (Table 1)
MU_INV = 1.0                 # invariant mean used to draw the initial signal value (Table 1)
TRAIN_EPISODES = 10_000       # N, number of training iterations (Table 1)
TEST_EPISODES = 500           # M, number of test episodes (Table 1)
TEST_STEPS = 2_000            # n, number of trades per test episode (Table 1)
BATCH_SIZE = 512               # b, training batch size (Table 1)
I_MAX = 10.0                  # maximum inventory (Table 1)
I_MIN = -10.0                 # minimum inventory (Table 1)
LAMBDA_COST = 0.05             # transaction cost per unit volume, lambda (Table 1)
LEARNING_RATE = 0.001          # Weighted ADAM learning rate (Table 1)
GAMMA = 0.999                  # for the synthetic experiments the paper reports an
                                # infinite-horizon discounted problem (Sec. 2.1); the
                                # real-data experiments explicitly use gamma=0.999
                                # (Table 8) - we use the same value throughout for
                                # consistency, since Section 4 does not give a
                                # distinct discount factor.
ACTOR_INNER_STEPS = 5           # l  (Table 1) -- Actor update repetitions per iteration
CRITIC_INNER_STEPS = 1          # script-ell (Table 1) -- Critic update repetitions
EPS_DECAY_A = 100.0             # a, exploration-decay constant (Table 1): eps = max(a/(a+m), eps_min)
EPS_MIN = 0.01                  # a floor for the exploration noise std (not explicit in
                                 # the paper beyond "eps_min < a"; chosen small and safe)
SOFT_UPDATE_TAU = 0.001          # Polyak/soft-update coefficient for the target critic

USE_TEXTBOOK_OU_STD = False       # False => sigma_inv = sigma/(2*kappa) (paper's literal formula)
                                  # True  => sigma_inv = sigma/sqrt(2*kappa) (textbook OU formula)

# ----------------------------------------------------- Actor/Critic sizes (Table 1)
# Same for all three scenarios; only the auxiliary GRU/classifier capacity changes
# with scenario complexity (see SCENARIOS below).
ACTOR_CRITIC_SIZES = {
    "hid-DDPG":  {"n_layers": 4, "hidden_dim": 20},
    "prob-DDPG": {"n_layers": 5, "hidden_dim": 64},
    "reg-DDPG":  {"n_layers": 5, "hidden_dim": 64},
}

# ------------------------------------------------------- transition-rate matrices
# Eq. (5): P(theta_t = phi_j | theta_{t-1} = phi_i) = [expm(A * tau)]_ij
A_THETA = np.array([
    [-0.10, 0.05, 0.05],
    [0.05, -0.10, 0.05],
    [0.05, 0.05, -0.10],
])
A_KAPPA = np.array([
    [-0.10, 0.10],
    [0.10, -0.10],
])
A_SIGMA = np.array([
    [-0.10, 0.10],
    [0.10, -0.10],
])

# ------------------------------------------------------------------ scenarios
# Regime values, and (n_layers, hidden_dim) for the auxiliary GRU/classifier of
# EACH method in that scenario. hid-DDPG's own GRU follows Table 3 (1 layer /
# 10 hidden units, except 2 layers in the hardest scenario); prob-DDPG's and
# reg-DDPG's first-step GRU follows Table 1's "Simulation parameters" numbers.
SCENARIOS = {
    "theta": {
        "label": "θt is a Markov chain",
        "theta_regimes": np.array([0.9, 1.0, 1.1]),
        "kappa": 5.0,                 # constant (Table 1, "Simulation parameters")
        "kappa_regimes": None,
        "sigma": 0.2,                  # constant
        "sigma_regimes": None,
        "gru_layers_two_step": 5, "gru_hidden_two_step": 16,  # Table 1 row for this scenario
        "gru_layers_hid": 1, "gru_hidden_hid": 10,             # Table 3 default
        "lookback_prob": 10, "lookback_reg": 10, "lookback_hid": 10,  # Table 2/3 defaults
    },
    "theta_kappa": {
        "label": "θt, κt are Markov chains",
        "theta_regimes": np.array([0.9, 1.0, 1.1]),
        "kappa": None,
        "kappa_regimes": np.array([3.0, 7.0]),
        "sigma": 0.2,
        "sigma_regimes": None,
        "gru_layers_two_step": 5, "gru_hidden_two_step": 20,
        "gru_layers_hid": 1, "gru_hidden_hid": 10,
        "lookback_prob": 10, "lookback_reg": 10, "lookback_hid": 10,
    },
    "theta_kappa_sigma": {
        "label": "θt, κt, σt are Markov chains",
        "theta_regimes": np.array([0.9, 1.0, 1.1]),
        "kappa": None,
        "kappa_regimes": np.array([3.0, 7.0]),
        "sigma": None,
        "sigma_regimes": np.array([0.1, 0.3]),
        "gru_layers_two_step": 6, "gru_hidden_two_step": 20,
        "gru_layers_hid": 2, "gru_hidden_hid": 10,           # Table 3 override for this scenario
        "lookback_prob": 20,        # Sec. 4.3 footnote 6: widened window for prob-DDPG here
        "lookback_reg": 50,          # Table 2: "W=50 reg-DDPG"
        "lookback_hid": 10,
    },
}

# reg-DDPG's look-back window differs from prob-DDPG's already in the easy
# scenarios too (Table 2: "W = 10 prob-DDPG (W = 50 reg-DDPG)"), so fix it
# consistently across all three scenarios except where a scenario-specific
# override is given above:
for _name, _sc in SCENARIOS.items():
    _sc.setdefault("lookback_reg", 50)

PREDICT_WINDOW = 1  # GRU forecasts / classifies using a single next step (Table 2)

# classifier / regressor head sizes (Sec. 3.3.1 / 3.3.2): "5 layers, 64 hidden nodes"
HEAD_N_LAYERS = 5
HEAD_HIDDEN_DIM = 64
