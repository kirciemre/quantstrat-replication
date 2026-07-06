import argparse
import numpy as np
import torch
import os
from scipy.linalg import expm

from ou_env import OUEnvironment
from gru_filters import GRUClassifier, GRURegressor
from train import train_prob_filter, train_reg_filter, train_ddpg_agent
from evaluate import evaluate_agent, plot_rewards_distribution

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

def run_synthetic_replication(full_training=False, steps=None, pretrain_steps=None, test_episodes=None, test_steps=None):
    # Determine training steps
    N_train = 10000 if full_training else 200
    N_pretrain = 10000 if full_training else 200
    M_test = 500 if full_training else 100
    n_test = 2000 if full_training else 500
    
    # Overrides
    if steps is not None:
        N_train = steps
    if pretrain_steps is not None:
        N_pretrain = pretrain_steps
    if test_episodes is not None:
        M_test = test_episodes
    if test_steps is not None:
        n_test = test_steps

    print("=" * 60)
    print("RUNNING SYNTHETIC REPLICATION")
    print(f"Parameters: N_train={N_train}, N_pretrain={N_pretrain}, M_test={M_test}, n_test={n_test}")
    print("=" * 60)

    results = {}
    
    for scenario in [1, 2, 3]:
        scenario_name = f"Scenario {scenario}"
        if scenario == 1:
            desc = "theta only is MC"
            gru_layers = 5
            gru_hidden = 16
        elif scenario == 2:
            desc = "theta and kappa are MCs"
            gru_layers = 5
            gru_hidden = 20
        else:
            desc = "theta, kappa, sigma are MCs"
            gru_layers = 6
            gru_hidden = 20
            
        print(f"\n--- Running {scenario_name} ({desc}) ---")
        env = OUEnvironment(scenario=scenario, dt=0.2)
        
        # 1. hid-DDPG (One-step approach)
        # Note: hid-DDPG GRU params from Table 3: layers=1 (2 for Scenario 3), hidden=10, lookback W=10.
        hid_W = 10
        hid_actor, hid_gru = train_ddpg_agent(
            env, model_type="hid-DDPG", W=hid_W, N=N_train, batch_size=512,
            lr=0.001, lambda_cost=0.05, gamma=0.99, tau=0.001,
            l_act=5, ell_crit=1, a_noise=100.0, eps_min=0.01,
            layers_ddpg=4, hidden_ddpg=20
        )
        
        # 2. prob-DDPG (Two-step posterior regime approach)
        # Pre-train GRU Classifier
        prob_W = 20 if scenario == 3 else 10 # W=20 for scenario 3 prob-DDPG, W=10 for 1 and 2
        classifier = GRUClassifier(
            input_dim=1, hidden_dim=gru_hidden, num_layers=gru_layers, num_classes=3,
            ffn_layers=5, ffn_hidden=64
        )
        classifier = train_prob_filter(env, classifier, epochs=N_pretrain, batch_size=512, W=prob_W, lr=0.001)
        # Train DDPG agent
        prob_actor = train_ddpg_agent(
            env, model_type="prob-DDPG", W=prob_W, N=N_train, batch_size=512,
            lr=0.001, lambda_cost=0.05, gamma=0.99, tau=0.001,
            l_act=5, ell_crit=1, a_noise=100.0, eps_min=0.01,
            gru_model=classifier, layers_ddpg=5, hidden_ddpg=64
        )
        
        # 3. reg-DDPG (Two-step next-signal prediction approach)
        reg_W = 50 # Look-back window W = 50 for reg-DDPG
        regressor = GRURegressor(
            input_dim=1, hidden_dim=gru_hidden, num_layers=gru_layers,
            ffn_layers=5, ffn_hidden=64
        )
        regressor = train_reg_filter(env, regressor, epochs=N_pretrain, batch_size=512, W=reg_W, lr=0.001)
        # Train DDPG agent
        reg_actor = train_ddpg_agent(
            env, model_type="reg-DDPG", W=reg_W, N=N_train, batch_size=512,
            lr=0.001, lambda_cost=0.05, gamma=0.99, tau=0.001,
            l_act=5, ell_crit=1, a_noise=100.0, eps_min=0.01,
            gru_model=regressor, layers_ddpg=5, hidden_ddpg=64
        )
        
        # --- EVALUATION ---
        print(f"\nEvaluating agents for {scenario_name}...")
        
        rewards_dict = {}
        
        # hid-DDPG
        rewards_hid, mean_hid, std_hid = evaluate_agent(
            env, hid_actor, model_type="hid-DDPG", gru_model=hid_gru,
            M=M_test, n=n_test, W=hid_W, lambda_cost=0.05
        )
        rewards_dict["hid-DDPG"] = rewards_hid
        
        # prob-DDPG
        rewards_prob, mean_prob, std_prob = evaluate_agent(
            env, prob_actor, model_type="prob-DDPG", gru_model=classifier,
            M=M_test, n=n_test, W=prob_W, lambda_cost=0.05
        )
        rewards_dict["prob-DDPG"] = rewards_prob
        
        # reg-DDPG
        rewards_reg, mean_reg, std_reg = evaluate_agent(
            env, reg_actor, model_type="reg-DDPG", gru_model=regressor,
            M=M_test, n=n_test, W=reg_W, lambda_cost=0.05
        )
        rewards_dict["reg-DDPG"] = rewards_reg
        
        # Store results
        results[scenario_name] = {
            "hid-DDPG": (mean_hid, std_hid),
            "prob-DDPG": (mean_prob, std_prob),
            "reg-DDPG": (mean_reg, std_reg)
        }
        
        # Plot and save histogram
        plot_rewards_distribution(rewards_dict, scenario_name)

    # --- PRINT SUMMARY TABLE (Table 4) ---
    print("\n" + "=" * 60)
    print("TABLE 4 REPLICATION: AVERAGE CUMULATIVE REWARDS")
    print("=" * 60)
    print(f"{'Model':<12} | {'Scenario 1 (theta)':<20} | {'Scenario 2 (theta,kappa)':<24} | {'Scenario 3 (theta,kappa,sigma)':<28}")
    print("-" * 95)
    for model_name in ["hid-DDPG", "reg-DDPG", "prob-DDPG"]:
        row = f"{model_name:<12}"
        for sc in ["Scenario 1", "Scenario 2", "Scenario 3"]:
            mean, std = results[sc][model_name]
            row += f" | {mean:5.2f} ± {std:4.2f}"
        print(row)
    print("=" * 95)


def run_pair_trading_demo(full_training=False, steps=None, pretrain_steps=None, test_episodes=None, test_steps=None):
    N_train = 5000 if full_training else 200
    N_pretrain = 5000 if full_training else 200
    M_test = 500 if full_training else 50
    n_test = 2000 if full_training else 500

    if steps is not None:
        N_train = steps
    if pretrain_steps is not None:
        N_pretrain = pretrain_steps
    if test_episodes is not None:
        M_test = test_episodes
    if test_steps is not None:
        n_test = test_steps

    print("\n" + "=" * 60)
    print("RUNNING SECTION 5: COINTEGRATED PAIR TRADING DEMO")
    print("=" * 60)

    # 1. Simulating Cointegrated SMH & INTC mid-prices
    # We will generate synthetic mid-price data matching the properties of the pair
    total_steps = n_test + 100 + 2000  # historical + test length
    
    # SMH price is simulated as random walk
    S_smh = np.zeros(total_steps)
    S_smh[0] = 290.0
    for t in range(1, total_steps):
        S_smh[t] = S_smh[t-1] + np.random.normal(0, 0.4)
        
    # Cointegrated spread S_t = 2.856 * S_smh - 0.804 * S_intc
    # S_t will switch between two regimes: 0.2216 and 0.5658 (after min-max normalized to [0,1])
    # Let's say unnormalized spread switches between 15.0 and 35.0
    regimes = np.array([15.0, 35.0])
    P = expm(np.array([[-0.1, 0.1], [0.1, -0.1]]) * 0.2)
    
    spread = np.zeros(total_steps)
    spread[0] = 25.0
    current_regime = 0
    
    for t in range(1, total_steps):
        cum_prob = P[current_regime]
        current_regime = 0 if np.random.rand() < cum_prob[0] else 1
        theta = regimes[current_regime]
        kappa = 5.0
        sigma = 1.5
        
        prev = spread[t-1]
        exp_term = np.exp(-kappa * 0.2)
        mean_t = prev * exp_term + theta * (1.0 - exp_term)
        std_t = sigma * np.sqrt((1.0 - np.exp(-2.0 * kappa * 0.2)) / (2.0 * kappa))
        spread[t] = mean_t + std_t * np.random.normal()
        
    # Re-calculate INTC price to satisfy the cointegrating vector
    S_intc = (2.856 * S_smh - spread) / 0.804

    # Save dummy pair dataset
    os.makedirs("data", exist_ok=True)
    np.savez("data/pair_trading_prices.npz", smh=S_smh, intc=S_intc, spread=spread)
    print("Saved simulated cointegrated price series to 'data/pair_trading_prices.npz'")

    # Set up custom environment based on the simulated cointegrated spread
    # For training, we can reuse our OU Environment configured with 2 regimes
    class CointegratedSpreadEnv:
        def __init__(self, dt=0.2):
            self.scenario = 2  # use 2 regimes
            self.dt = dt
            self.P_theta = expm(np.array([[-0.1, 0.1], [0.1, -0.1]]) * self.dt)
            self.const_kappa = 5.0
            self.const_sigma = 1.5
            self.mu_inv = 25.0
            self.sigma_inv = self.const_sigma / np.sqrt(2 * self.const_kappa)
            self.init_std = 3.0 * self.sigma_inv
            self.regimes_theta = regimes
            
        def step_markov_chain(self, current_states, P):
            batch_size = current_states.shape[0]
            cum_probs = P[current_states]
            r = np.random.rand(batch_size)
            next_states = (r[:, None] > cum_probs.cumsum(axis=-1)).sum(axis=-1)
            return np.clip(next_states, 0, P.shape[1] - 1)
            
        def generate_paths(self, batch_size, seq_len):
            # Same path generation but using the 2-regime specifications
            theta_indices = np.random.choice(self.P_theta.shape[0], size=batch_size)
            S = np.zeros((batch_size, seq_len))
            theta_val_arr = np.zeros((batch_size, seq_len))
            theta_idx_arr = np.zeros((batch_size, seq_len), dtype=int)
            kappa_val_arr = np.ones((batch_size, seq_len)) * self.const_kappa
            sigma_val_arr = np.ones((batch_size, seq_len)) * self.const_sigma
            
            S[:, 0] = np.random.normal(self.mu_inv, self.init_std, size=batch_size)
            theta_val_arr[:, 0] = self.regimes_theta[theta_indices]
            theta_idx_arr[:, 0] = theta_indices
            
            for t in range(1, seq_len):
                theta_indices = self.step_markov_chain(theta_indices, self.P_theta)
                theta = self.regimes_theta[theta_indices]
                theta_val_arr[:, t] = theta
                theta_idx_arr[:, t] = theta_indices
                
                prev_S = S[:, t-1]
                exp_term = np.exp(-self.const_kappa * self.dt)
                mean_t = prev_S * exp_term + theta * (1.0 - exp_term)
                std_t = self.const_sigma * np.sqrt((1.0 - np.exp(-2.0 * self.const_kappa * self.dt)) / (2.0 * self.const_kappa))
                S[:, t] = mean_t + std_t * np.random.normal(size=batch_size)
                
            return S, theta_val_arr, theta_idx_arr, kappa_val_arr, sigma_val_arr

    pair_env = CointegratedSpreadEnv()

    # Normalize functions for the custom spread
    # Min/max bounds estimated from spread data
    spread_min, spread_max = 5.0, 45.0
    
    def norm_spread(S):
        return np.clip((S - spread_min) / (spread_max - spread_min), 0.0, 1.0)

    # 2. Train prob-DDPG on custom environment (with 2 classes/regimes)
    classifier_2 = GRUClassifier(
        input_dim=1, hidden_dim=20, num_layers=5, num_classes=2,
        ffn_layers=5, ffn_hidden=64
    )
    # We need to temporarily patch normalize_signal in train.py or handle it:
    # Since our OUEnvironment has custom min-max, let's create a wrapper
    import ou_env
    original_norm = ou_env.normalize_signal
    original_norm_tensor = ou_env.normalize_signal_tensor
    ou_env.normalize_signal = norm_spread
    ou_env.normalize_signal_tensor = lambda S: torch.clamp((S - spread_min) / (spread_max - spread_min), 0.0, 1.0)

    try:
        classifier_2 = train_prob_filter(pair_env, classifier_2, epochs=N_pretrain, batch_size=512, W=10, lr=0.001)
        prob_actor = train_ddpg_agent(
            pair_env, model_type="prob-DDPG", W=10, N=N_train, batch_size=512,
            lr=0.001, lambda_cost=0.05, gamma=0.99, tau=0.001,
            l_act=5, ell_crit=1, a_noise=100.0, eps_min=0.01,
            gru_model=classifier_2, layers_ddpg=5, hidden_ddpg=64
        )
        
        # 3. Train hid-DDPG online
        hid_actor, hid_gru = train_ddpg_agent(
            pair_env, model_type="hid-DDPG", W=10, N=N_train, batch_size=512,
            lr=0.001, lambda_cost=0.05, gamma=0.99, tau=0.001,
            l_act=5, ell_crit=1, a_noise=100.0, eps_min=0.01,
            layers_ddpg=4, hidden_ddpg=20
        )
    finally:
        # Restore normalizer
        ou_env.normalize_signal = original_norm
        ou_env.normalize_signal_tensor = original_norm_tensor

    # 4. Rolling Z-Score Benchmark Implementation
    # We will compute the rolling Z-Score over the test window
    def evaluate_zscore_strategy(spread_series, M, n, W, lambda_cost=0.05):
        # spread_series: shape (M, W + n + 1)
        I = np.zeros(M)
        cum_rewards = np.zeros(M)
        
        for t in range(n):
            k = W + t
            # Compute rolling mean and std over past W+1 steps (from k-W to k)
            window = spread_series[:, k-W : k+1]
            mean = np.mean(window, axis=1)
            std = np.std(window, axis=1)
            std[std < 1e-5] = 1e-5  # avoid zero division
            
            z = (spread_series[:, k] - mean) / std
            
            # Action: Accumulate units proportional to negative Z-score
            # Scaling factor of 3.0 maps Z-score of -3.0 to positive long position of 9.0
            I_next = np.clip(-3.0 * z, -10.0, 10.0)
            
            # Step reward
            r_t = I_next * (spread_series[:, k+1] - spread_series[:, k]) - lambda_cost * np.abs(I_next - I)
            cum_rewards += r_t
            I = I_next
            
        return cum_rewards, np.mean(cum_rewards), np.std(cum_rewards)

    # 5. Evaluate all on a fresh out-of-sample set of test paths
    print("\nEvaluating all strategies on Out-Of-Sample Pair Trading paths...")
    test_spreads = pair_env.generate_paths(M_test, 10 + n_test + 1)[0]
    
    # Evaluate Z-Score
    rewards_z, mean_z, std_z = evaluate_zscore_strategy(test_spreads, M_test, n_test, 10, lambda_cost=0.05)
    
    # Patch normalizers again for evaluation
    ou_env.normalize_signal = norm_spread
    ou_env.normalize_signal_tensor = lambda S: torch.clamp((S - spread_min) / (spread_max - spread_min), 0.0, 1.0)
    
    try:
        # Evaluate prob-DDPG
        _, mean_p, std_p = evaluate_agent(
            pair_env, prob_actor, model_type="prob-DDPG", gru_model=classifier_2,
            M=M_test, n=n_test, W=10, lambda_cost=0.05
        )
        
        # Evaluate hid-DDPG
        _, mean_h, std_h = evaluate_agent(
            pair_env, hid_actor, model_type="hid-DDPG", gru_model=hid_gru,
            M=M_test, n=n_test, W=10, lambda_cost=0.05
        )
    finally:
        # Restore normalizer
        ou_env.normalize_signal = original_norm
        ou_env.normalize_signal_tensor = original_norm_tensor

    # --- PRINT SUMMARY TABLE (Table 9) ---
    print("\n" + "=" * 60)
    print("TABLE 9 REPLICATION: OUT-OF-SAMPLE PAIR TRADING REWARDS")
    print("=" * 60)
    print(f"{'Strategy':<15} | {'Average Reward':<16} | {'Std. Dev.':<16}")
    print("-" * 55)
    print(f"{'prob-DDPG':<15} | {mean_p:16.4f} | {std_p:16.4f}")
    print(f"{'hid-DDPG':<15} | {mean_h:16.4f} | {std_h:16.4f}")
    print(f"{'Rolling Z-Score':<15} | {mean_z:16.4f} | {std_z:16.4f}")
    print("=" * 55)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replicate DDPG optimal trading framework.")
    parser.add_argument("--full", action="store_true", help="Run full paper training steps (10k iterations)")
    parser.add_argument("--steps", type=int, default=None, help="Override DDPG training steps")
    parser.add_argument("--pretrain_steps", type=int, default=None, help="Override classifier/regressor pre-training epochs")
    parser.add_argument("--test_episodes", type=int, default=None, help="Override evaluation episodes")
    parser.add_argument("--test_steps", type=int, default=None, help="Override evaluation steps")
    args = parser.parse_args()

    # Run replication of synthetic experiments
    run_synthetic_replication(
        full_training=args.full,
        steps=args.steps,
        pretrain_steps=args.pretrain_steps,
        test_episodes=args.test_episodes,
        test_steps=args.test_steps
    )
    
    # Run replication of Section 5 pair trading application
    run_pair_trading_demo(
        full_training=args.full,
        steps=args.steps,
        pretrain_steps=args.pretrain_steps,
        test_episodes=args.test_episodes,
        test_steps=args.test_steps
    )
