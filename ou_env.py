import numpy as np
from scipy.linalg import expm
import torch

# Transition rate matrices (continuous time)
A_theta = np.array([
    [-0.1, 0.05, 0.05],
    [0.05, -0.1, 0.05],
    [0.05, 0.05, -0.1]
])

A_kappa = np.array([
    [-0.1, 0.1],
    [0.1, -0.1]
])

A_sigma = np.array([
    [-0.1, 0.1],
    [0.1, -0.1]
])

# Regimes
regimes_theta = np.array([0.9, 1.0, 1.1])
regimes_kappa = np.array([3.0, 7.0])
regimes_sigma = np.array([0.1, 0.3])

class OUEnvironment:
    def __init__(self, scenario=1, dt=0.2):
        """
        Scenario 1: theta is MC, kappa=5, sigma=0.2
        Scenario 2: theta, kappa are MCs, sigma=0.2
        Scenario 3: theta, kappa, sigma are MCs
        """
        self.scenario = scenario
        self.dt = dt
        
        # Discretize continuous transition rate matrices to step probabilities
        self.P_theta = expm(A_theta * self.dt)
        self.P_kappa = expm(A_kappa * self.dt)
        self.P_sigma = expm(A_sigma * self.dt)
        
        # Define regimes as attributes
        self.regimes_theta = regimes_theta
        self.regimes_kappa = regimes_kappa
        self.regimes_sigma = regimes_sigma
        
        # Constant parameters (if not switching)
        self.const_kappa = 5.0
        self.const_sigma = 0.2
        
        # Calculate invariant parameters based on minimal parameters
        min_kappa = 5.0 if scenario == 1 else 3.0
        min_sigma = 0.2 if scenario in (1, 2) else 0.1
        self.mu_inv = 1.0
        self.sigma_inv = min_sigma / np.sqrt(2 * min_kappa)
        self.init_std = 3.0 * self.sigma_inv

    def step_markov_chain(self, current_states, P):
        """
        Vectorized transition of Markov chains for batch of paths.
        current_states: shape (batch_size,) integer indices
        P: transition matrix
        """
        batch_size = current_states.shape[0]
        cum_probs = P[current_states] # shape (batch_size, num_states)
        r = np.random.rand(batch_size)
        next_states = (r[:, None] > cum_probs.cumsum(axis=-1)).sum(axis=-1)
        return np.clip(next_states, 0, P.shape[1] - 1)

    def generate_paths(self, batch_size, seq_len):
        """
        Simulate batch of OU and regime paths.
        seq_len: total sequence length (e.g. W + 2 for training, or n + W + 1 for testing)
        
        Returns:
            S: Signal, (batch_size, seq_len)
            theta_vals: theta parameters, (batch_size, seq_len)
            theta_indices: theta regime indices, (batch_size, seq_len)
            kappa_vals: kappa parameters, (batch_size, seq_len)
            sigma_vals: sigma parameters, (batch_size, seq_len)
        """
        # Initial states based on transition matrix shapes
        theta_indices = np.random.choice(self.P_theta.shape[0], size=batch_size)
        if self.scenario >= 2:
            kappa_indices = np.random.choice(self.P_kappa.shape[0], size=batch_size)
        else:
            kappa_indices = np.zeros(batch_size, dtype=int)
            
        if self.scenario == 3:
            sigma_indices = np.random.choice(self.P_sigma.shape[0], size=batch_size)
        else:
            sigma_indices = np.zeros(batch_size, dtype=int)
            
        S = np.zeros((batch_size, seq_len))
        theta_val_arr = np.zeros((batch_size, seq_len))
        theta_idx_arr = np.zeros((batch_size, seq_len), dtype=int)
        kappa_val_arr = np.zeros((batch_size, seq_len))
        sigma_val_arr = np.zeros((batch_size, seq_len))
        
        # Initialize starting values S_{t-W} ~ N(mu_inv, 3*sigma_inv)
        S[:, 0] = np.random.normal(self.mu_inv, self.init_std, size=batch_size)
        
        theta_val_arr[:, 0] = self.regimes_theta[theta_indices]
        theta_idx_arr[:, 0] = theta_indices
        kappa_val_arr[:, 0] = self.regimes_kappa[kappa_indices] if self.scenario >= 2 else self.const_kappa
        sigma_val_arr[:, 0] = self.regimes_sigma[sigma_indices] if self.scenario == 3 else self.const_sigma
        
        for t in range(1, seq_len):
            # Transition parameters
            theta_indices = self.step_markov_chain(theta_indices, self.P_theta)
            if self.scenario >= 2:
                kappa_indices = self.step_markov_chain(kappa_indices, self.P_kappa)
            if self.scenario == 3:
                sigma_indices = self.step_markov_chain(sigma_indices, self.P_sigma)
                
            theta = self.regimes_theta[theta_indices]
            kappa = self.regimes_kappa[kappa_indices] if self.scenario >= 2 else self.const_kappa
            sigma = self.regimes_sigma[sigma_indices] if self.scenario == 3 else self.const_sigma
            
            theta_val_arr[:, t] = theta
            theta_idx_arr[:, t] = theta_indices
            kappa_val_arr[:, t] = kappa
            sigma_val_arr[:, t] = sigma
            
            # OU dynamics exact solution
            prev_S = S[:, t-1]
            exp_term = np.exp(-kappa * self.dt)
            mean_t = prev_S * exp_term + theta * (1 - exp_term)
            std_t = sigma * np.sqrt((1 - np.exp(-2.0 * kappa * self.dt)) / (2.0 * kappa))
            S[:, t] = mean_t + std_t * np.random.normal(size=batch_size)
            
        return S, theta_val_arr, theta_idx_arr, kappa_val_arr, sigma_val_arr

def normalize_signal(S, S_min=0.5, S_max=1.5):
    """Normalize numpy array S to [0, 1]"""
    return np.clip((S - S_min) / (S_max - S_min), 0.0, 1.0)

def normalize_inventory(I, I_min=-10.0, I_max=10.0):
    """Normalize numpy array I to [0, 1]"""
    return np.clip((I - I_min) / (I_max - I_min), 0.0, 1.0)

def normalize_signal_tensor(S, S_min=0.5, S_max=1.5):
    """Normalize torch tensor S to [0, 1]"""
    return torch.clamp((S - S_min) / (S_max - S_min), 0.0, 1.0)

def normalize_inventory_tensor(I, I_min=-10.0, I_max=10.0):
    """Normalize torch tensor I to [0, 1]"""
    return torch.clamp((I - I_min) / (I_max - I_min), 0.0, 1.0)
