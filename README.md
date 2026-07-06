# quantstrat-replication

This is our replication project for Deep reinforcement learning for optimal trading with partial information.

You can start full test of paper with this command.

```bash
python main.py --full
```

It will start tarining first and then run synthetic repication part.

It may take a bit long depends on your GPU and it will designed to use accelerator mps for Apple Silicon or CUDA for Nvidia GPU.

You can validate code, tables and models with this code;

```bash
python main.py --steps 10 --pretrain_steps 10 --test_episodes 5 --test_steps 10
```

It should be faster than full testing code.


# Deep Reinforcement Learning for Optimal Trading with Partial Information - Replication Codebase

This repository is a PyTorch-based replication of the DDPG-GRU trading framework proposed in the paper **"Deep reinforcement learning for optimal trading with partial information" (2025)** by Andrea Macrì, Sebastian Jaimungal, and Fabrizio Lillo. 

The codebase implements three Deep Deterministic Policy Gradient (DDPG) variants integrated with Gated Recurrent Unit (GRU) networks:
1. **`hid-DDPG`**: A one-step approach directly encoding temporal hidden states from the GRU into the RL trader.
2. **`prob-DDPG`**: A two-step method using posterior regime probability estimates of the mean-reversion levels.
3. **`reg-DDPG`**: A two-step method relying on forecasts of the next signal value.

---

## 🛠️ Requirements

To run this project, make sure Python 3.8+ and the following libraries are installed:
```bash
pip install numpy torch scipy matplotlib
```
*Note: PyTorch automatically selects the `mps` device for hardware acceleration on Apple Silicon (M1/M2/M3 Macs) or `cuda` on Nvidia GPUs for optimal performance.*

---

## 🚀 Usage & Testing

We provide dynamic parameters via CLI arguments to make debugging and full execution simple.

### 1. Fast Verification Run
To verify that all environments, filters, tensor shapes, and evaluation modules compile and run successfully (completes in **10-15 seconds**), run:
```bash
python main.py --steps 10 --pretrain_steps 10 --test_episodes 5 --test_steps 10
```

### 2. Full Paper Replication (10k Iterations)
To train all three variants matching the original paper specifications (10,000 steps, 500 test episodes, and 2,000 steps) and output Table 4 and Table 9:
```bash
python main.py --full
```

### 3. Custom Run
You can customize the number of training, pre-training, and testing steps manually:
```bash
python main.py --steps 1000 --pretrain_steps 1000 --test_episodes 100 --test_steps 500
```

---

## 📂 Codebase Structure

* **`ou_env.py`**: Simulates continuous-time parameter transitions via Markov Chains discretized using the matrix exponential ($P = e^{A \Delta t}$) and contains the exact solution step of the Ornstein-Uhlenbeck process.
* **`gru_filters.py`**: Defines PyTorch GRU modules for representation learning:
  * `GRUEncoder` (trained online with an auxiliary next-step regression head for `hid-DDPG`)
  * `GRUClassifier` (trained offline with cross-entropy loss for `prob-DDPG`)
  * `GRURegressor` (trained offline with MSE loss for `reg-DDPG`)
* **`ddpg.py`**: Implements Actor-Critic architectures, action clamping (inventories restricted to $[-10, 10]$), and soft target updates ($\tau_{\text{tgt}} = 0.001$).
* **`train.py`**: Implements offline filter pre-training loops and Algorithm 1 replay-free batch DDPG training.
* **`evaluate.py`**: Handles out-of-sample path evaluation starting from $S_0 = 1.0$ and $I_0 = 0.0$, running test episodes in parallel. Plots cumulative reward distributions.
* **`main.py`**: Coordinates Scenarios 1, 2, and 3, runs Section 5's cointegrated pair-trading demonstration (SMH/INTC spread), and prints Table 4 and Table 9.

---

## 📊 Outputs

Upon completion, the script:
1. Prints **Table 4 (Synthetic OU Results)** and **Table 9 (Pair Trading Results)** directly to the terminal.
2. Saves comparative reward histograms to the `plots/` directory (e.g., `rewards_scenario_1.png`).
3. Saves the simulated cointegrated spread data to `data/pair_trading_prices.npz`.
