# Deep Reinforcement Learning for Optimal Trading with Partial Information

This repository is a PyTorch-based replication codebase for the optimal trading framework proposed in the paper **"Deep reinforcement learning for optimal trading with partial information" (2025)**. 

The codebase implements three Gated Recurrent Unit (GRU) integrated reinforcement learning variants (supporting **PPO, DDPG, and TD3**):
1. **`hid-DDPG / hid-PPO`**: A one-step approach directly encoding temporal hidden states from the GRU into the RL trader.
2. **`prob-DDPG / prob-PPO`**: A two-step method using posterior regime probability estimates of the mean-reversion levels.
3. **`reg-DDPG / reg-PPO`**: A two-step method relying on forecasts of the next signal value.

---

## 🛠️ Requirements

To run this project, make sure Python 3.8+ and the following libraries are installed:
```bash
pip install -r requirements.txt
```
Or manually:
```bash
pip install numpy torch scipy yaml matplotlib
```

---

## 📂 Codebase Structure

The pipeline is organized in the modular **`src/`** directory:
* **`src/env/`**: Contains the trading environment (`trading_env.py`) and step reward formulas (`reward.py`).
* **`src/data/`**: Ornstein-Uhlenbeck price signal paths simulation engine (`ou_simulator.py`).
* **`src/models/`**: Neural network architectures:
  * `gru.py`: GRU hidden-state encoder.
  * `gru_classifier.py`: Regime classification network.
  * `regressor.py`: Price signal next-step prediction network.
  * `ppo.py`: PPO actor-critic algorithm implementation.
  * `ddpg.py` / `agents.py`: DDPG and TD3 agent structures.
* **`src/train/`**: Training scripts for pre-training filters and main policy optimization.
* **`src/eval/`**: Evaluation and path metrics rollouts.
* **`configs/`**: YAML configuration files defining hyperparameter sweeps for all scenarios.

---

## 🚀 Execution & Usage

All training and evaluation scripts are run as python modules from the root workspace directory.

### 1. Training

#### Scenario 1/2/3 hid-DDPG / hid-PPO / hid-TD3:
```bash
python3 -m src.train.train_hid --scenario [1|2|3] --model [ppo|ddpg|td3] [--seed SEED]
```

#### Scenario 1/2/3 prob-DDPG / prob-PPO / prob-TD3:
1. **Pre-train the Regime Classifier:**
   ```bash
   python3 -m src.train.train_classifier --scenario [1|2|3] --theta-only [--seed SEED]
   ```
2. **Train the RL Agent:**
   ```bash
   python3 -m src.train.train_prob --scenario [1|2|3] --model [ppo|ddpg|td3] --theta-only [--seed SEED]
   ```

#### Scenario 1/2/3 reg-DDPG / reg-PPO / reg-TD3:
1. **Pre-train the Price Regressor:**
   ```bash
   python3 -m src.train.train_regressor --scenario [1|2|3] [--seed SEED]
   ```
2. **Train the RL Agent:**
   ```bash
   python3 -m src.train.train_reg --scenario [1|2|3] --model [ppo|ddpg|td3] [--seed SEED]
   ```

---

### 2. Evaluation

To evaluate a trained agent out-of-sample and check metrics (P&L mean, std, inventory bound hits):

* **hid Models:**
  ```bash
  python3 -m src.eval.eval_hid --scenario [1|2|3] --model [ppo|ddpg|td3] [--seed EVAL_SEED]
  ```
* **prob Models:**
  ```bash
  python3 -m src.eval.eval_prob --scenario [1|2|3] --model [ppo|ddpg|td3] --theta-only [--seed EVAL_SEED]
  ```
* **reg Models:**
  ```bash
  python3 -m src.eval.eval_reg --scenario [1|2|3] --model [ppo|ddpg|td3] [--seed EVAL_SEED]
  ```

---

### 3. Master Replications (Capstone Script)

To execute the master benchmark replication over multiple training seeds (default 5 seeds per cell) and print the comparative performance grid (Table 4 style):

```bash
chmod +x scripts/capstone.sh
./scripts/capstone.sh [num_seeds]
```
Example running 3 seeds (quicker):
```bash
./scripts/capstone.sh 3
```

---

## 📊 Outputs

- Trained models and checkpoints are saved under the `artifacts/` folder (e.g. `artifacts/scenario1_ppo_hid.pt`).
- Reward comparison charts are saved in the `figures/` directory.
