# Scalable Counterfactual Explanations for Time Series via Policy Learning
This repository contains the complete implementation for generating counterfactual explanations on time series data using policy learning. The code accompanies our submission to KDD 2026 Cycle 2.
## 📂 Core Implementation Files

### Main Scripts
- **`mainRL_time0.py`**  
  Main training script implementing the CFPL framework.

- **`utils.py`**  
  Utility functions for:
  - data loading
  - preprocessing
  - evaluation metrics
  - visualization

### Modified Alibi Library (`alibi/`)
Customized version of the Alibi library with CFPL support, including:

- Custom explainer implementations
- Time-series encoder and decoder models
- Reinforcement learning components

---

## ⚙️ Prerequisites

- Python 3.7+
- TensorFlow 2.4+
- CUDA-capable GPU (recommended)
- 16GB+ RAM
- UCR Time Series Archive 2018


---

## 🔧 Setup

Install dependencies and login to Weights & Biases:

```bash
pip install -r requirements.txt
wandb login
```
---

## 🚀 Execution Scripts

Run batch experiments with experience replay:

```bash
bash rl_run_with_replay_all.sh
```

Run batch experiments without replay (ablation study):

```bash
bash rl_run_ablation_all.sh
```

Run a single dataset experiment using the template script:

```bash
bash rl_run.sh
```

