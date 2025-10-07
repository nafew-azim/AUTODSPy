# AUTODSPy

# DSPy Pipeline Optimization with Reinforcement Learning

This repository contains three different reinforcement learning approaches for training a policy model to generate optimal DSPy pipelines for question-answering tasks.

## Overview

All three methods train a language model (GPT-2) to learn which DSPy module and signature combinations work best for different types of questions. The model learns to construct pipelines like `["CoT", "question -> answer"]` that are then executed using DSPy.

<p align="center">
  <img src="assets/Figure_1 (F)-1.png" alt="AutoDSPy Framework" width="600"/>
</p>


## Files

1. **`ppo_training.py`** - Proximal Policy Optimization (PPO)
2. **`grpo_training.py`** - Group Relative Policy Optimization (GRPO)
3. **`reinforce_training.py`** - REINFORCE with Baseline

---

## Common Components

All three implementations share:

### Datasets
- **GSM8K**: Math word problems
- **HotpotQA**: Multi-hop question answering
- Combined and shuffled for training

### Action Space
- **Modules**: `["CoT", "Predict"]` 
- **Signatures**: 15 predefined input-output signatures like `"question -> answer"`, `"problem -> solution"`
- **Pipeline Structure**: Module → Signature → Stop

### Model Architecture
- Base model: GPT-2 (or GPT-2 tokenizer with custom model)
- Device: CUDA if available, else CPU
- Tokenizer: GPT-2 with padding token set to EOS token

You can replace the above model with your own model of choice.

### Core Functions
- `generate_pipeline()`: Generates action sequences using the policy model
- `execute_pipeline()`: Executes the generated pipeline using DSPy modules
- `compute_reward()`: Evaluates response quality against ground truth
- `test_model()`: Tests the trained model on new prompts
- `evaluate_module()`: Evaluates on test datasets

### Reward Computation
All use a two-stage reward system:
1. **Regex matching**: Extract answers from brackets `[answer]` or numeric values
2. **LLM judge**: If regex fails, use LLM to score (0.0 to 1.0)

---

## Key Differences

### 1. PPO (Proximal Policy Optimization)

**File**: `ppo_training.py`

**Unique Features**:
- **Value Head**: Separate neural network to estimate state values
- **GAE (Generalized Advantage Estimation)**: Computes advantages using temporal differences
- **Clipped Objective**: Limits policy updates using clip ratio (default 0.2)
- **Batch Training**: Trains on multiple trajectories per episode
- **Multiple Epochs**: Updates policy multiple times per batch

**Training Parameters**:
```python
train_ppo(num_episodes=20, clip_eps=0.2, epochs_per_batch=10, batch_size=10)
```

**Training Process**:
1. Collect batch of trajectories (10 by default)
2. Compute advantages using GAE with γ=0.99, λ=0.95
3. Update policy and value head for multiple epochs
4. Use clipped surrogate objective to prevent large updates

---

### 2. GRPO (Group Relative Policy Optimization)

**File**: `grpo_training.py`

**Unique Features**:
- **Group-based advantages**: Generates K pipelines per prompt (default K=4)
- **Relative rewards**: Advantage = individual reward - group average
- **No value network**: Simpler architecture
- **Single update per episode**: More efficient than PPO

**Training Parameters**:
```python
train_grpo(num_episodes=200, learning_rate=2e-5, K=4)
```

**Training Process**:
1. Generate K different pipelines for same prompt
2. Compute group average reward
3. Calculate advantage: `reward - group_average`
4. Single policy gradient update

---

### 3. REINFORCE with Baseline

**File**: `reinforce_training.py`

**Unique Features**:
- **Exponential Moving Average Baseline**: Simple running average of rewards
- **Fixed Pipeline Comparison**: Also executes a hardcoded pipeline for comparison
- **Chain Response**: Uses predefined chain: `question -> reasoning -> answer`
- **Simplest Implementation**: No value network, no grouping

**Training Parameters**:
```python
train_rl(num_episodes=100, learning_rate=2e-5)
```

**Training Process**:
1. Generate single pipeline
2. Execute both generated pipeline AND fixed pipeline
3. Get max reward between the two
4. Update baseline: `baseline = 0.9 * baseline + 0.1 * reward`
5. Compute advantage: `reward - baseline`
6. Policy gradient update

---
Full Process Overview

<p align="center">
  <img src="assets/Figure_2 (F)-1.png" alt="AutoDSPy Framework" width="600"/>
</p>

## Running the Code

### Prerequisites

```bash
pip install torch transformers dspy-ai datasets sentence-transformers numpy ollama
```

### Start Ollama Server

All scripts require Ollama 
```bash
ollama serve
ollama pull llama3.1:8b
```
You can replace llama3.1:8b with your own llm model of choice


### Testing

All methods save trained models:
- PPO: `ppo_policy_model.pt` & `ppo_value_head.pt`
- GRPO: `gpt2_trained_policy_model_grpo.pt`
- REINFORCE: `trained_policy_model.pt`

Load and test:
```python
policy_model.load_state_dict(torch.load("model_path.pt"))
response = test_model(policy_model, "What is 5+3?")
```
For REINFORCE and GRPO you have to simply pass the policy_model as the first parameter into either test_model or evaluate_module function but for PPO, you need to pass the value_head as the second parameter into either of the functions.

### Evaluation

Run evaluation on test sets:
```python
# GSM8K
gtotal_time, gtotal_correct = evaluate_module(
    gsm8k_test_data, gsm8k_test, policy_model
)

# HotpotQA (with context)
htotal_time, htotal_correct = evaluate_module(
    hotpotqa_test_data, hotpotqa_test, policy_model, query=True
)
```

---

## Step-by-Step Guide

### Step 1: Environment Setup

1. **Install Dependencies**
```bash
pip install torch transformers dspy-ai datasets sentence-transformers numpy ollama
```

2. **Start Ollama Server**
```bash
# When running locally
# In a separate terminal
ollama serve

# Pull the model (if not already installed)
ollama pull llama3.1:8b

# When running in cloud such as google colab or kaggle
!curl -fsSL https://ollama.com/install.sh | sh

# In a separate cells run
import subprocess
process = subprocess.Popen("ollama serve", shell=True)

!ollama pull llama3.1:8b
```

### Step 2: Choose Your Training Method

Open the corresponding notebook file and follow the cell-by-cell execution:

### Step 3: Monitor Training

**What to expect during training:**

- **PPO**: Shows loss components (policy loss, value loss, entropy) per update
- **GRPO**: Shows average reward and loss per episode
- **REINFORCE**: Shows pipeline, reward, loss, and baseline per episode

**Training progress indicators:**
- Rewards should generally increase over time
- Loss values will fluctuate but should stabilize
- Pipeline diversity should decrease as model learns optimal patterns

### Step 4: Evaluate Results

After training completes, evaluate on test sets:

```python
# Load test datasets (already in notebooks)
gsm8k_test_data = [...]
hotpotqa_test_data = [...]

# Run evaluation
time, correct = evaluate_module(gsm8k_test_data, gsm8k_test, trained_model)
print(f"Correct: {correct}/{len(gsm8k_test_data)}, Time: {time:.2f}s")
```

### Step 5: Save and Load Models

**Saving** (automatic during training):
```python
torch.save(policy_model.state_dict(), "model_name.pt")
```

**Loading** (for continued training or inference):
```python
policy_model.load_state_dict(torch.load("model_name.pt"))
policy_model.eval()  # Set to evaluation mode
```

---

## Troubleshooting

### Common Issues

**1. Ollama Connection Error**
```
Error: Cannot connect to Ollama
```
**Solution**: Ensure Ollama is running (`ollama serve`) and model is installed (`ollama pull llama3.1:8b`)

**2. CUDA Out of Memory**
```
RuntimeError: CUDA out of memory
```
**Solution**: Reduce `batch_size` (PPO) or use CPU by modifying device setup:
```python
device = torch.device("cpu")
```

**3. Import Errors**
```
ModuleNotFoundError: No module named 'dspy'
```
**Solution**: Install missing packages:
```bash
pip install dspy-ai
```

**4. Model Loading Error**
```
RuntimeError: Error(s) in loading state_dict
```
**Solution**: Start with fresh model or ensure compatible architecture

**5. Low/Zero Rewards**
- Check if LLM judge is working correctly
- Verify ground truth formatting
- Increase training episodes
- Adjust learning rate (try 1e-5 to 5e-5)

---

## Tips for Best Results

1. **Start Small**: Use small datasets (10-100 examples) for initial testing
2. **Monitor Early**: Watch first 10 episodes to ensure training is working
3. **Adjust Learning Rate**: If loss explodes, reduce LR; if no learning, increase LR
4. **Save Frequently**: Models save automatically after training
5. **Test Incrementally**: Test after every 50-100 episodes to track progress

## Citations
```
@inproceedings{azim2024autodspy,
  title     = {AutoDSPy: Automating Modular Prompt Design with Reinforcement Learning for Small and Large Language Models},
  author    = {
    Azim, Nafew and
    {Ur Alam}, Abrar and
    {Bin Omar}, Hasan and
    {Adnan Jami}, Abdullah Mohammad Muntasir and
    {Ibn Ahad}, Jawad and
    Kabir, Muhammad Rafsan and
    Hossain, Md. Ismail and
    Rahman, Fuad and
    Amin, Mohammad Ruhul and
    Rahman, Shafin and
    Mohammed, Nabeel
  },
  booktitle = {Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing (EMNLP 2024)},
  year      = {2024},
  doi       = {10.18653/v1/2024.emnlp-main.597}
}

```
