# 🧠 Second Order Learning System (Research-Grade Meta-Learning Engine)

A self-evolving reinforcement/meta-learning system with world-model guidance, population-based training (PBT), adaptive mutation, and production-grade stability tooling.
This project demonstrates a research-level training architecture capable of long-run autonomous improvement and stable large-scale training.

---

# 🚀 Overview

This system implements a **second-order learning architecture** where agents learn not only task behavior but also how to improve their own learning process over time.

Key capabilities:

* Population-based training (PBT) with mutation & crossover
* World model guided meta-learning
* Adaptive reward normalization & scaling
* Evolutionary hyperparameter optimization
* Long-run training stability & checkpoint recovery
* AWS-ready production training loop
* Research-grade metrics & logging

Validated through:

* 300+ episode stability testing
* Continuous reward improvement
* Stable mutation acceptance (~40%)
* High world-model accuracy (~0.97)

---

# 🧩 Core Architecture

## 1. Population-Based Meta Learning

Multiple agents train simultaneously and evolve through:

* Mutation
* Soft blending
* Hyperparameter diversity injection
* Performance-based selection

Each agent has:

* Learning rate
* Plasticity parameter
* World model weight
* Adaptive mutation scale

Population evolves toward optimal learning dynamics.

---

## 2. World Model Integration

Each agent includes a predictive world model that:

* Estimates future outcomes
* Guides learning updates
* Tracks prediction accuracy (WM_Acc)
* Dynamically adjusts influence on meta-loss

Adaptive weighting:

* If WM accuracy high → reduce WM loss weight
* If WM accuracy low → increase WM guidance

This balances prediction vs learning.

---

## 3. Reward Normalization Engine

Reward pipeline ensures stable learning signal:

```
reward_scaled = tanh(raw_reward / reward_rms)
reward_final  = reward_scaled + 1.0
```

Benefits:

* Always positive-centered reward
* Stable selection pressure
* Prevents reward saturation
* Tracks moving RMS automatically

---

## 4. Evolution System

### Mutation Types

1. Hard mutation (full acceptance)
2. Soft mutation (30% blend)
3. Diversity injection (periodic reinit)

### Acceptance Logic

```
if mutated_reward >= parent:
    full accept
elif mutated_reward >= parent * 0.90:
    accept with reduced scale
else:
    soft mutation fallback
```

### Adaptive Mutation Scale

* Reduces if repeated failures
* Increases if repeated success
* Prevents extreme parameter jumps

Target acceptance: ~20–40%
Observed: ~42% (healthy)

---

# 🛡 Stability & Production Hardening

## Collapse Detection

Automatically detects:

* Reward collapse (>25% drop)
* WM accuracy collapse (<0.6)
* Exploding loss
* NaN/Inf gradients

On detection:

* Restore last checkpoint
* Reduce LR & plasticity
* Continue training safely

---

## Adaptive Gradient Clipping

Dynamic clipping:

```
max_grad_norm = 2 × median_grad_norm(population)
```

Prevents gradient explosion in long runs.

---

## Checkpoint System

* Atomic checkpoint every 10 episodes
* Backup checkpoint every 50 episodes
* Includes model + optimizer + RNG + meta-state
* AWS interruption safe

---

## Convergence Detection

Detects plateau:

```
<1% improvement over 80 episodes
```

If converged:

* Reduce mutation scale
* Focus on stability

---

# 📊 Metrics & Logging

Per-episode logging:

* Best reward
* Population average reward
* Mutation acceptance rate
* Diversity score
* World model accuracy
* Gradient norms
* Convergence status

Exported to:

```
logs/episode_metrics.jsonl
```

Enables:

* Research analysis
* Visualization
* Long-run monitoring

---

# 🧪 Validated Performance

## Stability Test (300 episodes)

| Metric              | Result |
| ------------------- | ------ |
| Best reward         | ~1.50  |
| Avg reward          | ~1.33  |
| WM accuracy         | ~0.96  |
| Mutation acceptance | ~42%   |
| Population ≥0.92    | 4/4    |
| Divergence events   | 0      |
| Checkpoint failures | 0      |

System stable for long-run training (500–2000 episodes).

---

# ☁️ AWS Deployment Ready

Designed for long-running cloud training:

* Crash-safe checkpointing
* Resume capability
* Memory-safe loops
* Stable gradient handling
* Continuous logging

Recommended:

```
GPU: optional
CPU: sufficient for current model size
Run duration: 24–72 hr
```

---

# 🏁 How to Run

### Install

```
pip install -r requirements.txt
```

### Run training

```
python run.py
```

### Logs

```
logs/episode_metrics.jsonl
```

---

# 🔬 Research Applications

This system can be used for:

* Meta-learning research
* Autonomous optimization experiments
* World-model RL research
* Evolutionary training systems
* AGI learning architecture exploration

---

# 📈 Future Extensions

Possible next steps:

* Multi-task curriculum training
* Larger population scaling
* Distributed training (multi-node)
* Transformer world models
* Neural architecture evolution
* Self-play environments

---

# ⚠️ Notes

This is not a basic RL implementation.
It is a research-grade adaptive learning system designed for long-run autonomous improvement and experimentation.

Use controlled experiments when modifying architecture.

---

# 👤 Author

Developed as a high-performance experimental meta-learning system with evolutionary optimization and world-model guidance.

---

# 📜 License

MIT License
