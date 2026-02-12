# Phase B: Convergence Acceleration & Multi-Agent Improvement
Target: ≥0.96 Reward | Max 6 low-risk patches

## Problem Analysis
- **Phase A Result**: Best reward = -0.14 (negative; target positive 0.96)
- **Root Cause**: Initial loss too high; weak reward signal; limited agent diversity; slow PBT cycles
- **Goal**: Accelerate convergence through:
  - Explicit positive reward normalization
  - Wider meta-parameter exploration from episode 0
  - Faster PBT feedback (every 10 instead of 20 eps)
  - Adaptive meta-loss for WM (reduce when accuracy already >85%)
  - Population diversity bonus (encourage hyperparameter spread)

## Phase B Patch Set (Deploy in Order)

### Patch B1: Positive Reward Normalization
**File**: `training/rewards.py`
**Change**: Shift and rescale reward to always be in [0, 1] + scale by improvement magnitude
**Rationale**: PBT selection needs monotonic reward ranking; negative rewards confuse selection
**Risk**: Low (reward formula change only; no training loop changes)

```python
# After computing reward (at end of compute_reward):
reward = (step_improvement * 5.0 + 
          improvement * 1.0 + 
          recovery_bonus + 
          wm_bonus - 
          0.1 * stability_penalty)

# NEW: Shift & scale to [0, 1] + amplify by improvement magnitude
# Ensure reward is always >= 0 and amplify when improvement is high
improvement_magnitude = abs(improvement) if improvement > 0 else 0.01
reward_shifted = reward + 1.0  # Shift to non-negative
reward_normalized = reward_shifted / (2.0 + improvement_magnitude)  # Scale by magnitude
return max(0.0, reward_normalized)
```

### Patch B2: Aggressive Meta-Parameter Initialization
**File**: `training/meta_training.py` (PopulationMember initialization)
**Change**: Initialize each agent with different LR & plasticity from start (not uniform)
**Rationale**: Population diversity needed from episode 0; mutations alone can't bridge large LR range
**Risk**: Low (init-time change; preserves rest of training loop)

```python
# In PopulationMember.__init__, replace:
# self.meta_lr = meta_lr
# self.plasticity_scale = 1.0

# NEW: Diverse initialization based on member ID
lr_factors = [0.5, 0.8, 1.2, 1.5]  # 4 agents: 0.0005, 0.0008, 0.0012, 0.0015
self.meta_lr = meta_lr * lr_factors[id % len(lr_factors)]

plasticity_factors = [0.7, 0.9, 1.0, 1.3]
self.plasticity_scale = plasticity_factors[id % len(plasticity_factors)]

# Similarly for WM loss weight
wm_factors = [0.05, 0.08, 0.12, 0.15]
self.wm_loss_weight = wm_factors[id % len(wm_factors)]
```

### Patch B3: Faster PBT Cadence
**File**: `training/meta_training.py` (PBT Evolution loop)
**Change**: Run PBT every 10 episodes instead of 20
**Rationale**: Faster feedback loop → faster convergence; with 4 agents, sufficient data at episode 10
**Risk**: Low (schedule change only)

```python
# Change from:
# if episode > 0 and episode % 20 == 0:
# To:
if episode > 0 and episode % 10 == 0:
```

### Patch B4: Adaptive Meta-Loss Weighting
**File**: `training/meta_training.py` (meta-loss composition)
**Change**: Reduce WM loss when WM accuracy is already high (>0.85)
**Rationale**: High WM accuracy means WM is already learning well; focus meta-loss on base model learning
**Risk**: Low (loss composition change; backward-compatible)

```python
# In rollout loop, replace:
# if prev_wm_pred is not None and wm_predictions is not None:
#     wm_loss = member.wm_loss_weight * (...)
#     meta_loss += wm_loss

# NEW: Adaptive scaling based on current WM accuracy
if prev_wm_pred is not None and wm_predictions is not None:
    wm_loss_error = (prev_wm_pred['pred_loss'] - loss) ** 2
    wm_errors.append(wm_loss_error.detach().item())
    
    if len(trajectory_losses) >= 2:
        actual_trend = trajectory_losses[-1] - trajectory_losses[-2]
        wm_trend_error = (prev_wm_pred['pred_trend'] - actual_trend) ** 2
    else:
        wm_trend_error = 0.0
    
    # Adaptive weighting: reduce if WM already accurate
    current_wm_acc = 1.0 - np.mean(wm_errors) if wm_errors else 0.0
    adaptive_wm_weight = member.wm_loss_weight * (1.0 if current_wm_acc < 0.85 else 0.3)
    
    wm_loss = adaptive_wm_weight * (wm_loss_error + 0.5 * wm_trend_error)
    meta_loss += wm_loss
```

### Patch B5: Population Diversity Tracking & Bonus
**File**: `training/meta_training.py` (PBT Evolution section)
**Change**: Add diversity bonus to reward + incentivize hyperparameter spread
**Rationale**: Prevent premature convergence to single strategy; multi-agent improvement requires heterogeneity
**Risk**: Low (post-hoc reward adjustment in PBT)

```python
# In PBT Evolution section, after sorting by reward:
population.sort(key=lambda m: (m.recent_reward - baseline), reverse=True)

# NEW: Compute and reward diversity
lr_diversity = np.std([m.meta_lr for m in population])
plasticity_diversity = np.std([m.plasticity_scale for m in population])
wm_diversity = np.std([m.wm_loss_weight for m in population])

total_diversity = lr_diversity + plasticity_diversity + wm_diversity
diversity_bonus_scale = 0.05 * min(1.0, total_diversity / 0.5)  # Bonus if diversity > 0.5

# Apply diversity bonus to all members (encourage exploration)
for m in population:
    m.recent_reward += diversity_bonus_scale

print(f"    Diversity (LR_std={lr_diversity:.5f}, Plasticity_std={plasticity_diversity:.3f}, "
      f"WM_std={wm_diversity:.3f}); Bonus={diversity_bonus_scale:.4f}")
```

### Patch B6: Early Stopping on Divergence
**File**: `training/meta_training.py` (after optimizer.step())
**Change**: Detect NaN/Inf in parameters and reset agent instead of just skipping update
**Rationale**: Prevents agents from getting stuck in bad state; increases exploration robustness
**Risk**: Low (safety net; rare trigger)

```python
# After optimizer.step() (or rollback), detect param divergence:

# Check for NaN/Inf in parameters
param_diverged = False
for p_name, p in [(n, m) for n, m in member.learner.named_parameters()]:
    if torch.isnan(p).any() or torch.isinf(p).any():
        param_diverged = True
        print(f"Parameter divergence in member {member.id}, param {p_name}; resetting agent.")
        break

if param_diverged:
    # Reset learner to initial state (clone from base)
    member.learner = copy.deepcopy(base_model)
    for p in member.learner.parameters():
        p.requires_grad = True
    # Re-init optimizer
    member.params = list(member.learner.parameters()) + \
                    list(member.meta_controller.parameters()) + \
                    list(member.world_model.parameters())
    member.optimizer = torch.optim.Adam(member.params, lr=member.meta_lr)
    member.recent_reward = -10.0  # Penalty for divergence
```

## Deployment Order
1. B1 (Reward Normalization) — Ensures positive signal
2. B2 (Meta-Param Initialization) — Seed population diversity
3. B3 (Faster PBT Cadence) — Accelerate selection feedback
4. B4 (Adaptive Meta-Loss) — Balance WM and base model training
5. B5 (Diversity Bonus) — Encourage heterogeneous population
6. B6 (Divergence Detection) — Safety net

## Testing Checklist
- [ ] Run 60-episode harness on Phase B
- [ ] Verify best_reward > 0 & trending upward
- [ ] Check diversity metrics increasing over time
- [ ] Confirm WM accuracy remains stable (>0.7)
- [ ] No agent divergences (no param NaN/Inf resets)
- [ ] PBT mutations accepted >50% of the time
- [ ] Final best_reward >= 0.96 (target)

## Rollback Criteria
- If best_reward < Phase A baseline (-0.14) at episode 40, rollback immediately
- If >2 divergences per 10 episodes, disable Patch B6
- If diversity bonus causes oscillations (reward variance >2x), disable Patch B5
