import numpy as np
import torch


class RewardNormalizer:
    """Tracks moving RMS of raw rewards for adaptive normalization."""
    def __init__(self, ema_alpha=0.1):
        self.ema_alpha = ema_alpha
        self.reward_rms = 1.0
        self.reward_var = 1.0
        
    def update(self, raw_reward):
        """Update RMS estimate with new raw reward."""
        reward_sq = raw_reward ** 2
        self.reward_var = (1.0 - self.ema_alpha) * self.reward_var + self.ema_alpha * reward_sq
        self.reward_rms = np.sqrt(self.reward_var + 1e-8)
        
    def normalize(self, raw_reward):
        """Scale reward: tanh(raw / rms) + 1.0 for positive centering."""
        scaled = np.tanh(raw_reward / (self.reward_rms + 1e-8))
        normalized = scaled + 1.0  # Shift to [0, 2]
        return normalized


def compute_reward(loss_history, initial_loss, world_model_error=None, wm_accuracy=None):
    """
    Computes a reward signal based on:
    - Speed of loss reduction (improvement over initial and recent)
    - Stability (negative variance)
    - Recovery from divergence
    - World Model Accuracy
    
    Returns raw reward (before RMS normalization).
    """
    if not loss_history:
        return 0.0
        
    current_loss = loss_history[-1]
    
    # 1. Improvement Reward: Positive if loss decreased from initial
    improvement = (initial_loss - current_loss) / (initial_loss + 1e-8)
    
    # 2. Instantaneous Improvement (Speed)
    if len(loss_history) > 1:
        prev_loss = loss_history[-2]
        step_improvement = (prev_loss - current_loss) / (prev_loss + 1e-8)
    else:
        step_improvement = 0.0
        
    # 3. Stability Reward: Penalize high variance in recent history
    if len(loss_history) > 5:
        stability_penalty = np.std(loss_history[-5:])
    else:
        stability_penalty = 0.0
    
    # 4. Recovery Bonus: Detect recovery from loss spikes
    recovery_bonus = 0.0
    if len(loss_history) >= 10:
        # Check if there was a spike and recovery
        recent_window = loss_history[-10:]
        max_loss = max(recent_window)
        if max_loss > initial_loss * 1.5 and current_loss < max_loss * 0.7:
            # Recovered from spike
            recovery_bonus = 0.5
        
    # 5. World Model Accuracy Bonus
    wm_bonus = 0.0
    if wm_accuracy is not None:
        # Reward high prediction accuracy
        wm_bonus = wm_accuracy * 0.2
    elif world_model_error is not None:
        # Legacy: reward low prediction error
        wm_bonus = max(0, 1.0 - world_model_error) * 0.1
        
    # Combine terms
    # Primary: Speed (5x) + Global improvement (1x) + Recovery + WM accuracy
    # Penalty: Instability
    reward = (step_improvement * 5.0 + 
              improvement * 1.0 + 
              recovery_bonus + 
              wm_bonus - 
              0.1 * stability_penalty)
    
    return reward
