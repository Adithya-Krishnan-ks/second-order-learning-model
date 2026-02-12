import torch
import torch.nn.functional as F
import numpy as np

def calculate_update_norm(delta):
    """Calculates the L2 norm of the parameter update."""
    if isinstance(delta, torch.Tensor):
        return delta.norm().item()
    return 0.0

def calculate_gradient_alignment(grad_vec, prev_grad_vec):
    """
    Calculates the cosine similarity between current and previous gradients.
    Returns 0.0 if prev_grad_vec is None.
    """
    if prev_grad_vec is None:
        return 0.0
    return F.cosine_similarity(grad_vec.unsqueeze(0), prev_grad_vec.unsqueeze(0)).item()

def estimate_entropy(probs):
    """Estimates entropy of a probability distribution."""
    if isinstance(probs, torch.Tensor):
        return -torch.sum(probs * torch.log(probs + 1e-8)).item()
    return 0.0

def detect_learning_phase(loss_history, window=10):
    """
    Detects the current learning phase based on loss history.
    Returns: 'exploration', 'stable', 'plateau', or 'divergence'.
    """
    if len(loss_history) < window:
        return 'exploration'
    
    recent_losses = loss_history[-window:]
    avg_loss = np.mean(recent_losses)
    std_loss = np.std(recent_losses)
    
    if std_loss > avg_loss * 0.5: # High variance
        return 'divergence'
    elif std_loss < avg_loss * 0.01: # Very low variance
        return 'plateau'
    else:
        return 'stable'


def compute_population_metrics(population):
    """Compute aggregate metrics for a population and return structured dict.

    Includes best/avg reward, population size, and average WM accuracy (if present).
    """
    rewards = [float(getattr(p, 'reward', float('nan'))) for p in population]
    wm_accs = [float(getattr(p, 'wm_acc', 0.0)) for p in population]
    valid_rewards = [r for r in rewards if not (r != r)]  # filter NaN
    return {
        'best_reward': max(valid_rewards) if valid_rewards else float('nan'),
        'avg_reward': sum(valid_rewards)/len(valid_rewards) if valid_rewards else float('nan'),
        'pop_size': len(population),
        'avg_wm_acc': sum(wm_accs)/len(wm_accs) if wm_accs else float('nan')
    }
