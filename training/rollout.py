import torch
import torch.func as tf
from utils.metrics import calculate_update_norm, calculate_gradient_alignment

def rollout_step(params, buffers, model, loss_fn, x, y, state, learner, meta_controller, world_model=None):
    """
    Functional rollout step using torch.func (functional_call).
    Supports differentiation through the update step (meta-learning).
    
    Args:
        params: Dict of model parameters (initial or current)
        buffers: Dict of model buffers
        model: The base model instance (stateless)
        loss_fn: Loss function
        x, y: Input data
        state: LearningState object (has hidden_state)
        learner: LearningDynamicsEngine instance
        meta_controller: MetaController instance
        world_model: LearningWorldModel instance
        
    Returns:
        loss (scalar tensor)
        new_params (dict)
        metrics (dict)
    """
    
    # 1. Functional Forward Pass
    def forward_loss(p, b, data_x, data_y):
        pred = tf.functional_call(model, (p, b), (data_x,))
        # Adapt output dimension to target dimension (Prompt 5 support)
        if len(data_y.shape) > 1 and pred.shape[1] > data_y.shape[1]:
            pred = pred[:, :data_y.shape[1]]
        return loss_fn(pred, data_y)

    grads = tf.grad(forward_loss)(params, buffers, x, y)
    loss = forward_loss(params, buffers, x, y)
    
    # 2. Get Meta-Controller Outputs (Prompt 6)
    # Create simple metrics vector for meta-controller: [loss, step] + stats
    # For simplicity, just use loss and step.
    mc_input = torch.tensor([loss.item(), state.step], dtype=torch.float32, device=x.device).unsqueeze(0)
    loss_weight, gate, noise_scale = meta_controller(mc_input)
    
    # Apply loss weight (this scales the loss for the *next* or *current* update? 
    # Usually it scales the loss used for gradients, but we already computed grads.
    # Let's scale the gradients instead, effectively the same for linear ops.)
    # Gradients are a dict.
    
    # Flatten gradients for the learner
    # Note: grads is a dict matching params structure
    flat_grads = []
    for k in params.keys():
        flat_grads.append(grads[k].flatten())
    grad_vec = torch.cat(flat_grads)
    
    # 3. Learner Update (Prompt 4, 1)
    learner_input = grad_vec * loss_weight
    
    # State update and Learner Step
    delta, divergence = learner(learner_input.unsqueeze(0), state)
    delta = delta.squeeze(0)
    
    # 3.5 Counterfactual Update Selection via World Model
    wm_predictions = None
    wm_metrics = {
        'wm_divergence_prob': 0.0,
        'wm_pred_loss': 0.0,
        'wm_pred_trend': 0.0,
        'wm_penalty': 0.0,
        'num_candidates': 1,
        'selected_candidate': 0
    }
    
    if world_model is not None:
        # Generate K candidate updates
        num_candidates = 5
        exploration_prob = 0.1
        
        candidates = []
        
        # Candidate 0: Original proposed update
        candidates.append(delta)
        
        # Candidates 1-4: Perturbations
        # - Scaled versions (0.5x, 1.5x)
        # - Noisy versions
        candidates.append(delta * 0.5)
        candidates.append(delta * 1.5)
        candidates.append(delta + torch.randn_like(delta) * 0.1 * delta.std())
        candidates.append(delta + torch.randn_like(delta) * 0.2 * delta.std())
        
        # Prepare shared inputs
        state_embed = state.hidden_state
        loss_embed = loss.unsqueeze(0).unsqueeze(0)
        
        # Evaluate each candidate
        candidate_scores = []
        candidate_predictions = []
        
        for i, candidate_delta in enumerate(candidates):
            # Compute action statistics
            delta_stats = torch.stack([
                candidate_delta.norm(),
                candidate_delta.mean(),
                candidate_delta.std(),
                candidate_delta.max()
            ]).unsqueeze(0)
            
            # Query World Model
            pred_loss, pred_trend, div_prob = world_model(state_embed, delta_stats, loss_embed)
            
            # Compute selection score: minimize predicted loss + divergence risk
            # Lower score = better
            score = pred_loss.squeeze() + 2.0 * div_prob.squeeze()  # Weight divergence heavily
            
            candidate_scores.append(score)
            candidate_predictions.append({
                'pred_loss': pred_loss.squeeze(),
                'pred_trend': pred_trend.squeeze(),
                'div_prob': div_prob.squeeze()
            })
        
        # Stack scores for selection
        scores_tensor = torch.stack(candidate_scores)
        
        # Selection: argmin with exploration
        if torch.rand(1).item() < exploration_prob:
            # Explore: random candidate
            selected_idx = torch.randint(0, num_candidates, (1,)).item()
        else:
            # Exploit: best predicted candidate
            selected_idx = torch.argmin(scores_tensor).item()
        
        # Apply selected update
        delta = candidates[selected_idx]
        wm_predictions = candidate_predictions[selected_idx]
        
        # Metrics (from selected candidate)
        wm_metrics['wm_divergence_prob'] = wm_predictions['div_prob'].detach().item()
        wm_metrics['wm_pred_loss'] = wm_predictions['pred_loss'].detach().item()
        wm_metrics['wm_pred_trend'] = wm_predictions['pred_trend'].detach().item()
        wm_metrics['num_candidates'] = num_candidates
        wm_metrics['selected_candidate'] = selected_idx
        
        # Soft safety: if selected candidate still risky, scale down
        if wm_predictions['div_prob'].detach().item() > 0.8:
            penalty_scale = (1.0 - wm_predictions['div_prob'].detach()).clamp(min=0.1)
            delta = delta * penalty_scale
            wm_metrics['wm_penalty'] = 1.0

    # 4. Apply Update with Gate and Noise (Prompt 6)
    new_params = {}
    offset = 0
    divergence = False 
    
    update_norm = delta.norm() 
    
    for k, p in params.items():
        size = p.numel()
        d_p = delta[offset:offset+size].view_as(p)
        
        # Add exploration noise
        noise = torch.randn_like(p) * noise_scale
        
        # Apply update
        new_params[k] = p + gate * d_p + noise
        
        offset += size
        
    # Metrics
    metrics = {
        'loss': loss.item(),
        'grad_norm': grad_vec.norm().item(),
        'update_norm': update_norm.item(),
        'divergence': divergence,
        'gate': gate.item(),
        'loss_weight': loss_weight.item()
    }
    
    # Merge World Model metrics
    metrics.update(wm_metrics)
    
    # Update state counters (non-differentiable)
    state.update(loss.item())
    
    return loss, new_params, metrics, wm_predictions
