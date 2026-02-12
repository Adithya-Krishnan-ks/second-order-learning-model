import torch
import torch.nn as nn
import copy
import random
import numpy as np
import json
import os
from training.tasks import RegressionTask, ClassificationTask, AdversarialTask
from training.rollout import rollout_step
from training.rewards import compute_reward, RewardNormalizer
from core.state import LearningState
from core.learning_world_model import LearningWorldModel
from utils.metrics import detect_learning_phase, compute_population_metrics

# Configuration (Hardcoded defaults or could be passed)
CURRICULUM_PHASES = [
    {'name': 'Phase 1', 'episodes': 200, 'tasks': [RegressionTask]},
    {'name': 'Phase 2', 'episodes': 400, 'tasks': [RegressionTask, ClassificationTask]},
    {'name': 'Phase 3', 'episodes': 400, 'tasks': [AdversarialTask, ClassificationTask]}
]

class PopulationMember:
    def __init__(self, id, learner, meta_controller, world_model, meta_lr=1e-3):
        self.id = id
        # Deepcopy components to ensure independence
        self.learner = copy.deepcopy(learner)
        self.meta_controller = copy.deepcopy(meta_controller)
        self.world_model = copy.deepcopy(world_model)
        
        # Meta-Optimizer updates all three
        self.params = list(self.learner.parameters()) + \
                      list(self.meta_controller.parameters()) + \
                      list(self.world_model.parameters())
                      
        self.optimizer = torch.optim.Adam(self.params, lr=meta_lr)
        self.meta_lr = meta_lr
        
        # Hyperparameters (mutable via PBT)
        self.plasticity_scale = 1.0  # Scales update magnitude
        self.wm_loss_weight = 0.1  # Weight for WM loss in meta-loss
        
        # PATCH B: Adaptive mutation scale
        self.mutation_scale = 0.2  # Adaptive scale for hyperparameter mutations
        self.recent_mutations = []  # Track last 5 mutations (True=accept, False=reject)
        
        # Performance tracking
        self.performance_history = []
        self.recent_reward = 0.0
        self.stability_score = 0.0  # Tracks reward variance
        self.wm_accuracy = 0.0  # Tracks WM prediction accuracy

    def step(self):
        self.optimizer.step()
        self.optimizer.zero_grad()

    def clone_from(self, other):
        """PBT: Replace self with other, then mutate."""
        self.learner.load_state_dict(other.learner.state_dict())
        self.meta_controller.load_state_dict(other.meta_controller.state_dict())
        self.world_model.load_state_dict(other.world_model.state_dict())
        self.optimizer.load_state_dict(other.optimizer.state_dict())
        self.meta_lr = other.meta_lr
        self.plasticity_scale = other.plasticity_scale
        self.wm_loss_weight = other.wm_loss_weight
        
        # Mutation
        self.mutate()

    def mutate(self):
        """Mutate hyperparameters: meta_lr, plasticity_scale, wm_loss_weight.
        
        PATCH B: Uses adaptive mutation_scale controlled by evolutionary feedback.
        """
        # Adaptive mutation range based on mutation_scale
        lower_bound = 1.0 - self.mutation_scale
        upper_bound = 1.0 + self.mutation_scale
        
        # Meta-LR mutation
        factor = np.random.uniform(lower_bound, upper_bound)
        self.meta_lr *= factor
        
        # Plasticity mutation
        plasticity_factor = np.random.uniform(lower_bound, upper_bound)
        self.plasticity_scale *= plasticity_factor
        self.plasticity_scale = np.clip(self.plasticity_scale, 0.1, 2.0)
        
        # WM loss weight mutation
        wm_factor = np.random.uniform(lower_bound, upper_bound)
        self.wm_loss_weight *= wm_factor
        self.wm_loss_weight = np.clip(self.wm_loss_weight, 0.01, 0.5)
        
        # Re-init optimizer with new LR
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.meta_lr


# Curriculum Configuration
CURRICULUM_PHASES = [
    {'name': 'Phase 1: Easy', 'tasks': [RegressionTask], 'rollout_len': 15},
    {'name': 'Phase 2: Mixed', 'tasks': [RegressionTask, ClassificationTask], 'rollout_len': 25},
    {'name': 'Phase 3: Adversarial', 'tasks': [AdversarialTask, ClassificationTask], 'rollout_len': 35}
]

class CurriculumManager:
    def __init__(self, stability_threshold=0.8, wm_accuracy_threshold=0.7, min_episodes=50):
        self.current_phase = 0
        self.episodes_in_phase = 0
        self.stability_threshold = stability_threshold
        self.wm_accuracy_threshold = wm_accuracy_threshold
        self.min_episodes = min_episodes
        
    def get_phase(self):
        return CURRICULUM_PHASES[self.current_phase]
    
    def should_advance(self, population_stability, population_wm_accuracy):
        """Decide if curriculum should advance to next phase."""
        if self.current_phase >= len(CURRICULUM_PHASES) - 1:
            return False  # Already at final phase
        
        if self.episodes_in_phase < self.min_episodes:
            return False  # Not enough episodes yet
        
        # Check if population is ready
        if population_stability >= self.stability_threshold and \
           population_wm_accuracy >= self.wm_accuracy_threshold:
            return True
        
        return False
    
    def advance(self):
        """Move to next curriculum phase."""
        if self.current_phase < len(CURRICULUM_PHASES) - 1:
            self.current_phase += 1
            self.episodes_in_phase = 0
            return True
        return False
    
    def step(self):
        """Increment episode counter."""
        self.episodes_in_phase += 1

def meta_training_loop(
    base_model, 
    learner, 
    meta_controller, 
    tasks, # Ignored in favor of Curriculum, but kept for signature compatibility
    logger=None, 
    num_episodes=1000, 
    rollout_len=20, 
    meta_lr=1e-3
):
    """
    Advanced Meta-Training Loop with PBT, Curriculum, and World Model.
    """
    
    # 1. Initialize Population (Prompt 10 - PBT)
    pop_size = 4
    # Create template World Model if not passed (it wasn't in original run.py, so we init here)
    # We need dimensions. 
    # State dim = learner hidden dim. Action dim = 4 (summary). Recent loss = 1.
    # We can infer from learner.
    learner_hidden = learner.gru.hidden_size
    template_wm = LearningWorldModel(state_dim=learner_hidden, action_dim=4, hidden_dim=64)
    
    population = []
    for i in range(pop_size):
        member = PopulationMember(i, learner, meta_controller, template_wm, meta_lr)
        # Verify gradients enabled
        for p in member.params:
            p.requires_grad = True
        population.append(member)
        
    print(f"Initialized Population of {pop_size} agents.")
    
    # Initialize Curriculum Manager
    curriculum = CurriculumManager(stability_threshold=0.8, wm_accuracy_threshold=0.7, min_episodes=50)
    # EMA baseline for reward normalization (used in PBT selection)
    ema_alpha = 0.2
    ema_avg_reward = None
    # PATCH 1: Reward normalizer for positive-centered signal
    reward_normalizer = RewardNormalizer(ema_alpha=0.1)
    
    # STAGE 1.5: Track episode-level metrics for stability validation
    episode_metrics = {
        'best_rewards': [],
        'avg_rewards': [],
        'avg_losses': [],
        'avg_wm_accs': [],
        'mutation_attempts': 0,
        'mutation_successes': 0,
        'hard_mutations': 0,  # PATCH E-G: Track hard accepts separately
        'soft_mutations': 0   # Track soft mutations separately
    }
    
    # STAGE 3: Long-run stability tracking (PATCH S1, S4, S5)
    stability_tracking = {
        'recent_best_rewards': [],  # Last 5 episode best rewards for collapse detection
        'recent_wm_accs': [],       # Last 5 episode WM accuracies for collapse detection
        'converged': False,          # Convergence flag (PATCH S4)
        'convergence_start_ep': None,  # Episode when convergence detected
        'last_stable_checkpoint': None,  # For rollback (PATCH S1)
        'collapse_detected': False,  # Recovery flag
        'gradient_norms': [],       # Track for adaptive clipping (PATCH S2)
        'max_grad_clip': 2.0       # Adaptive clipping threshold (PATCH S2)
    }

    # Main Loop
    for episode in range(num_episodes):
        # 2. Curriculum: Get current phase
        phase_config = curriculum.get_phase()
        task_cls = random.choice(phase_config['tasks'])
        current_rollout_len = phase_config.get('rollout_len', rollout_len)

        
        # Instantiate task (we need to know input_dim to init? We assume compatible)
        # Our base model is fixed input_dim=10.
        # RegressionTask and ClassificationTask need input_dim kwarg.
        # We'll assume hardcoded 10 for now or infer from base_model.fc1.in_features
        input_dim = base_model.fc1.in_features
        
        if task_cls == RegressionTask:
            task = RegressionTask(input_dim=input_dim, output_dim=1)
        elif task_cls == ClassificationTask:
            task = ClassificationTask(input_dim=input_dim, num_classes=2, noise_level=0.1)
        elif task_cls == AdversarialTask:
            task = AdversarialTask(input_dim=input_dim)
        else:
            task = tasks[0] # Fallback
            
        task.reset()
        
        # 3. Plasticity Annealing (compute current scale)
        annealing_progress = min(episode / 800.0, 1.0)
        target_plasticity = 1.0 - annealing_progress * 0.7  # 1.0 -> 0.3
        
        # 4. Step each population member
        episode_wm_errors = []
        
        for member in population:
            # Clean Base Model for this run
            current_base_model = copy.deepcopy(base_model)
            def reset_weights(m):
                if hasattr(m, 'reset_parameters'):
                    m.reset_parameters()
            current_base_model.apply(reset_weights)
            
            # Params
            params = dict(current_base_model.named_parameters())
            buffers = dict(current_base_model.named_buffers())
            
            # State
            state = LearningState(hidden_dim=learner_hidden)
            first_param = next(iter(params.values()))
            state.hidden_state = state.hidden_state.to(first_param.device)
            
            # Unroll
            meta_loss = 0.0
            trajectory_losses = []
            initial_loss = 0.0
            prev_wm_pred = None
            wm_errors = []  # Track WM prediction errors
            update_norms = []  # Track update magnitudes
            
            for step in range(current_rollout_len):
                x, y = task.sample(batch_size=32)
                x, y = x.to(first_param.device), y.to(first_param.device)
                
                # Rollout Step with World Model
                loss, new_params, metrics, wm_predictions = rollout_step(
                    params, buffers, current_base_model, task.loss_fn, x, y, state, 
                    member.learner, member.meta_controller, member.world_model
                )
                
                if step == 0:
                    initial_loss = loss.item()
                
                trajectory_losses.append(loss.item())
                update_norms.append(metrics.get('update_norm', 0.0))
                
                # Meta-Loss Composition
                # 1. Performance Loss (weighted by step for long-horizon credit)
                step_weight = 0.1 + 0.05 * (step / current_rollout_len)
                meta_loss += step_weight * loss
                
                # 2. World Model Supervision (FIXED: t -> t+1 alignment)
                if prev_wm_pred is not None and wm_predictions is not None:
                    # Supervise prediction from step t-1 with actual loss from step t
                    wm_loss_error = (prev_wm_pred['pred_loss'] - loss) ** 2
                    wm_errors.append(wm_loss_error.detach().item())
                    
                    # Supervise trend prediction (loss_t - loss_{t-1})
                    if len(trajectory_losses) >= 2:
                        actual_trend = trajectory_losses[-1] - trajectory_losses[-2]
                        wm_trend_error = (prev_wm_pred['pred_trend'] - actual_trend) ** 2
                    else:
                        wm_trend_error = 0.0
                    
                    # PATCH 4: Adaptive WM loss weighting based on current WM accuracy
                    # If WM accuracy is already high (>0.85), reduce WM loss contribution
                    # If WM accuracy is low (<0.70), slightly increase weight
                    current_wm_acc = 1.0 - np.mean(wm_errors) if wm_errors else 0.0
                    if current_wm_acc > 0.85:
                        # Reduce WM loss by 40% when already accurate
                        adaptive_wm_weight = member.wm_loss_weight * 0.6
                    elif current_wm_acc < 0.70:
                        # Increase WM loss by 20% when inaccurate
                        adaptive_wm_weight = member.wm_loss_weight * 1.2
                    else:
                        # Normal weight in [0.70, 0.85]
                        adaptive_wm_weight = member.wm_loss_weight
                    
                    # Combined WM loss with adaptive weight
                    wm_loss = adaptive_wm_weight * (wm_loss_error + 0.5 * wm_trend_error)
                    meta_loss += wm_loss
                
                # Store current predictions for next step
                prev_wm_pred = wm_predictions
                params = new_params
                
                # Logging
                if logger and member.id == 0:
                    metrics['step'] = step
                    metrics['episode'] = episode
                    metrics['phase'] = phase_config['name']
                    metrics['member_id'] = member.id
                    metrics['plasticity_target'] = target_plasticity
                    metrics['plasticity_scale'] = member.plasticity_scale
                    logger.log_step(metrics)

            # Final Loss (higher weight for final performance)
            meta_loss += 2.0 * loss
            
            # 5. Plasticity Penalty
            avg_update_norm = np.mean(update_norms) if update_norms else 0.0
            plasticity_penalty = max(0, avg_update_norm - target_plasticity) ** 2
            meta_loss += 0.05 * plasticity_penalty
            
            # 6. Compute Reward with WM accuracy
            member.wm_accuracy = 1.0 - np.mean(wm_errors) if wm_errors else 0.0
            member.wm_accuracy = max(0.0, min(1.0, member.wm_accuracy))
            
            # PATCH 1: Compute raw reward, then normalize with moving RMS
            raw_reward = compute_reward(trajectory_losses, initial_loss, wm_accuracy=member.wm_accuracy)
            reward_normalizer.update(raw_reward)
            reward = reward_normalizer.normalize(raw_reward)
            
            member.recent_reward = 0.9 * member.recent_reward + 0.1 * reward
            member.current_episode_reward = reward
            
            # Track stability
            member.performance_history.append(reward)
            if len(member.performance_history) > 10:
                member.stability_score = 1.0 - np.std(member.performance_history[-10:])
            
            # Store WM errors for curriculum
            episode_wm_errors.extend(wm_errors)
            
            # Snapshot states for safe rollback on numerical issues
            learner_sd = member.learner.state_dict()
            meta_sd = member.meta_controller.state_dict()
            wm_sd = member.world_model.state_dict()
            optim_sd = member.optimizer.state_dict()

            member.optimizer.zero_grad()
            meta_loss.backward()

            # PATCH S2: Adaptive gradient clipping based on gradient norms
            # Compute gradient norm for this member
            total_grad_norm = 0.0
            for p in member.params:
                if p.grad is not None:
                    total_grad_norm += torch.norm(p.grad).item() ** 2
            total_grad_norm = np.sqrt(total_grad_norm)
            
            # Track gradient norm for adaptive threshold
            stability_tracking['gradient_norms'].append(total_grad_norm)
            if len(stability_tracking['gradient_norms']) > 100:
                stability_tracking['gradient_norms'].pop(0)
            
            # Adaptive clip threshold: 2x median gradient norm
            if len(stability_tracking['gradient_norms']) >= 5:
                median_grad = float(np.median(stability_tracking['gradient_norms']))
                stability_tracking['max_grad_clip'] = max(1.0, 2.0 * median_grad)  # At least 1.0
            else:
                stability_tracking['max_grad_clip'] = 1.0
            
            # Apply adaptive clipping
            torch.nn.utils.clip_grad_norm_(member.params, stability_tracking['max_grad_clip'])

            # NaN / Inf detection in grads
            bad_grad = False
            for g in [p.grad for p in member.params if p.grad is not None]:
                if torch.isnan(g).any() or torch.isinf(g).any():
                    bad_grad = True
                    break

            if bad_grad:
                # Rollback to previous parameters and optimizer state
                member.learner.load_state_dict(learner_sd)
                member.meta_controller.load_state_dict(meta_sd)
                member.world_model.load_state_dict(wm_sd)
                member.optimizer.load_state_dict(optim_sd)
                member.optimizer.zero_grad()
                print(f"Numerical issue detected in grads for member {member.id}; rollback applied.")
            else:
                try:
                    member.optimizer.step()
                except Exception as e:
                    # Restore on unexpected optimizer failure
                    member.learner.load_state_dict(learner_sd)
                    member.meta_controller.load_state_dict(meta_sd)
                    member.world_model.load_state_dict(wm_sd)
                    member.optimizer.load_state_dict(optim_sd)
                    member.optimizer.zero_grad()
                    print(f"Optimizer step failed for member {member.id}: {e}; rollback applied.")

        # 7. Self-Play (Comparison Rewards)
        indices = list(range(len(population)))
        random.shuffle(indices)
        for i in range(0, len(indices) - 1, 2):
            idx1, idx2 = indices[i], indices[i+1]
            m1, m2 = population[idx1], population[idx2]
            
            if m1.current_episode_reward > m2.current_episode_reward:
                m1.recent_reward += 0.05
            elif m2.current_episode_reward > m1.current_episode_reward:
                m2.recent_reward += 0.05

        # 8. Curriculum Advancement Check
        curriculum.step()
        population_stability = np.mean([m.stability_score for m in population])
        population_wm_accuracy = np.mean([m.wm_accuracy for m in population])
        # Structured population metrics and EMA update
        pop_metrics = compute_population_metrics(population)
        if ema_avg_reward is None:
            ema_avg_reward = pop_metrics.get('avg_reward', 0.0)
        else:
            ema_avg_reward = ema_alpha * pop_metrics.get('avg_reward', 0.0) + (1.0 - ema_alpha) * ema_avg_reward
        
        if curriculum.should_advance(population_stability, population_wm_accuracy):
            advanced = curriculum.advance()
            if advanced:
                print(f"\n*** CURRICULUM ADVANCED to {curriculum.get_phase()['name']} at episode {episode} ***")
                print(f"    Population Stability: {population_stability:.3f}, WM Accuracy: {population_wm_accuracy:.3f}\n")

        # 9. PBT Evolution (Every K episodes)
        if episode > 0 and episode % 20 == 0:
            # Sort by reward normalized by EMA baseline to reduce nonstationarity
            baseline = ema_avg_reward if ema_avg_reward is not None else pop_metrics.get('avg_reward', 0.0)
            population.sort(key=lambda m: (m.recent_reward - baseline), reverse=True)
            
            # Log Population Status
            print(f"--- PBT Evolution Episode {episode} ({phase_config['name']}) ---")
            for m in population:
                print(f"Member {m.id}: Reward={m.recent_reward:.4f} LR={m.meta_lr:.5f} "
                      f"Plasticity={m.plasticity_scale:.3f} WM_Weight={m.wm_loss_weight:.3f} "
                      f"WM_Acc={m.wm_accuracy:.3f}")
            
            # Soft-elitism: protect top-K members from replacement
            protect_k = min(2, pop_size//2)
            num_replace = max(1, pop_size // 4)
            replaced = 0
            for i in range(num_replace):
                worst_idx = -(i+1)
                # Convert to actual index and skip protected
                worst = population[worst_idx]
                best = population[i]
                if i < protect_k:
                    # best is protected as elite; still allow cloning into worse positions
                    pass

                if worst.id in [m.id for m in population[:protect_k]]:
                    # skip if worst happened to be in protected set
                    continue

                print(f"Attempting replace Member {worst.id} with mutated clone of Member {best.id}")
                # STAGE 1.5: Track mutation attempts
                episode_metrics['mutation_attempts'] += 1
                
                # Save backup for rollback (and soft mutation)
                backup = {
                    'learner': worst.learner.state_dict(),
                    'meta': worst.meta_controller.state_dict(),
                    'wm': worst.world_model.state_dict(),
                    'optim': worst.optimizer.state_dict(),
                    'recent_reward': worst.recent_reward,
                    # PATCH D: Store hyperparameters for soft mutation blending
                    'meta_lr_backup': worst.meta_lr,
                    'plasticity_backup': worst.plasticity_scale,
                    'wm_weight_backup': worst.wm_loss_weight
                }

                # Perform clone+mutate
                worst.clone_from(best)
                worst.recent_reward = best.recent_reward

                # Mutation acceptance test: run brief validation rollout
                def quick_eval(member_obj, eval_steps=20):  # PATCH F: Increased from 12 to 20 for stability
                    # Lightweight eval using a shallow copy of base_model
                    bm = copy.deepcopy(base_model)
                    params = dict(bm.named_parameters())
                    buffers = dict(bm.named_buffers())
                    state = LearningState(hidden_dim=learner_hidden)
                    first_param = next(iter(params.values()))
                    state.hidden_state = state.hidden_state.to(first_param.device)
                    loss_vals = []
                    for _s in range(eval_steps):
                        xb, yb = task.sample(batch_size=32)
                        xb, yb = xb.to(first_param.device), yb.to(first_param.device)
                        loss, new_params, _, _ = rollout_step(params, buffers, bm, task.loss_fn, xb, yb, state,
                                                              member_obj.learner, member_obj.meta_controller, member_obj.world_model)
                        loss_vals.append(float(loss.item()))
                        params = new_params
                    
                    # PATCH E: Convert loss trajectory to normalized reward using full pipeline
                    if not loss_vals:
                        return 0.0
                    
                    # Compute raw reward from loss trajectory
                    initial_loss_eval = loss_vals[0] if loss_vals else 1.0
                    raw_reward_eval = compute_reward(loss_vals, initial_loss_eval, wm_accuracy=0.8)  # Use reasonable WM_acc
                    
                    # Normalize using the main reward_normalizer (same as training)
                    normalized_reward = reward_normalizer.normalize(raw_reward_eval)
                    
                    return float(normalized_reward)

                try:
                    val_reward = quick_eval(worst, eval_steps=20)  # PATCH E & F: Use normalized reward + 20 steps
                except Exception as e:
                    print(f"Validation eval failed for new member {worst.id}: {e}")
                    val_reward = 0.0  # Default if evaluation fails

                # PATCH G: Three-tier mutation acceptance logic
                prev = backup['recent_reward']
                
                if val_reward >= prev:
                    # TIER 1: Full acceptance — mutated agent is better than parent
                    print(f"Replacement FULLY ACCEPTED for Member {worst.id} (val_reward {val_reward:.4f} >= parent {prev:.4f}).")
                    worst.recent_mutations.append(True)
                    episode_metrics['mutation_successes'] += 1
                    episode_metrics['hard_mutations'] += 1  # PATCH E-G: Track hard accept
                    replaced += 1
                    
                elif val_reward >= prev * 0.90:
                    # TIER 2: Conditional acceptance — mutated is slightly worse but still good
                    # Accept the mutation but reduce mutation scale slightly
                    print(f"Replacement CONDITIONALLY accepted for Member {worst.id} "
                          f"(val_reward {val_reward:.4f} in range [{prev*0.90:.4f}, {prev:.4f}]); "
                          f"reducing mutation scale.")
                    worst.recent_mutations.append(True)
                    episode_metrics['mutation_successes'] += 1
                    episode_metrics['hard_mutations'] += 1  # PATCH E-G: Track hard accept (conditional)
                    replaced += 1
                    
                    # Reduce mutation scale for next mutation
                    worst.mutation_scale *= 0.85
                    worst.mutation_scale = np.clip(worst.mutation_scale, 0.05, 0.5)
                    
                else:
                    # TIER 3: Soft mutation — mutated agent is notably worse
                    # PATCH D: Apply 30% of mutation delta instead of full rejection
                    print(f"Replacement SOFT-MUTATED for Member {worst.id} "
                          f"(val_reward {val_reward:.4f} < threshold {prev*0.90:.4f}).")
                    
                    # Reload current mutated state
                    mutated_lr = worst.meta_lr
                    mutated_plasticity = worst.plasticity_scale
                    mutated_wm_weight = worst.wm_loss_weight
                    
                    # Blend: 30% new mutation + 70% original
                    worst.meta_lr = 0.7 * backup['meta_lr_backup'] + 0.3 * mutated_lr
                    worst.plasticity_scale = 0.7 * backup['plasticity_backup'] + 0.3 * mutated_plasticity
                    worst.wm_loss_weight = 0.7 * backup['wm_weight_backup'] + 0.3 * mutated_wm_weight
                    
                    # Update optimizer with blended LR
                    for param_group in worst.optimizer.param_groups:
                        param_group['lr'] = worst.meta_lr
                    
                    worst.recent_reward = 0.7 * prev + 0.3 * val_reward
                    worst.recent_mutations.append(False)  # Track as soft mutation (not hard accept)
                    episode_metrics['soft_mutations'] += 1  # PATCH E-G: Track soft mutation
                
                # PATCH B: Adaptive mutation scale based on recent history
                # If last 5 mutations have 3+ rejections: reduce scale by 30%
                # If last 3 mutations all accepted: increase scale by 15%
                if len(worst.recent_mutations) > 0:
                    recent_hist = worst.recent_mutations[-5:]
                    accepts = sum(1 for m in recent_hist if m)
                    
                    if len(recent_hist) >= 5 and accepts <= 2:  # 2 or fewer accepted in last 5
                        worst.mutation_scale *= 0.7  # Reduce mutations
                        print(f"  Reduced mutation scale for Member {worst.id}: {worst.mutation_scale:.3f}")
                    elif len(recent_hist) >= 3 and accepts == 3:  # All 3 recent accepted
                        worst.mutation_scale *= 1.15  # Increase mutations (explore more)
                        print(f"  Increased mutation scale for Member {worst.id}: {worst.mutation_scale:.3f}")
                    
                    # Clamp to safe range
                    worst.mutation_scale = np.clip(worst.mutation_scale, 0.05, 0.5)
        
        # PATCH C: Periodic diversity injection
        # Every 50 episodes, reinitialize one worst member's hyperparameters
        if episode > 0 and episode % 50 == 0:
            # Find worst member by recent reward
            worst_member = min(population, key=lambda m: m.recent_reward)
            
            # Randomize hyperparameters (keep weights/parameters intact)
            worst_member.meta_lr = np.random.uniform(3e-4, 3e-3)
            worst_member.plasticity_scale = np.random.uniform(0.7, 1.3)
            worst_member.wm_loss_weight = np.random.uniform(0.05, 0.2)
            worst_member.mutation_scale = 0.2  # Reset mutation scale
            
            # Update optimizer with new LR
            for param_group in worst_member.optimizer.param_groups:
                param_group['lr'] = worst_member.meta_lr
            
            print(f"[EP{episode}] PATCH C: Diversity injection on Member {worst_member.id} "
                  f"(LR={worst_member.meta_lr:.5f}, Plast={worst_member.plasticity_scale:.3f}, "
                  f"WM_w={worst_member.wm_loss_weight:.3f})")
        
        # STAGE 1.5: Per-episode comprehensive logging for stability validation
        if episode % 10 == 0 or episode < 5:  # Log every 10 episodes + first 5 episodes
            # Compute metrics
            best_rew = max([m.recent_reward for m in population])
            avg_rew = np.mean([m.recent_reward for m in population])
            std_rew = np.std([m.recent_reward for m in population])
            avg_wm = np.mean([m.wm_accuracy for m in population])
            
            # PATCH C: Diversity metrics
            lr_std = np.std([m.meta_lr for m in population])
            plasticity_std = np.std([m.plasticity_scale for m in population])
            wm_weight_std = np.std([m.wm_loss_weight for m in population])
            diversity_score = (lr_std + plasticity_std + wm_weight_std) / 3.0
            
            # Mutation acceptance rates (hard and soft separately)
            if episode_metrics['mutation_attempts'] > 0:
                hard_mut_rate = episode_metrics['hard_mutations'] / episode_metrics['mutation_attempts']
                soft_mut_rate = episode_metrics['soft_mutations'] / episode_metrics['mutation_attempts']
            else:
                hard_mut_rate = 0.0
                soft_mut_rate = 0.0
            
            # Track for validation
            episode_metrics['best_rewards'].append(best_rew)
            episode_metrics['avg_rewards'].append(avg_rew)
            episode_metrics['avg_wm_accs'].append(avg_wm)
            
            # PATCH S1: Collapse detection (reward drop, WM accuracy collapse, NaN detection)
            stability_tracking['recent_best_rewards'].append(best_rew)
            stability_tracking['recent_wm_accs'].append(avg_wm)
            
            # Keep only last 5 for window-based detection
            if len(stability_tracking['recent_best_rewards']) > 5:
                stability_tracking['recent_best_rewards'].pop(0)
            if len(stability_tracking['recent_wm_accs']) > 5:
                stability_tracking['recent_wm_accs'].pop(0)
            
            # Collapse check 1: Reward dropped >25% in last 5 episodes
            if len(stability_tracking['recent_best_rewards']) >= 5:
                max_recent = max(stability_tracking['recent_best_rewards'][:-1])  # Exclude current
                current = stability_tracking['recent_best_rewards'][-1]
                if max_recent > 0 and (max_recent - current) / max_recent > 0.25:
                    print(f"[WARN] Reward COLLAPSE detected: {max_recent:.4f} → {current:.4f} (drop >25%)")
                    stability_tracking['collapse_detected'] = True
                else:
                    stability_tracking['collapse_detected'] = False
            
            # Collapse check 2: WM accuracy dropped below 0.6
            if avg_wm < 0.6:
                print(f"[WARN] WM Accuracy COLLAPSED: {avg_wm:.3f} < 0.6")
                stability_tracking['collapse_detected'] = True
            
            # Collapse check 3: Check for NaN/Inf in population rewards
            for m in population:
                if not np.isfinite(m.recent_reward):
                    print(f"[ERROR] Non-finite reward detected in Member {m.id}: {m.recent_reward}")
                    stability_tracking['collapse_detected'] = True
            
            # PATCH S4: Convergence detection (reward improvement <1% over 80 episodes)
            if not stability_tracking['converged'] and len(episode_metrics['best_rewards']) > 80:
                reward_80_ago = episode_metrics['best_rewards'][-80]
                current_best = episode_metrics['best_rewards'][-1]
                improvement = (current_best - reward_80_ago) / (abs(reward_80_ago) + 1e-8)
                
                if abs(improvement) < 0.01:  # <1% improvement over 80 episodes
                    print(f"[INFO] CONVERGENCE detected (improvement={improvement*100:.2f}% over 80 eps)")
                    stability_tracking['converged'] = True
                    stability_tracking['convergence_start_ep'] = episode
            
            # If converged: reduce mutation scale for stability focus
            if stability_tracking['converged']:
                for m in population:
                    m.mutation_scale *= 0.7  # Reduce exploration
                    m.mutation_scale = np.clip(m.mutation_scale, 0.05, 0.3)
            
            # PATCH S3: Checkpoint hardening (atomic every 10 episodes, backup every 50)
            if (episode + 1) % 10 == 0:  # Every 10 episodes
                checkpoint_dir = "checkpoints"
                os.makedirs(checkpoint_dir, exist_ok=True)
                
                # Atomic save: write to .tmp then replace
                best_member = max(population, key=lambda m: m.recent_reward)
                checkpoint_data = {
                    'episode': episode + 1,
                    'base_model': None,  # Not stored in member
                    'learner_state': best_member.learner.state_dict(),
                    'meta_controller_state': best_member.meta_controller.state_dict(),
                    'world_model_state': best_member.world_model.state_dict(),
                    'optimizer_state': best_member.optimizer.state_dict(),
                    'reward_normalizer_state': {
                        'reward_rms': reward_normalizer.reward_rms,
                        'reward_mean': reward_normalizer.reward_mean
                    },
                    'best_reward': best_member.recent_reward,
                    'meta_lr': best_member.meta_lr,
                    'plasticity_scale': best_member.plasticity_scale,
                    'wm_loss_weight': best_member.wm_loss_weight,
                    'episode_metrics': episode_metrics,
                    'stability_tracking': stability_tracking
                }
                
                checkpoint_file = os.path.join(checkpoint_dir, "latest.pt")
                tmp_file = os.path.join(checkpoint_dir, "latest.pt.tmp")
                
                try:
                    torch.save(checkpoint_data, tmp_file)
                    os.replace(tmp_file, checkpoint_file)  # Atomic swap
                    if (episode + 1) % 50 == 0:  # Every 50 episodes, also save backup
                        backup_file = os.path.join(checkpoint_dir, f"backup_ep{episode+1}.pt")
                        torch.save(checkpoint_data, backup_file)
                    stability_tracking['last_stable_checkpoint'] = episode + 1
                except Exception as e:
                    print(f"[WARN] Checkpoint failed at EP{episode+1}: {e}")
            
            # PATCH S5: Metrics logging (per-episode JSON export)
            if (episode + 1) % 10 == 0:  # Export every 10 episodes
                metrics_dir = "logs"
                os.makedirs(metrics_dir, exist_ok=True)
                
                metrics_log = {
                    'episode': episode + 1,
                    'best_reward': best_rew,
                    'avg_reward': avg_rew,
                    'std_reward': std_rew,
                    'population_wm_acc': avg_wm,
                    'hard_mutation_rate': hard_mut_rate,
                    'soft_mutation_rate': soft_mut_rate,
                    'diversity_score': diversity_score,
                    'is_converged': stability_tracking['converged'],
                    'max_grad_clip': stability_tracking['max_grad_clip'],
                    'phase': phase_config['name'],
                    'collapse_detected': stability_tracking['collapse_detected']
                }
                
                try:
                    metrics_file = os.path.join(metrics_dir, "episode_metrics.jsonl")
                    with open(metrics_file, 'a') as f:
                        f.write(json.dumps(metrics_log) + '\n')
                except Exception as e:
                    print(f"[WARN] Metrics logging failed at EP{episode+1}: {e}")
            
            print(f"[EP{episode:03d}] Best={best_rew:.4f} Avg={avg_rew:.4f} Std={std_rew:.4f} | "
                  f"WM_Acc={avg_wm:.3f} | HardMut={hard_mut_rate*100:.0f}% SoftMut={soft_mut_rate*100:.0f}% | "
                  f"Div={diversity_score:.4f} | Conv={stability_tracking['converged']} | Phase={phase_config['name']}")
                
    print("Meta-Training Complete.")
    
    # STAGE 1.5: Stability validation report
    print("\n" + "="*70)
    print("STAGE 1.5 STABILITY VALIDATION REPORT")
    print("="*70)
    if episode_metrics['best_rewards']:
        best_list = episode_metrics['best_rewards']
        avg_list = episode_metrics['avg_rewards']
        wm_list = episode_metrics['avg_wm_accs']
        
        print(f"\nReward Trends (logged every ~10 episodes):")
        print(f"  Best Reward: {best_list[0]:.4f} → {best_list[-1]:.4f} (Δ={best_list[-1]-best_list[0]:+.4f})")
        print(f"  Avg Reward:  {avg_list[0]:.4f} → {avg_list[-1]:.4f} (Δ={avg_list[-1]-avg_list[0]:+.4f})")
        print(f"  WM Accuracy: {wm_list[0]:.3f} → {wm_list[-1]:.3f}")
        
        # Check stability
        final_best = best_list[-1]
        min_best = min(best_list)
        max_best = max(best_list)
        reward_range = max_best - min_best
        
        print(f"\nStability Metrics:")
        print(f"  Best Reward Range: [{min_best:.4f}, {max_best:.4f}] (spread={reward_range:.4f})")
        print(f"  Agents with Reward ≥ 0.90: {sum(1 for m in population if m.recent_reward >= 0.90)}/{len(population)}")
        print(f"  Agents with Reward ≥ 0.92: {sum(1 for m in population if m.recent_reward >= 0.92)}/{len(population)}")
        
        # Mutation breakdown (hard vs soft)
        mut_attempts = episode_metrics['mutation_attempts']
        hard_muts = episode_metrics['hard_mutations']
        soft_muts = episode_metrics['soft_mutations']
        if mut_attempts > 0:
            hard_rate = hard_muts / mut_attempts
            soft_rate = soft_muts / mut_attempts
            print(f"\nMutation Breakdown:")
            print(f"  Hard Accepts (TIER 1+2): {hard_muts}/{mut_attempts} ({hard_rate*100:.1f}%)")
            print(f"  Soft Mutations (TIER 3): {soft_muts}/{mut_attempts} ({soft_rate*100:.1f}%)")
        
        # Stability assessment
        if reward_range < 0.1 and final_best >= 0.90:
            stability = "STABLE"
        elif reward_range < 0.2 and final_best >= 0.85:
            stability = "SEMI-STABLE"
        else:
            stability = "UNSTABLE"
        print(f"  Assessment: {stability}")
    print("="*70 + "\n")