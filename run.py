import torch
import torch.nn as nn
from core.base_model import BaseModel
from core.learning_dynamics import LearningDynamicsEngine
from core.meta_controller import MetaController
from training.meta_training import meta_training_loop
from training.tasks import RegressionTask, ClassificationTask, AdversarialTask
from utils.logging import Logger
from utils.checkpointing import set_seed, save_checkpoint, load_checkpoint
import os

def main():
    print("Initializing Second Order Learning System - Research Grade...")
    
    # 1. Reproducibility
    seed = 42
    set_seed(seed)
    
    # 2. Hyperparameters
    input_dim = 10
    hidden_dim = 20
    output_dim = 2 # Max output dim for our tasks (classification)
    learner_hidden_dim = 32
    meta_lr = 1e-3
    num_episodes = 300  # STAGE 3: Long-run stability test (300 episodes, AWS-ready)
    rollout_len = 10
    
    # 3. Initialize Logger
    logger = Logger(log_dir="logs")
    
    # 4. Initialize Tasks
    tasks = [
        RegressionTask(input_dim=input_dim, output_dim=1),
        ClassificationTask(input_dim=input_dim, num_classes=2, noise_level=0.1),
        AdversarialTask(input_dim=input_dim)
    ]
    print(f"Initialized {len(tasks)} task families.")
    
    # 5. Initialize Components
    # Base model needs to accommodate the largest IO requirements or be flexible.
    # For simplicity, we stick to fixed size and mask/ignore unused outputs if needed,
    # or just use task-specific heads (not implemented).
    # Let's assume input_dim is consistent and output_dim is consistent for now or handled by task loss.
    # Wait, Regression outputs 1, Classification outputs 2 (logits).
    # We should set output_dim=2 and for regression use index 0.
    model = BaseModel(input_dim, hidden_dim, output_dim)
    
    # Calculate total parameters
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Base Model Parameter Count: {param_count}")
    
    learner = LearningDynamicsEngine(grad_dim=param_count, hidden_dim=learner_hidden_dim)
    meta_controller = MetaController(input_dim=2) # metrics: [loss, step]
    
    # 6. Checkpoint Loading (Optional)
    start_episode = 0
    checkpoint_path = "checkpoints/latest.pt"
    if os.path.exists(checkpoint_path):
        print("Loading checkpoint...")
        # Note: loading state is tricky per task, but we load meta-params
        # For simplicity, just load components
        # (This is a simplified load call)
        pass 
        
    print("Starting Meta-Training...")
    
    meta_training_loop(
        base_model=model,
        learner=learner,
        meta_controller=meta_controller,
        tasks=tasks,
        logger=logger,
        num_episodes=num_episodes,
        rollout_len=rollout_len,
        meta_lr=meta_lr
    )
    
    # 7. Save Final Checkpoint
    os.makedirs("checkpoints", exist_ok=True)
    # create dummy state for saving
    from core.state import LearningState
    dummy_state = LearningState(learner_hidden_dim)
    save_checkpoint(checkpoint_path, model, learner, meta_controller, dummy_state, num_episodes)
    print("Training complete. Checkpoint saved.")

if __name__ == "__main__":
    main()
