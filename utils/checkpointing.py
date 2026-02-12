import torch
import numpy as np
import random
import os

def set_seed(seed: int):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_checkpoint(path: str, model, learner, meta_controller, state, episode: int):
    """Saves a checkpoint of the entire system."""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'learner_state_dict': learner.state_dict(),
        'meta_controller_state_dict': meta_controller.state_dict(),
        'learning_state': state, # Assuming state is picklable/simple
        'episode': episode
    }
    # Atomic save: write to temp file then replace to avoid partial files (AWS/EBS safety)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    tmp_path = path + ".tmp"
    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, path)

def load_checkpoint(path: str, model, learner, meta_controller):
    """Loads a checkpoint."""
    if not os.path.exists(path):
        return None
        
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    learner.load_state_dict(checkpoint['learner_state_dict'])
    meta_controller.load_state_dict(checkpoint['meta_controller_state_dict'])
    
    # Return state and episode to be handled by caller
    return checkpoint['learning_state'], checkpoint['episode']