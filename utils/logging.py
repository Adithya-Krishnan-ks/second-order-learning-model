import json
import os
import time
from typing import Dict, Any

class Logger:
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.step_log_file = os.path.join(log_dir, "step_log.jsonl")
        self.episode_log_file = os.path.join(log_dir, "episode_log.jsonl")
        
        # Clear previous logs if they exist (optional, maybe append is better for resuming?)
        # For now, let's append.

    def log_step(self, step_data: Dict[str, Any]):
        """Logs data for a single inner-loop step."""
        with open(self.step_log_file, "a") as f:
            f.write(json.dumps(step_data) + "\n")

    def log_episode(self, episode_data: Dict[str, Any]):
        """Logs data for a generic episode/trajectory."""
        with open(self.episode_log_file, "a") as f:
            f.write(json.dumps(episode_data) + "\n")
            
    def info(self, message: str):
        print(f"[INFO] {message}")
