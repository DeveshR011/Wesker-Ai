# See wesker_ai/README.md for full documentation
import random
import numpy as np
import torch
import os

try:
    from .config import Config
    from .train import Trainer
except ImportError:
    # Allow direct execution: py wesker_ai/main.py
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from wesker_ai.config import Config
    from wesker_ai.train import Trainer

def set_deterministic_seed(seed: int):
    """Enforce deterministic seeds across numpy, random, and torch."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU
        
    # Enforce deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"Set all random seeds to {seed} and enforced determinism.")

def main():
    # Load default configuration
    config = Config()
    
    # Enforce deterministic seeds
    set_deterministic_seed(config.seed)
    
    # Initialize trainer and start training
    trainer = Trainer(config)
    trainer.train()

if __name__ == "__main__":
    main()
