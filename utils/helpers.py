import numpy as np
import yaml
import os

def load_config(config_path: str = "config/config.yaml"):
    if not os.path.exists(config_path):
        # Default config
        config = {
            'problem': {
                'type': 'n_queens',
                'size': 4,
                'penalty': 10.0
            },
            'annealing': {
                'T_start': 10.0,
                'T_end': 0.01,
                'cooling_rate': 0.99,
                'steps': 100
            },
            'ml': {
                'hidden_dim': 64,
                'model_path': 'models/gnn_n_queens.pth'
            }
        }
        return config
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def save_config(config, config_path: str = "config/config.yaml"):
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as f:
        yaml.dump(config, f)

def compute_qubo_energy(x: np.ndarray, Q: np.ndarray) -> float:
    return float(x @ Q @ x)
