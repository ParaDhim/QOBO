import time
import numpy as np
from typing import Dict

def get_performance_metrics(solver_name: str, 
                            solution: np.ndarray, 
                            Q: np.ndarray, 
                            runtime: float, 
                            initial_energy: float = None,
                            iterations_to_best: int = None,
                            ground_truth_energy: float = None) -> Dict:
    """
    Calculates research-grade metrics for a solver's output.
    """
    final_energy = solution @ Q @ solution
    metrics = {
        'solver': solver_name,
        'runtime': runtime,
        'final_energy': final_energy,
        'initial_energy': initial_energy,
        'iterations_to_best': iterations_to_best,
        'energy_drop': (initial_energy - final_energy) if initial_energy is not None else 0.0,
        'is_successful': final_energy <= (ground_truth_energy if ground_truth_energy is not None else 0.0)
    }
    
    if ground_truth_energy is not None:
        metrics['energy_error'] = abs(final_energy - ground_truth_energy)
        
    return metrics
