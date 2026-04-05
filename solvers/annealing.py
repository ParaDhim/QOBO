import numpy as np
import time
from typing import Optional, List, Tuple, Dict

class SimulatedAnnealing:
    """
    Simulated Annealing solver for QUBO problems.
    Can be initialized with a GNN-predicted solution.
    """
    def __init__(self, 
                 T_start: float = 10.0, 
                 T_end: float = 0.01, 
                 cooling_rate: float = 0.995, 
                 steps_per_T: int = 100):
        self.T_start = T_start
        self.T_end = T_end
        self.cooling_rate = cooling_rate
        self.steps_per_T = steps_per_T

    def solve(self, Q: np.ndarray, initial_solution: Optional[np.ndarray] = None, max_steps: Optional[int] = None) -> Dict:
        num_vars = Q.shape[0]
        
        # Initialization
        if initial_solution is not None:
            # GNN gives probabilities, we threshold them to get binary initialization
            curr_sol = (initial_solution > 0.5).astype(float)
        else:
            curr_sol = np.random.randint(0, 2, size=num_vars).astype(float)
            
        initial_energy = curr_sol @ Q @ curr_sol
        curr_energy = initial_energy
        
        best_sol = curr_sol.copy()
        best_energy = curr_energy
        
        T = self.T_start
        energy_history = [curr_energy]
        iterations_to_best = 0
        total_steps = 0
        
        while T > self.T_end:
            for _ in range(self.steps_per_T):
                total_steps += 1
                if max_steps and total_steps > max_steps:
                    return {
                        'best_sol': best_sol,
                        'energy_history': energy_history,
                        'initial_energy': initial_energy,
                        'final_energy': best_energy,
                        'iterations_to_best': iterations_to_best,
                        'total_steps': total_steps
                    }

                # Bit-flip neighbor
                idx = np.random.randint(0, num_vars)
                
                # Delta energy calculation
                delta_e = (1 - 2 * curr_sol[idx]) * (Q[idx, idx] + np.sum((Q[idx, :] + Q[:, idx]) * curr_sol) - 2 * Q[idx, idx] * curr_sol[idx])
                
                # Metropolis criteria
                if delta_e < 0 or np.random.rand() < np.exp(-delta_e / T):
                    curr_sol[idx] = 1 - curr_sol[idx]
                    curr_energy += delta_e
                    
                    if curr_energy < best_energy:
                        best_energy = curr_energy
                        best_sol = curr_sol.copy()
                        iterations_to_best = total_steps
            
            energy_history.append(curr_energy)
            T *= self.cooling_rate
            
        return {
            'best_sol': best_sol,
            'energy_history': energy_history,
            'initial_energy': initial_energy,
            'final_energy': best_energy,
            'iterations_to_best': iterations_to_best,
            'total_steps': total_steps
        }
