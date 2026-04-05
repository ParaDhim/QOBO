import time
import os
import torch
import numpy as np
from typing import List, Dict
from qubo.n_queens import NQueensQUBO
from solvers.classical import ClassicalSolver
from solvers.annealing import SimulatedAnnealing
from ml.gnn_model import QUBO_GNN, get_graph_data_from_qubo
from utils.metrics import get_performance_metrics
from visualization.dashboard import plot_benchmark_results

def run_benchmarks(size: int = 16, penalty: float = 10.0, use_gnn: bool = True, max_sa_steps: int = 500):
    print(f"\n{'='*40}")
    print(f"RESEARCH BENCHMARK: N-Queens Size {size}")
    print(f"Resource Constraint: Max {max_sa_steps} SA steps")
    print(f"{'='*40}\n")
    
    # 1. Build QUBO
    nq = NQueensQUBO(size, penalty)
    Q = nq.build_qubo()
    
    results = []
    histories = {}
    
    # 2. Classical Solver (Ground Truth if possible)
    print("Running Classical Baseline...")
    start = time.time()
    classical_sol = ClassicalSolver.solve_n_queens(size)
    runtime_c = time.time() - start
    if classical_sol is not None:
        metrics_c = get_performance_metrics("Classical", classical_sol, Q, runtime_c)
        results.append(metrics_c)
        print(f"  Classical found solution! Energy: {metrics_c['final_energy']}")
    
    # 3. Simulated Annealing (Random Initialization)
    print(f"Running SA (Random Init) with {max_sa_steps} steps limit...")
    sa_random = SimulatedAnnealing(T_start=20.0, T_end=0.01, cooling_rate=0.95, steps_per_T=50)
    start = time.time()
    res_rand = sa_random.solve(Q, initial_solution=None, max_steps=max_sa_steps)
    runtime_rand = time.time() - start
    metrics_rand = get_performance_metrics("SA (Random)", res_rand['best_sol'], Q, runtime_rand, 
                                           initial_energy=res_rand['initial_energy'],
                                           iterations_to_best=res_rand['iterations_to_best'])
    results.append(metrics_rand)
    histories['SA (Random)'] = res_rand['energy_history']
    
    # 4. Simulated Annealing (GNN Guided)
    if use_gnn:
        print(f"Running SA (GNN Init) with {max_sa_steps} steps limit...")
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = QUBO_GNN(node_features=2, edge_features=1, hidden_dim=64).to(device)
            model_path = f"models/gnn_n_queens.pth"
            if os.path.exists(model_path):
                model.load_state_dict(torch.load(model_path, map_location=device))
                model.eval()
                
                Q_torch = torch.from_numpy(Q).float().to(device)
                node_feats, edge_idx, edge_attr = get_graph_data_from_qubo(Q_torch)
                
                with torch.no_grad():
                    gnn_probs = model(node_feats, edge_idx, edge_attr).cpu().numpy()
                
                start = time.time()
                res_gnn = sa_random.solve(Q, initial_solution=gnn_probs, max_steps=max_sa_steps)
                runtime_gnn = time.time() - start
                metrics_gnn = get_performance_metrics("SA (GNN)", res_gnn['best_sol'], Q, runtime_gnn,
                                                      initial_energy=res_gnn['initial_energy'],
                                                      iterations_to_best=res_gnn['iterations_to_best'])
                results.append(metrics_gnn)
                histories['SA (GNN)'] = res_gnn['energy_history']
            else:
                print(f"  Warning: GNN model not found at {model_path}. Run with --train first.")
        except Exception as e:
            print(f"  Error in GNN benchmark: {e}")

    # Plot results
    from visualization.dashboard import plot_energy_history
    plot_benchmark_results(results)
    plot_energy_history(histories, title=f"Convergence Stress Test (N={size}, Steps={max_sa_steps})")
    
    print(f"\n{'='*60}")
    print(f"{'Solver':<15} | {'Init Energy':<12} | {'Final Energy':<12} | {'To Best':<8} | {'Runtime':<8}")
    print(f"{'-'*60}")
    for r in results:
        init_e = f"{r['initial_energy']:.1f}" if r['initial_energy'] is not None else "N/A"
        iter_b = f"{r['iterations_to_best']}" if r['iterations_to_best'] is not None else "N/A"
        print(f"{r['solver']:<15} | {init_e:<12} | {r['final_energy']:<12.1f} | {iter_b:<8} | {r['runtime']:.4f}s")
    print(f"{'='*60}\n")
    
    return results

if __name__ == "__main__":
    run_benchmarks()
