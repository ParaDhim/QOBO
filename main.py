import argparse
import os
import torch
import numpy as np
from utils.helpers import load_config
from ml.train import train_gnn
from experiments.benchmarks import run_benchmarks
from qubo.n_queens import NQueensQUBO
from solvers.annealing import SimulatedAnnealing
from ml.gnn_model import QUBO_GNN, get_graph_data_from_qubo
from visualization.dashboard import plot_energy_history

def main():
    parser = argparse.ArgumentParser(description="Hybrid Quantum-Classical QUBO Solver")
    parser.add_argument("--train", action="store_true", help="Train GNN model")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmarks")
    parser.add_argument("--size", type=int, default=16, help="Problem size")
    parser.add_argument("--max-steps", type=int, default=1000, help="Max SA steps for benchmark stress test")
    args = parser.parse_args()

    # Create directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("visualization", exist_ok=True)

    if args.train:
        train_gnn(problem_type="n_queens", size=args.size, epochs=150)

    if args.benchmark:
        run_benchmarks(size=args.size, max_sa_steps=args.max_steps)
    else:
        # Default run: solve one instance
        print(f"Solving N-Queens for size {args.size}...")
        nq = NQueensQUBO(args.size)
        Q = nq.build_qubo()
        
        sa = SimulatedAnnealing(T_start=20, T_end=0.01, cooling_rate=0.99)
        
        # Try GNN initialization
        initial_sol = None
        model_path = f"models/gnn_n_queens.pth"
        if os.path.exists(model_path):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = QUBO_GNN(node_features=2, edge_features=1, hidden_dim=64).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            
            Q_torch = torch.from_numpy(Q).float().to(device)
            x, edge_idx, edge_attr = get_graph_data_from_qubo(Q_torch)
            with torch.no_grad():
                initial_sol = model(x, edge_idx, edge_attr).cpu().numpy()
            print(f"GNN-guided initialization loaded (using {device}).")
        
        res = sa.solve(Q, initial_solution=initial_sol)
        sol = res['best_sol']
        history = res['energy_history']
        
        print(f"Final Energy: {sol @ Q @ sol}")
        plot_energy_history({"SA": history})
        print("Plots saved to visualization/energy_plot.png")

if __name__ == "__main__":
    # Ensure config path is handled correctly if needed, but here we use argparse for simplicity
    main()
