import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from qubo.sudoku import SudokuQUBO
from qubo.n_queens import NQueensQUBO
from qubo.graph_coloring import GraphColoringQUBO
from solvers.classical import ClassicalSolver
from tqdm import tqdm
from ml.gnn_model import get_graph_data_from_qubo

class QUBODataset(Dataset):
    def __init__(self, num_samples: int = 100, problem_type: str = "n_queens", size: int = 4):
        self.samples = []
        self._generate_data(num_samples, problem_type, size)

    def _generate_data(self, num_samples: int, problem_type: str, size: int):
        print(f"  Generating {num_samples} samples for N={size}...")
        for _ in tqdm(range(num_samples), desc=f"N={size}", leave=False):
            if problem_type == "n_queens":
                nq = NQueensQUBO(size)
                Q = nq.build_qubo()
                label = ClassicalSolver.solve_n_queens(size)
            elif problem_type == "graph_coloring":
                num_nodes = size
                num_colors = 3
                # Random E-R graph
                edges = []
                for i in range(num_nodes):
                    for j in range(i + 1, num_nodes):
                        if np.random.rand() > 0.5:
                            edges.append((i, j))
                gc = GraphColoringQUBO(num_nodes, num_colors)
                Q = gc.build_qubo(edges)
                label = ClassicalSolver.solve_graph_coloring(num_nodes, num_colors, edges)
            else: # sudoku
                sq = SudokuQUBO(size)
                # Random partial grid (very simple for demo)
                grid = np.zeros((size, size))
                Q = sq.build_qubo(grid)
                label = ClassicalSolver.solve_sudoku(grid)

            if label is not None:
                Q_torch = torch.from_numpy(Q).float()
                node_feats, edge_idx, edge_attr = get_graph_data_from_qubo(Q_torch)
                self.samples.append({
                    'x': node_feats,
                    'edge_index': edge_idx,
                    'edge_attr': edge_attr,
                    'y': torch.from_numpy(label).float()
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def collate_fn(batch):
    """
    Custom collate function for GNN batching.
    Since graphs have different sizes or just to keep it simple, we process one by one
    or aggregate them into a large disjoint graph.
    For this project, we'll keep it simple and process samples individually in loop 
    OR aggregate them correctly.
    """
    return batch
