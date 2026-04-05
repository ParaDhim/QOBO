import numpy as np
from typing import List, Tuple, Dict

class SudokuQUBO:
    """
    Formulates the Sudoku problem as a QUBO (Quadratic Unconstrained Binary Optimization).
    Variables: x_{i,j,k} = 1 if cell (i,j) contains digit k, else 0.
    Total variables: N^3 (for 9x9, N=9, total 729 variables).
    """
    def __init__(self, size: int = 9, penalty: float = 10.0):
        self.N = size
        self.M = int(np.sqrt(size))
        self.num_vars = self.N**3
        self.penalty = penalty

    def _get_index(self, r: int, c: int, v: int) -> int:
        """Helper to map (row, col, value) to linear index."""
        return r * self.N**2 + c * self.N + (v - 1)

    def build_qubo(self, grid: np.ndarray) -> np.ndarray:
        """
        Builds the QUBO matrix Q such that minimizing x^T Q x solves Sudoku.
        """
        Q = np.zeros((self.num_vars, self.num_vars))
        
        # 1. Constraint: One digit per cell
        # sum_v x_{i,j,v} = 1  => (sum_v x_{i,j,v} - 1)^2 = sum_v x_{i,j,v}^2 + 2 * sum_{v1<v2} x_{i,j,v1}x_{i,j,v2} - 2 * sum_v x_{i,j,v} + 1
        # Since x^2 = x for binary, term becomes: -sum_v x_{i,j,v} + 2 * sum_{v1<v2} x_{i,j,v1}x_{i,j,v2} + 1
        for i in range(self.N):
            for j in range(self.N):
                for v1 in range(1, self.N + 1):
                    idx1 = self._get_index(i, j, v1)
                    Q[idx1, idx1] -= self.penalty
                    for v2 in range(v1 + 1, self.N + 1):
                        idx2 = self._get_index(i, j, v2)
                        Q[idx1, idx2] += 2 * self.penalty
                        Q[idx2, idx1] += 2 * self.penalty

        # 2. Constraint: One of each digit per row
        for i in range(self.N):
            for v in range(1, self.N + 1):
                for j1 in range(self.N):
                    idx1 = self._get_index(i, j1, v)
                    Q[idx1, idx1] -= self.penalty
                    for j2 in range(j1 + 1, self.N):
                        idx2 = self._get_index(i, j2, v)
                        Q[idx1, idx2] += 2 * self.penalty
                        Q[idx2, idx1] += 2 * self.penalty

        # 3. Constraint: One of each digit per column
        for j in range(self.N):
            for v in range(1, self.N + 1):
                for i1 in range(self.N):
                    idx1 = self._get_index(i1, j, v)
                    Q[idx1, idx1] -= self.penalty
                    for i2 in range(i1 + 1, self.N):
                        idx2 = self._get_index(i2, j, v)
                        Q[idx1, idx2] += 2 * self.penalty
                        Q[idx2, idx1] += 2 * self.penalty

        # 4. Constraint: One of each digit per M x M subgrid
        for v in range(1, self.N + 1):
            for row_block in range(self.M):
                for col_block in range(self.M):
                    indices = []
                    for r in range(row_block * self.M, (row_block + 1) * self.M):
                        for c in range(col_block * self.M, (col_block + 1) * self.M):
                            indices.append(self._get_index(r, c, v))
                    
                    for k1 in range(len(indices)):
                        idx1 = indices[k1]
                        Q[idx1, idx1] -= self.penalty
                        for k2 in range(k1 + 1, len(indices)):
                            idx2 = indices[k2]
                            Q[idx1, idx2] += 2 * self.penalty
                            Q[idx2, idx1] += 2 * self.penalty

        # 5. Pre-filled cells (Strong penalty for not matching)
        for i in range(self.N):
            for j in range(self.N):
                if grid[i, j] != 0:
                    v_fixed = int(grid[i, j])
                    idx_fixed = self._get_index(i, j, v_fixed)
                    # We want to encourage x_{i,j,v_fixed} to be 1.
                    # Add a large negative value to the diagonal to push it to 1.
                    Q[idx_fixed, idx_fixed] -= 10 * self.penalty 

        return Q

    def decode_solution(self, solution: np.ndarray) -> np.ndarray:
        """Converts binary solution vector back to Sudoku grid."""
        grid = np.zeros((self.N, self.N), dtype=int)
        for i in range(self.N):
            for j in range(self.N):
                vals = []
                for v in range(1, self.N + 1):
                    if solution[self._get_index(i, j, v)] == 1:
                        vals.append(v)
                if len(vals) == 1:
                    grid[i, j] = vals[0]
                elif len(vals) > 1:
                    grid[i, j] = -1 # Conflict
        return grid

def compute_energy(solution: np.ndarray, Q: np.ndarray) -> float:
    """Computes the energy of a binary solution x^T Q x."""
    return float(solution @ Q @ solution)
