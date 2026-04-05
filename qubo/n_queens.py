import numpy as np
from typing import List, Tuple

class NQueensQUBO:
    """
    Formulates the N-Queens problem as a QUBO.
    Variables: x_{i,j} = 1 if queen at row i, col j.
    Constraints:
    1. One queen per row.
    2. One queen per column.
    3. At most one queen per diagonal.
    """
    def __init__(self, size: int, penalty: float = 10.0):
        self.N = size
        self.num_vars = size * size
        self.penalty = penalty

    def _get_index(self, r: int, c: int) -> int:
        return r * self.N + c

    def build_qubo(self) -> np.ndarray:
        Q = np.zeros((self.num_vars, self.num_vars))
        
        # 1. Constraint: One queen per row
        for i in range(self.N):
            for j1 in range(self.N):
                idx1 = self._get_index(i, j1)
                Q[idx1, idx1] -= self.penalty
                for j2 in range(j1 + 1, self.N):
                    idx2 = self._get_index(i, j2)
                    Q[idx1, idx2] += 2 * self.penalty
                    Q[idx2, idx1] += 2 * self.penalty

        # 2. Constraint: One queen per col
        for j in range(self.N):
            for i1 in range(self.N):
                idx1 = self._get_index(i1, j)
                Q[idx1, idx1] -= self.penalty
                for i2 in range(i1 + 1, self.N):
                    idx2 = self._get_index(i2, j)
                    Q[idx1, idx2] += 2 * self.penalty
                    Q[idx2, idx1] += 2 * self.penalty

        # 3. Constraint: At most one queen per diagonal
        # Diagonals (top-left to bottom-right)
        for k in range(1 - self.N, self.N):
            indices = []
            for i in range(self.N):
                j = i - k
                if 0 <= j < self.N:
                    indices.append(self._get_index(i, j))
            
            for m1 in range(len(indices)):
                for m2 in range(m1 + 1, len(indices)):
                    idx1, idx2 = indices[m1], indices[m2]
                    Q[idx1, idx2] += self.penalty
                    Q[idx2, idx1] += self.penalty

        # 4. Constraint: At most one queen per anti-diagonal
        for k in range(0, 2 * self.N - 1):
            indices = []
            for i in range(self.N):
                j = k - i
                if 0 <= j < self.N:
                    indices.append(self._get_index(i, j))
            
            for m1 in range(len(indices)):
                for m2 in range(m1 + 1, len(indices)):
                    idx1, idx2 = indices[m1], indices[m2]
                    Q[idx1, idx2] += self.penalty
                    Q[idx2, idx1] += self.penalty

        return Q

    def decode_solution(self, solution: np.ndarray) -> np.ndarray:
        board = np.zeros((self.N, self.N), dtype=int)
        for i in range(self.N):
            for j in range(self.N):
                if solution[self._get_index(i, j)] == 1:
                    board[i, j] = 1
        return board
