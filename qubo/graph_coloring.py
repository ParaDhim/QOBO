import numpy as np
from typing import List, Tuple

class GraphColoringQUBO:
    """
    Formulates the Graph Coloring problem as a QUBO.
    Variables: x_{v,c} = 1 if vertex v has color c.
    Constraints:
    1. Each vertex must have exactly one color.
    2. Adjacent vertices must have different colors.
    """
    def __init__(self, num_nodes: int, num_colors: int, penalty: float = 5.0):
        self.num_nodes = num_nodes
        self.num_colors = num_colors
        self.num_vars = num_nodes * num_colors
        self.penalty = penalty

    def _get_index(self, v: int, c: int) -> int:
        return v * self.num_colors + c

    def build_qubo(self, edges: List[Tuple[int, int]]) -> np.ndarray:
        Q = np.zeros((self.num_vars, self.num_vars))
        
        # 1. Constraint: Each vertex exactly one color
        # (sum_c x_{v,c} - 1)^2 = sum_c x_{v,c}^2 + 2*sum_{c1<c2} x_{v,c1}x_{v,c2} - 2*sum_c x_{v,c} + 1
        for v in range(self.num_nodes):
            for c1 in range(self.num_colors):
                idx1 = self._get_index(v, c1)
                Q[idx1, idx1] -= self.penalty
                for c2 in range(c1 + 1, self.num_colors):
                    idx2 = self._get_index(v, c2)
                    Q[idx1, idx2] += 2 * self.penalty
                    Q[idx2, idx1] += 2 * self.penalty

        # 2. Constraint: Adjacent vertices different colors
        # (x_{u,c} + x_{v,c} <= 1) => x_{u,c} * x_{v,c} = 0
        for u, v in edges:
            for c in range(self.num_colors):
                idx_u = self._get_index(u, c)
                idx_v = self._get_index(v, c)
                Q[idx_u, idx_v] += self.penalty
                Q[idx_v, idx_u] += self.penalty

        return Q

    def decode_solution(self, solution: np.ndarray) -> List[int]:
        coloring = [-1] * self.num_nodes
        for v in range(self.num_nodes):
            colors = [c for c in range(self.num_colors) if solution[self._get_index(v, c)] == 1]
            if len(colors) == 1:
                coloring[v] = colors[0]
            elif len(colors) > 1:
                coloring[v] = -2 # Multi-color conflict
        return coloring
