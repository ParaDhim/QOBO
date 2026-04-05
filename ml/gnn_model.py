import torch
import torch.nn as nn
import torch.nn.functional as F

class MessagePassingLayer(nn.Module):
    """
    Custom Message Passing Layer for GNN.
    m_i = sum_{j in N(i)}  W_msg * concat(h_i, h_j, e_ij)
    h_i' = W_update * concat(h_i, m_i)
    """
    def __init__(self, node_in_dim: int, edge_in_dim: int, out_dim: int):
        super(MessagePassingLayer, self).__init__()
        self.msg_mlp = nn.Sequential(
            nn.Linear(2 * node_in_dim + edge_in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(node_in_dim + out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        row, col = edge_index
        # Get source and target node features
        x_src = x[row]
        x_target = x[col]
        
        # Compute messages
        msg_input = torch.cat([x_src, x_target, edge_attr], dim=-1)
        messages = self.msg_mlp(msg_input)
        
        # Aggregate messages (sum)
        num_nodes = x.size(0)
        aggr_msg = torch.zeros((num_nodes, messages.size(-1)), device=x.device)
        aggr_msg.index_add_(0, col, messages) # Scatter sum
        
        # Update node features
        update_input = torch.cat([x, aggr_msg], dim=-1)
        return self.update_mlp(update_input)

class QUBO_GNN(nn.Module):
    """
    Graph Neural Network to predict QUBO solution probabilities.
    """
    def __init__(self, node_features: int = 2, edge_features: int = 1, hidden_dim: int = 32):
        super(QUBO_GNN, self).__init__()
        self.conv1 = MessagePassingLayer(node_features, edge_features, hidden_dim)
        self.conv2 = MessagePassingLayer(hidden_dim, edge_features, hidden_dim)
        self.conv3 = MessagePassingLayer(hidden_dim, edge_features, hidden_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x, edge_index, edge_attr)
        h = F.relu(h)
        h = self.conv2(h, edge_index, edge_attr)
        h = F.relu(h)
        h = self.conv3(h, edge_index, edge_attr)
        h = F.relu(h)
        
        return self.classifier(h).squeeze(-1)

def get_graph_data_from_qubo(Q: torch.Tensor):
    """
    Converts QUBO matrix to graph format for GNN.
    - Nodes: variables
    - Node Features: [diagonal element, degree]
    - Edges: non-zero off-diagonals
    - Edge Features: [Q_ij]
    """
    N = Q.size(0)
    diag = torch.diag(Q)
    
    # Off-diagonal elements
    off_diag = Q.clone()
    off_diag.fill_diagonal_(0)
    
    # Find edges
    edge_index = torch.nonzero(off_diag).t()
    edge_attr = off_diag[edge_index[0], edge_index[1]].unsqueeze(-1)
    
    # Node features: [diag, degree]
    degree = torch.count_nonzero(off_diag, dim=1).float().unsqueeze(-1)
    node_features = torch.cat([diag.unsqueeze(-1), degree], dim=-1)
    
    return node_features, edge_index, edge_attr
