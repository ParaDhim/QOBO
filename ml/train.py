import torch
import torch.nn as nn
import torch.optim as optim
from ml.gnn_model import QUBO_GNN
from ml.dataset import QUBODataset
import os
from tqdm import tqdm

def train_gnn(problem_type: str = "n_queens", size: int = 16, epochs: int = 150, lr: float = 0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting training for {problem_type} on {device} (Diverse sizes up to {size})...")
    
    # Generate data
    datasets = []
    # Include N=20 as requested by user, and intermediate sizes
    for s in [4, 8, 12, 16, 20]:
        if s > 0:
            ds = QUBODataset(num_samples=40, problem_type=problem_type, size=s)
            datasets.extend(ds.samples)
    
    if len(datasets) == 0:
        print("Dataset empty. No valid solutions found.")
        return None

    model = QUBO_GNN(node_features=2, edge_features=1, hidden_dim=64).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    model.train()
    pbar = tqdm(range(epochs), desc="Training GNN")
    for epoch in pbar:
        total_loss = 0
        for data in datasets:
            optimizer.zero_grad()
            x, edge_idx, edge_attr, y = data['x'].to(device), data['edge_index'].to(device), data['edge_attr'].to(device), data['y'].to(device)
            
            out = model(x, edge_idx, edge_attr)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(datasets)
        pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

    # Save model
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), f"models/gnn_{problem_type}.pth")
    print(f"Model saved to models/gnn_{problem_type}.pth")
    return model

if __name__ == "__main__":
    train_gnn()
