import argparse
import random

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score

from model import ActiveSiteGNN

def split_dataset(graphs, val_frac=0.15, test_frac=0.15, seed=42):
    random.Random(seed).shuffle(graphs)
    n = len(graphs)
    n_val = int(n * val_frac)
    n_test = int(n * test_frac)
    
    val = graphs[:n_val]
    test = graphs[n_val:n_val + n_test]
    train = graphs[n_val + n_test:]
    return train, val, test

def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index)
            preds = out.argmax(dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(batch.y.cpu())
            
    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    return precision, recall, f1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_file", default="data/dataset.py")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_channels", type=int, default=64)
    parser.add_argument("--active-site-weight", type=float, default=10.0, help="Loss weight for the rare active-site class")
    parser.add_argument("--model_out", default="data/model.pt")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    graphs = torch.load(args.dataset_file, weights_only=False)
    print(f"Loaded {len(graphs)} graphs")
    
    train_graphs, val_graphs, test_graphs = split_dataset(graphs)
    print(f"Train/val/test sizes: {len(train_graphs)}/{len(val_graphs)}/{len(test_graphs)}")
    
    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=args.batch_size)
    test_loader = DataLoader(test_graphs, batch_size=args.batch_size)
    
    in_channels = graphs[0].x.shape[1]
    model = ActiveSiteGNN(in_channels=in_channels, hidden_channels=args.hidden_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    class_weights = torch.tensor([1.0, args.active_site_weight]).to(device)
    
    best_val_f1 = 0.0
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)
            loss = F.cross_entropy(out, batch.y, weight=class_weights)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        val_precision, val_recall, val_f1 = evaluate(model, val_loader, device)
        print(f"Epoch {epoch + 1}/{args.epochs} | loss={total_loss:.4f} | "
              f"val_precision={val_precision:.3f} val_recall={val_recall:.3f} val_f1={val_f1:.3f}")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), args.model_out)
            
    print(f"\nBest val F1: {best_val_f1:.3f} - model saved to {args.model_out}")
    
    model.load_state_dict(torch.load(args.model_out, weights_only=True))
    test_precision, test_recall, test_f1 = evaluate(model, test_loader, device)
    print(f"Test set: precision={test_precision:.3f} recall={test_recall:.3f} f1={test_f1:.3f}")
    
if __name__ == "__main__":
    main()