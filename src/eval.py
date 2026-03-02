import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv
from torch_geometric.loader import NeighborLoader
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


data = torch.load("data/elliptic_graph.pt")

valid_mask = data.y != -1
data.train_mask &= valid_mask
data.val_mask &= valid_mask
data.test_mask &= valid_mask
data.y = data.y.clamp(min=0)

data = data.to(device)



test_loader = NeighborLoader(
    data,
    num_neighbors=[20, 15, 10],
    batch_size=1024,
    input_nodes=data.test_mask,
    num_workers=0
)

class GraphSAGE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, 64)
        self.conv2 = SAGEConv(64, 64)
        self.lin = nn.Linear(64, 1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.lin(x)
        return x.squeeze()


class GATNet(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = GATConv(in_channels, 64, heads=4, dropout=0.3)
        self.conv2 = GATConv(64 * 4, 64, heads=1, concat=False, dropout=0.3)
        self.lin = nn.Linear(64, 1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = self.lin(x)
        return x.squeeze()


def evaluate_metrics(y_true, y_scores, top_percent=0.05):

    y_true = y_true.cpu().numpy()
    y_scores = y_scores.cpu().numpy()

    roc = roc_auc_score(y_true, y_scores)
    pr = average_precision_score(y_true, y_scores)

    n = len(y_scores)
    k = int(n * top_percent)

    sorted_idx = np.argsort(-y_scores)
    top_k_idx = sorted_idx[:k]

   
    precision_at_k = y_true[top_k_idx].mean()

 
    total_fraud = y_true.sum()
    fraud_captured = y_true[top_k_idx].sum()
    recall_at_k = fraud_captured / total_fraud

    baseline_rate = y_true.mean()
    lift_at_k = precision_at_k / baseline_rate

    return roc, pr, precision_at_k, recall_at_k, lift_at_k


def evaluate_model(model):
    model.eval()
    y_true_all = []
    y_pred_all = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index)
            probs = torch.sigmoid(out[:batch.batch_size])

            y_true_all.append(batch.y[:batch.batch_size].cpu())
            y_pred_all.append(probs.cpu())

    y_true = torch.cat(y_true_all)
    y_scores = torch.cat(y_pred_all)

    roc, pr, p5, r5, lift5 = evaluate_metrics(y_true, y_scores, 0.05)
    _, _, p1, r1, lift1 = evaluate_metrics(y_true, y_scores, 0.01)

    return roc, pr, p1, r1, lift1, p5, r5, lift5



graphsage_model = GraphSAGE(data.num_node_features).to(device)
graphsage_model.load_state_dict(torch.load("data/best_graphsage.pt", map_location=device))

roc_sage, pr_sage, p1_sage, r1_sage, lift1_sage, p5_sage, r5_sage, lift5_sage = evaluate_model(graphsage_model)


gat_model = GATNet(data.num_node_features).to(device)
gat_model.load_state_dict(torch.load("data/best_gat_sampling.pt", map_location=device))

roc_gat, pr_gat, p1_gat, r1_gat, lift1_gat, p5_gat, r5_gat, lift5_gat = evaluate_model(gat_model)



print("\n===== MODEL COMPARISON =====")
print(f"{'Metric':<20} {'GraphSAGE':<12} {'GAT':<12}")
print("-" * 55)

print(f"{'ROC-AUC':<20} {roc_sage:<12.4f} {roc_gat:<12.4f}")
print(f"{'PR-AUC':<20} {pr_sage:<12.4f} {pr_gat:<12.4f}")

print(f"{'Precision@1%':<20} {p1_sage:<12.4f} {p1_gat:<12.4f}")
print(f"{'Recall@1%':<20} {r1_sage:<12.4f} {r1_gat:<12.4f}")
print(f"{'Lift@1%':<20} {lift1_sage:<12.2f} {lift1_gat:<12.2f}")

print(f"{'Precision@5%':<20} {p5_sage:<12.4f} {p5_gat:<12.4f}")
print(f"{'Recall@5%':<20} {r5_sage:<12.4f} {r5_gat:<12.4f}")
print(f"{'Lift@5%':<20} {lift5_sage:<12.2f} {lift5_gat:<12.2f}")