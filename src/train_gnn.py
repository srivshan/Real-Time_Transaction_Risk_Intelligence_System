import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from sklearn.metrics import roc_auc_score, average_precision_score



hidden_channels = 64
dropout_rate = 0.6
weight_decay = 1e-3
lr = 0.005

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = torch.load("data/elliptic_graph.pt")
data = data.to(device)


valid_mask = data.y != -1
data.train_mask = data.train_mask & valid_mask
data.val_mask = data.val_mask & valid_mask
data.test_mask = data.test_mask & valid_mask

data.y = data.y.clamp(min=0)

print("Train label values:", torch.unique(data.y[data.train_mask]))


class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.lin = nn.Linear(hidden_channels, 1)
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=dropout_rate, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        x = self.lin(x)
        return x.squeeze()

model = GraphSAGE(
    in_channels=data.num_node_features,
    hidden_channels=hidden_channels
).to(device)



train_labels = data.y[data.train_mask]
pos_weight = (len(train_labels) - train_labels.sum()) / train_labels.sum()

criterion = nn.BCEWithLogitsLoss(
    pos_weight=pos_weight.to(device)
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=lr,
    weight_decay=weight_decay
)

best_val_pr = 0
best_epoch = 0
patience = 40
patience_counter = 0

def evaluate(mask):
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        probs = torch.sigmoid(logits)
        
        y_true = data.y[mask].cpu()
        y_pred = probs[mask].cpu()
        
        roc = roc_auc_score(y_true, y_pred)
        pr = average_precision_score(y_true, y_pred)
        
    return roc, pr

epochs = 200

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    
    logits = model(data.x, data.edge_index)
    
    loss = criterion(
        logits[data.train_mask],
        data.y[data.train_mask].float()
    )
    
    loss.backward()
    optimizer.step()
    
   
    val_roc, val_pr = evaluate(data.val_mask)
    
    print(f"Epoch {epoch:03d} | Loss {loss:.4f} | Val ROC {val_roc:.4f} | Val PR {val_pr:.4f}")
  
    if val_pr > best_val_pr:
        best_val_pr = val_pr
        best_epoch = epoch
        patience_counter = 0
        torch.save(model.state_dict(), "data/best_graphsage.pt")
    else:
        patience_counter += 1
        
    if patience_counter >= patience:
        print(f"\nEarly stopping at epoch {epoch}")
        break

model.load_state_dict(torch.load("data/best_graphsage.pt"))

test_roc, test_pr = evaluate(data.test_mask)

print("\n===== FINAL TEST (Best Model) =====")
print("Best Epoch:", best_epoch)
print("Best Val PR:", best_val_pr)
print("Test ROC:", test_roc)
print("Test PR:", test_pr)