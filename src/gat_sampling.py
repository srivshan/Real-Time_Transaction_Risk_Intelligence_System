import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GATNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, dropout, heads):
        super().__init__()
        
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=1, concat=False, dropout=dropout)
        self.lin = nn.Linear(hidden_channels, 1)
        self.dropout = dropout
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = self.lin(x)
        return x.squeeze()



if __name__ == "__main__":

    import random
    import numpy as np
    from torch_geometric.loader import NeighborLoader
    from sklearn.metrics import roc_auc_score, average_precision_score

   
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

   
    hidden_channels = 64
    dropout_rate = 0.3
    weight_decay = 1e-4
    lr = 0.0015
    heads = 4
    batch_size = 1024
    num_neighbors = [20, 15, 10]
    patience = 40
    max_epochs = 200

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.load("data/elliptic_graph.pt").to(device)

    valid_mask = data.y != -1
    data.train_mask = data.train_mask & valid_mask
    data.val_mask = data.val_mask & valid_mask
    data.test_mask = data.test_mask & valid_mask
    data.y = data.y.clamp(min=0)

    train_loader = NeighborLoader(data, num_neighbors=num_neighbors, batch_size=batch_size, input_nodes=data.train_mask, num_workers=0)
    val_loader = NeighborLoader(data, num_neighbors=num_neighbors, batch_size=batch_size, input_nodes=data.val_mask, num_workers=0)
    test_loader = NeighborLoader(data, num_neighbors=num_neighbors, batch_size=batch_size, input_nodes=data.test_mask, num_workers=0)


    model = GATNet(data.num_node_features, hidden_channels, dropout_rate, heads).to(device)

    train_labels = data.y[data.train_mask]
    pos_weight = (len(train_labels) - train_labels.sum()) / train_labels.sum()

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

 
    best_val_pr = 0
    patience_counter = 0

    for epoch in range(max_epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)
            loss = criterion(out[:batch.batch_size], batch.y[:batch.batch_size].float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch} | Loss {total_loss:.4f}")
