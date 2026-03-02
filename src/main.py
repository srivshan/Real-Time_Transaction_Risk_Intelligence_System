import torch
from fraud_agent_graph import build_graph
from gat_sampling import GATNet 
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = torch.load("data/elliptic_graph.pt").to(device)


model = GATNet(
    in_channels=data.num_node_features,
    hidden_channels=64,
    dropout=0.3,
    heads=4
).to(device)


model.load_state_dict(
    torch.load("data/best_gat_sampling.pt", map_location=device)
)

model.eval()


with torch.no_grad():
    global_scores = torch.sigmoid(
        model(data.x, data.edge_index)
    ).cpu().numpy()





review_capacity = 0.02  
capacity_threshold = np.percentile(
    global_scores,
    100 * (1 - review_capacity)
)
agent = build_graph(
    model,
    data,
    global_scores,
    capacity_threshold
)


result = agent.invoke({"node_id": 12345})

print("\n===== AGENT OUTPUT =====")
for k, v in result.items():
    print(f"{k}: {v}")