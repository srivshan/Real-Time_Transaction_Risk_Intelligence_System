

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import torch
import numpy as np
import time

from src.fraud_agent_graph import build_graph
from src.gat_sampling import GATNet


app = FastAPI(title="Graph Fraud Risk Intelligence API")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = torch.load("data/elliptic_graph.pt", map_location=device)
data = data.to(device)

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
    logits = model(data.x, data.edge_index)
    global_scores = torch.sigmoid(logits).cpu().numpy()


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


class BatchRequest(BaseModel):
    node_ids: List[int]


def validate_node_id(node_id: int):
    if node_id < 0 or node_id >= data.num_nodes:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid node_id: {node_id}"
        )




@app.get("/analyze/{node_id}")
def analyze_node(node_id: int):

    validate_node_id(node_id)

    start = time.perf_counter()

    result = agent.invoke({"node_id": node_id})

    end = time.perf_counter()
    latency_ms = (end - start) * 1000

    result["latency_ms"] = round(latency_ms, 2)

    return result




@app.post("/analyze_batch")
def analyze_batch(request: BatchRequest):

    batch_start = time.perf_counter()
    results = []

    for node_id in request.node_ids:

        validate_node_id(node_id)

        node_start = time.perf_counter()
        result = agent.invoke({"node_id": node_id})
        node_end = time.perf_counter()

        result["latency_ms"] = round(
            (node_end - node_start) * 1000,
            2
        )

        results.append(result)

    batch_end = time.perf_counter()
    batch_latency_ms = (batch_end - batch_start) * 1000

    return {
        "batch_latency_ms": round(batch_latency_ms, 2),
        "results": results
    }