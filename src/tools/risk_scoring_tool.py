import torch

def make_risk_scoring_tool(model, data):
    def risk_scoring_tool(state):
        node_id = state["node_id"]
        model.eval()
        with torch.no_grad():
            scores = torch.sigmoid(model(data.x, data.edge_index))
            score = scores[node_id].item()
        return {**state, "risk_score": float(score)}
    return risk_scoring_tool