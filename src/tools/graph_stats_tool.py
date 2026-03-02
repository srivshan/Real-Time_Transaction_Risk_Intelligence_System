def make_graph_stats_tool(data):
    def graph_stats_tool(state):

        node_id = state["node_id"]
        mask = data.edge_index[0] == node_id
        neighbors = data.edge_index[1][mask]

        degree = int(neighbors.numel())

        if degree == 0:
            fraud_neighbors = 0
        else:
            neighbor_labels = data.y[neighbors]
            fraud_neighbors = int((neighbor_labels == 1).sum().item())

        return {
            **state,
            "degree": degree,
            "fraud_neighbors": fraud_neighbors
        }

    return graph_stats_tool