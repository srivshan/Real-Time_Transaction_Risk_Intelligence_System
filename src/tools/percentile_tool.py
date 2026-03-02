def make_percentile_tool(global_scores):
    def percentile_tool(state):
        score = state["risk_score"]
        percentile = float((global_scores < score).mean())
        return {**state, "percentile": percentile}
    return percentile_tool