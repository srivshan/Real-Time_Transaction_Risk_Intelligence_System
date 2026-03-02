from langgraph.graph import StateGraph, END

from src.tools.risk_scoring_tool import make_risk_scoring_tool
from src.tools.percentile_tool import make_percentile_tool
from src.tools.graph_stats_tool import make_graph_stats_tool

from src.policy_engine import policy_engine
from src.explanation_agent import explanation_agent


def build_graph(model, data, global_scores, capacity_threshold):

    graph = StateGraph(dict)

    risk_tool = make_risk_scoring_tool(model, data)
    percentile_tool = make_percentile_tool(global_scores)
    graph_stats_tool = make_graph_stats_tool(data)
    policy_node = policy_engine(capacity_threshold)

    graph.add_node("risk_scoring", risk_tool)
    graph.add_node("percentile", percentile_tool)
    graph.add_node("graph_stats", graph_stats_tool)
    graph.add_node("policy", policy_node)
    graph.add_node("explanation", explanation_agent)

    graph.set_entry_point("risk_scoring")

    graph.add_edge("risk_scoring", "percentile")
    graph.add_edge("percentile", "graph_stats")
    graph.add_edge("graph_stats", "policy")
    graph.add_edge("policy", "explanation")
    graph.add_edge("explanation", END)

    return graph.compile()