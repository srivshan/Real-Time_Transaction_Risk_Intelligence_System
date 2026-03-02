import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)


llm = genai.GenerativeModel("gemini-2.5-flash")


def explanation_agent(state: dict):

    node_id = state["node_id"]
    risk_score = state["risk_score"]
    percentile = state["percentile"]
    risk_tier = state["risk_tier"]
    action = state["action"]

    
    prompt = (
        f"Explain briefly (max 2 sentences) why this transaction is high risk.\n\n"
        f"Node: {node_id}\n"
        f"Risk Score: {risk_score:.4f}\n"
        f"Percentile: {percentile:.4f}\n"
        f"Tier: {risk_tier}\n"
        f"Action: {action}"
    )

    response = llm.generate_content(
        prompt,
        generation_config={
            "temperature": 0.2,
            "max_output_tokens": 80,
            "top_p": 0.9,
        }
    )

    state["explanation"] = response.text.strip()

    return state