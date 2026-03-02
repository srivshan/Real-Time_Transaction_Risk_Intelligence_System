def policy_engine(capacity_threshold):

    def policy_engine(state):

        risk_score = state["risk_score"]
        percentile = state["percentile"]

       
        if percentile >= 0.99:
            risk_tier = "very_high"
        elif percentile >= 0.95:
            risk_tier = "high"
        elif percentile >= 0.90:
            risk_tier = "medium"
        else:
            risk_tier = "low"

   
        if risk_score >= capacity_threshold:
            action = "manual_review"
            priority = "high"
        else:
            action = "allow"
            priority = "low"

        if percentile >= 0.999:
            action = "block_transaction"
            priority = "critical"

        return {
            **state,
            "risk_tier": risk_tier,
            "action": action,
            "priority": priority
        }

    return policy_engine