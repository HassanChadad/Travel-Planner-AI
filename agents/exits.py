from agents.orchestrator import REQUIRED_FIELDS
'''
Exit     | When it happens                                  | Final status
Success   Synthesizer found a pair within budget	        done
ask_user  Orchestrator couldn’t extract all required fields	needs_info
give_up   Retry limit reached, still nothing within budget	failed
'''
def ask_user_node(state: dict) -> dict:
    missing = [f for f in REQUIRED_FIELDS if not state.get(f)]
    message = (
        "I couldn't plan your trip yet because some details are missing: "
        + ", ".join(missing)
        + ". Please include them in your request and try again."
    )
    return {"final_recommendation": message, "status": "needs_info"}

def give_up_node(state: dict) -> dict:
    budget = state.get("budget")
    attempts = state.get("search_attempts", 0)
    message = (
        f"I searched {attempts} times, including a cheapest-first pass, but couldn't find "
        f"a flight and hotel combination within your ${budget:.0f} budget. "
        "Try raising the budget, changing your dates, or choosing a closer destination."
    )
    return {"final_recommendation": message, "status": "failed"}

REJECTION_MESSAGES = {
    "invalid_input": "Please enter a trip request between 1 and 1000 characters.",
    "off_topic": (
        "I can only help plan trips, such as flights, hotels, and itineraries. For example: "
        "\"I'm flying from San Francisco to Lisbon on December 10 and returning December 14, "
        "with a $3000 budget.\""
    ),
    "check_failed": "I couldn't verify your request right now. Please try again in a moment.",
}
DEFAULT_REJECTION = "I can't help with that request. Please describe a trip you'd like me to plan."

def reject_node(state: dict) -> dict:
    reason = state.get("rejection_reason")
    message = REJECTION_MESSAGES.get(reason, DEFAULT_REJECTION)
    return {"final_recommendation": message, "status": "rejected"}