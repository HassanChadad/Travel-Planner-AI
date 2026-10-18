from dotenv import load_dotenv
load_dotenv()
from agents.orchestrator import orchestrator_node

dest_q = "Which one do you mean for your destination?"
cases = {
    "no dates given": (
        "Plan a trip to Tripoli\n"
        f"Clarification: {dest_q} Answer: lebanon\n"
        "Clarification: Which city are you departing from? Answer: Chantilly\n"
        "Clarification: Is this a one-way or round-trip flight? Answer: round trip\n"
        "Clarification: What is your total budget in USD? Answer: 2000"
    ),
    "nonsense city": "Flying from C to Tripoli, Lebanon on December 10, 2026, returning January 3, 2027. Budget 2000.",
    "all given": "Flying from Chantilly to Tripoli, Lebanon on December 10, 2026, returning January 3, 2027. Budget 2000.",
}

for name, text in cases.items():
    r = orchestrator_node({"user_request": text})
    print(name)
    print("  ", {k: r.get(k) for k in ("origin", "destination", "start_date", "end_date", "origin_code", "destination_code")})
    for q in r["clarifying_questions"]:
        print("   ?", q)
    print()