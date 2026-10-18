from dotenv import load_dotenv
load_dotenv()

from agents.orchestrator import orchestrator_node

requests = [
    "Flying from Richmond to Tripoli on Dec 10, 2026, returning Jan 3, 2027. Budget 2000.",
    "Flying from Richmond, Virginia to Tripoli, Lebanon on Dec 10, 2026, back Jan 3, 2027. Budget 2000.",
    "Flying from Richmond to Paris on Dec 10, 2026. Budget 2000.",
    "Flying from Richmond to Tripoli on Dec 10, 2026, returning Jan 3, 2027. Budget 2000.\n"
    "Clarification: Which Richmond do you mean for your departure city? Answer: Virginia\n"
    "Clarification: Which Tripoli do you mean for your destination? Answer: Lebanon",
]

for text in requests:
    result = orchestrator_node({"user_request": text})
    print(text.splitlines()[0])
    for q in result["clarifying_questions"]:
        print("  ?", q)
    print()