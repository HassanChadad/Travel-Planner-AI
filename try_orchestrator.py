from dotenv import load_dotenv
load_dotenv()

from agents.orchestrator import orchestrator_node, missing_fields

requests = [
    "Flying from San Francisco to Lisbon Dec 10 2026, returning Dec 14, budget 3000.",
    "My wife, our 5 year old, our baby and I want to fly one-way from Boston to Rome on Dec 10, fastest flights. Budget 4000.",
    "Plan a trip to Lisbon with minimal stops",
    "Plan a trip to Lisbon with cheapest flights, no hotel, and no activities. Budget 3000.",
]

for text in requests:
    result = orchestrator_node({"user_request": text})
    print(text)
    print({k: result[k] for k in ("trip_type", "travelers", "travelers_assumed",
                                  "include_hotels", "include_activities", "sort_by")})
    print("missing:", missing_fields(result), "\n")