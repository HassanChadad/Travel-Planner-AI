from agents.itinerary_agent import itinerary_node

state = {
    "destination": "Beirut Lebanon",
    "start_date": "2026-12-10",
    "end_date": "2027-01-03",
    "preferences": ["museums", "food"],
}

result = itinerary_node(state)

for day in result["itinerary_draft"]:
    print(f"Day {day['day']}: {day['theme']}")
    for activity in day["activities"]:
        print("  -", activity)