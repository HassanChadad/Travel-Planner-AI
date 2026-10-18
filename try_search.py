from agents.search_agent import search_node

state = {
    "origin": "Chantilly, VA",
    "destination": "Beirut Lebanon",
    "start_date": "2026-12-10",
    "end_date": "2026-12-17",
    "search_attempts": 0,
}

result = search_node(state)
print(result["search_attempts"], result["status"])
print(len(result["flight_options"]), "flights")
print(len(result["hotel_options"]), "hotels")