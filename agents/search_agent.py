from tools.flight_search import search_flights
from tools.hotel_search import search_hotels

# search_node follows the standard node pattern: read state, do work, return only the keys it changed.
def search_node(state: dict) -> dict:
    one_way = state.get("trip_type") == "one_way"
    search_attempts = state.get("search_attempts", 0)
    # After the first pass the Synthesizer finds no pair within budget, and the router checks 1 < MAX_SEARCH_ATTEMPTS. That’s true, so the graph goes back to Search. The node starts with the counter at 1, so cheapest_first is True, and the tools get the sort_by parameter we added.
    cheapest_first = search_attempts >= 1
    if cheapest_first:
        print(f"No viable flight + hotel combos found within budget. Searching again (attempt={search_attempts}), this time prioritizing cheapest options.")

    # The tools are called with .invoke({...}) because the @tool decorator turned them into LangChain tools that take a dict of arguments.
    flights = search_flights.invoke({
        "departure_id": state["origin_code"],
        "arrival_id": state["destination_code"],
        "outbound_date": state["start_date"],
        "return_date": None if one_way else state["end_date"],
        "one_way": one_way,
        "cheapest_first": cheapest_first,
        "sort_by": state.get("sort_by"),
    })

    if state.get("include_hotels", True):
        hotels = search_hotels.invoke({
            "destination": state["destination"],
            "check_in_date": state["start_date"],
            "check_out_date": state["end_date"],
            "cheapest_first": cheapest_first,
        })
    else:
        hotels = []

    return {
        "flight_options": flights,
        "hotel_options": hotels,
        "search_attempts": state.get("search_attempts", 0) + 1,
        "status": "itinerary",
    }

def route_after_search(state: dict) -> str:
    if state.get("include_activities", True):
        return "itinerary"
    return "synthesizer"