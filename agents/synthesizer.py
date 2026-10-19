from datetime import datetime
from utils.common import num_or_inf
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

MAX_SEARCH_ATTEMPTS = 2

SORT_KEYS = {
    "price":    lambda f: (num_or_inf(f.get("price")), num_or_inf(f.get("total_duration_min")), num_or_inf(f.get("stops"))),
    "stops":    lambda f: (num_or_inf(f.get("stops")), num_or_inf(f.get("price")), num_or_inf(f.get("total_duration_min"))),
    "duration": lambda f: (num_or_inf(f.get("total_duration_min")), num_or_inf(f.get("price")), num_or_inf(f.get("stops"))),
}

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

def _hotel_total(hotel: dict, nights: int):
    if hotel.get("total_price") is not None:
        return hotel["total_price"]
    if hotel.get("price_per_night") is not None:
        return hotel["price_per_night"] * nights
    return None

# a function that loops through the flight and hotel options, calculates the total cost for each combination, and filters out those that exceed the budget.
# It returns a list of viable combinations sorted by hotel rating (descending) and total cost (ascending).
def find_viable_combos(flights: list, hotels: list, budget: float, nights: int) -> list:
    combos = []
    for f in flights:
        if f.get("price") is None:
            continue
        for h in hotels:
            hotel_cost = _hotel_total(h, nights)
            if hotel_cost is None:
                continue
            total = f["price"] + hotel_cost
            if total <= budget:
                combos.append({"flight": f, "hotel": h, "total": total})
    combos.sort(key=lambda c: (-(c["hotel"].get("rating") or 0), c["total"]))
    return combos

def find_viable_flights(flights: list, budget: float, sort_by=None) -> list:
    affordable = [f for f in flights if f.get("price") is not None and f["price"] <= budget]
    return [{"flight": f, "hotel": None, "total": f["price"]} for f in sort_flights(affordable, sort_by)]

def sort_flights(flights: list, sort_by) -> list:
    return sorted(flights, key=SORT_KEYS.get(sort_by, SORT_KEYS["price"]))

def synthesizer_node(state: dict) -> dict:
    flights = state.get("flight_options") or []

    if state.get("include_hotels", True):
        start = datetime.strptime(state["start_date"], "%Y-%m-%d")
        end = datetime.strptime(state["end_date"], "%Y-%m-%d")
        combos = find_viable_combos(
            flights, state.get("hotel_options") or [], state["budget"], (end - start).days
        )
    else:
        combos = find_viable_flights(flights, state["budget"], state.get("sort_by"))

    if not combos:
        return {"status": "needs_retry"}

    best = combos[0]
    parts = [
        f"Write a friendly trip recommendation for {state['destination']}.",
        f"Budget: ${state['budget']:.0f}. Chosen total: ${best['total']:.0f}.",
        f"Flight: {best['flight']}",
    ]
    if best["hotel"]:
        parts.append(f"Hotel: {best['hotel']}")
    if state.get("itinerary_draft"):
        parts.append(f"Itinerary: {state['itinerary_draft']}")
    if state.get("travelers_assumed"):
        parts.append("Mention that this assumes 1 adult traveler.")
    parts.append(
        "Summarize only the items listed above, including total cost versus budget. "
        "Use only this data and do not recalculate any prices."
    )

    response = llm.invoke("\n".join(parts))
    return {"final_recommendation": response.content, "status": "done"}

# A function that determines the next step in the itinerary planning process based on the current state.
# If the status is "done", it returns "end" to indicate that the process is complete. 
# If the number of search attempts is less than the maximum allowed, it returns "search" 
# ortherwise, it returns "give_up" 
def route_after_synthesizer(state: dict) -> str:
    if state["status"] == "done":
        return "end"
    if state.get("search_attempts", 0) < MAX_SEARCH_ATTEMPTS:
        return "search"
    return "give_up"