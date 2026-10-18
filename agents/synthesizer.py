from datetime import datetime
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

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

def synthesizer_node(state: dict) -> dict:
    start = datetime.strptime(state["start_date"], "%Y-%m-%d")
    end = datetime.strptime(state["end_date"], "%Y-%m-%d")
    nights = (end - start).days

    combos = find_viable_combos(
        state.get("flight_options") or [],
        state.get("hotel_options") or [],
        state["budget"],
        nights,
    )

    if not combos:
        return {"status": "needs_retry"}

    best = combos[0]
    prompt = (
        f"Write a friendly trip recommendation for {state['destination']}.\n"
        f"Budget: ${state['budget']:.0f}. Chosen flight + hotel total: ${best['total']:.0f}.\n"
        f"Flight: {best['flight']}\n"
        f"Hotel: {best['hotel']}\n"
        f"Itinerary: {state.get('itinerary_draft')}\n\n"
        "Summarize the flight, hotel, and total cost versus budget, then give a short "
        "day-by-day overview. Use only the data above and do not recalculate any prices."
    )
    response = llm.invoke(prompt)

    return {"final_recommendation": response.content, "status": "done"}

MAX_SEARCH_ATTEMPTS = 2

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