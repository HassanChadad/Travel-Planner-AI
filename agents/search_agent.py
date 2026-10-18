from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from tools.flight_search import search_flights
from tools.hotel_search import search_hotels

class AirportCodes(BaseModel):
    origin_code: str = Field(description="3-letter IATA code of the main airport for the origin city")
    destination_code: str = Field(description="3-letter IATA code of the main airport for the destination city")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
code_llm = llm.with_structured_output(AirportCodes)

# search_node follows the standard node pattern: read state, do work, return only the keys it changed.
def search_node(state: dict) -> dict:
    search_attempts = state.get("search_attempts", 0)
    # After the first pass the Synthesizer finds no pair within budget, and the router checks 1 < MAX_SEARCH_ATTEMPTS. That’s true, so the graph goes back to Search. The node starts with the counter at 1, so cheapest_first is True, and the tools get the sort_by parameter we added.
    cheapest_first = search_attempts >= 1
    if cheapest_first:
        print(f"No viable flight + hotel combos found within budget. Searching again (attempt={search_attempts}), this time prioritizing cheapest options.")

    codes = code_llm.invoke(
        f"Give the main IATA airport codes for origin '{state['origin']}' "
        f"and destination '{state['destination']}'."
    )

    print(f"Airport codes: {codes}")

    # The tools are called with .invoke({...}) because the @tool decorator turned them into LangChain tools that take a dict of arguments.
    flights = search_flights.invoke({
        "departure_id": codes.origin_code,
        "arrival_id": codes.destination_code,
        "outbound_date": state["start_date"],
        "return_date": state["end_date"],
        "cheapest_first": cheapest_first,
    })

    hotels = search_hotels.invoke({
        "destination": state["destination"],
        "check_in_date": state["start_date"],
        "check_out_date": state["end_date"],
        "cheapest_first": cheapest_first,
    })

    return {
        "flight_options": flights,
        "hotel_options": hotels,
        "search_attempts": state.get("search_attempts", 0) + 1,
        "status": "itinerary",
    }