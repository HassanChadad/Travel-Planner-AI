import os
from dotenv import load_dotenv
from serpapi import GoogleSearch
from langchain_core.tools import tool

load_dotenv()

# @tool is a decorator that marks the function as a tool that can be used by the agent.
# the function comment is used as the tool description, which is used by langchain to generate prompts for the agent. The comment should be concise and describe what the tool does, so that the agent can understand when to use it.
@tool
def search_flights(departure_id: str, arrival_id: str, outbound_date: str, return_date: str, cheapest_first: bool = False) -> list[dict]:
    """Search round-trip flights. Use 3-letter airport codes (e.g. SFO, LIS) and YYYY-MM-DD dates."""
    params = {
        "engine": "google_flights",
        "departure_id": departure_id,
        "arrival_id": arrival_id,
        "outbound_date": outbound_date,
        "return_date": return_date,
        "currency": "USD",
        "hl": "en",
        "api_key": os.getenv("SERPAPI_API_KEY"),
    }

    if cheapest_first:
        params["sort_by"] = 2 # 2 = sort by price. It’s a code defined by SerpAPI. For Google Flights, the sort_by options are 1 for top flights (the default), 2 for price, 3 for departure time, 4 for arrival time, 5 for duration, and 6 for emissions.

    results = GoogleSearch(params).get_dict()

    if "error" in results:
        return []

    flights = results.get("best_flights", []) + results.get("other_flights", [])
    options = []
    for f in flights[:5]:
        legs = f["flights"]
        options.append({
            "price": f.get("price"),
            "total_duration_min": f.get("total_duration"),
            "airline": legs[0].get("airline"),
            "departure_time": legs[0]["departure_airport"].get("time"),
            "arrival_time": legs[-1]["arrival_airport"].get("time"),
            "stops": len(legs) - 1,
        })
    return options