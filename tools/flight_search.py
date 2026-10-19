import os
from dotenv import load_dotenv
from serpapi import GoogleSearch
from langchain_core.tools import tool
from typing import Optional

load_dotenv()

# SerpAPI Google Flights sort_by values (from SerpAPI's docs): 2 = price, 5 = duration
SORT_BY_CODES = {"price": 2, "duration": 5}
MAX_FLIGHTS = 10
FLIGHT_TYPE_ONE_WAY = 2  # SerpAPI Google Flights `type`: 1 = round trip (default), 2 = one way

# @tool is a decorator that marks the function as a tool that can be used by the agent.
# the function comment is used as the tool description, which is used by langchain to generate prompts for the agent. The comment should be concise and describe what the tool does, so that the agent can understand when to use it.
@tool
def search_flights(departure_id: str, arrival_id: str, outbound_date: str,
                   return_date: Optional[str] = None, cheapest_first: bool = False,
                   sort_by: Optional[str] = None, one_way: bool = False) -> list[dict]:
    """Search flights, round-trip by default or one-way when one_way is true. Use 3-letter airport codes and YYYY-MM-DD dates."""
    params = {
        "engine": "google_flights",
        "departure_id": departure_id,
        "arrival_id": arrival_id,
        "outbound_date": outbound_date,
        "currency": "USD",
        "hl": "en",
        "api_key": os.getenv("SERPAPI_API_KEY"),
    }

    if one_way:
        params["type"] = FLIGHT_TYPE_ONE_WAY
    else:
        params["return_date"] = return_date

    server_sort = "price" if cheapest_first else sort_by
    if server_sort in SORT_BY_CODES:
        params["sort_by"] = SORT_BY_CODES[server_sort]

    if cheapest_first:
        params["sort_by"] = 2 # 2 = sort by price. It’s a code defined by SerpAPI. For Google Flights, the sort_by options are 1 for top flights (the default), 2 for price, 3 for departure time, 4 for arrival time, 5 for duration, and 6 for emissions.

    results = GoogleSearch(params).get_dict()

    if "error" in results:
        return []

    flights = results.get("best_flights", []) + results.get("other_flights", [])
    options = []
    for f in flights[:MAX_FLIGHTS]:
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