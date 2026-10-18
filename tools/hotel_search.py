import os
from dotenv import load_dotenv
from serpapi import GoogleSearch
from langchain_core.tools import tool

load_dotenv()

@tool
def search_hotels(destination: str, check_in_date: str, check_out_date: str, cheapest_first: bool = False) -> list[dict]:
    """Search hotels in a destination. Dates are YYYY-MM-DD."""
    params = {
        "engine": "google_hotels",
        "q": f"hotels in {destination}",
        "check_in_date": check_in_date,
        "check_out_date": check_out_date,
        "adults": 1,
        "currency": "USD",
        "hl": "en",
        "api_key": os.getenv("SERPAPI_API_KEY"),
    }
    if cheapest_first:
        params["sort_by"] = 3  # 3 = lowest price. (it's a code defined by SerpAPI. For Google Hotels, the sort_by options are 1 for top hotels (the default), 2 for best rating, 3 for lowest price, and 4 for highest price.)
    
    results = GoogleSearch(params).get_dict()

    if "error" in results:
        return []

    options = []
    for h in results.get("properties", [])[:5]:
        options.append({
            "name": h.get("name"),
            "rating": h.get("overall_rating"),
            "price_per_night": h.get("rate_per_night", {}).get("extracted_lowest"),
            "total_price": h.get("total_rate", {}).get("extracted_lowest"),
            "hotel_class": h.get("hotel_class"),
            "link": h.get("link"),
        })
    return options