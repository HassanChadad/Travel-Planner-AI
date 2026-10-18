from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_tavily import TavilySearch

load_dotenv()

_tavily = TavilySearch(max_results=5)

@tool
def research_attractions(destination: str, interests: str = "") -> list[dict]:
    """Research things to do, attractions, and local tips for a destination."""
    query = f"best things to do in {destination}"
    if interests:
        query += f" for {interests}"

    response = _tavily.invoke({"query": query})

    if not isinstance(response, dict):
        return []

    return [
        {
            "title": r.get("title"),
            "url": r.get("url"),
            "summary": r.get("content"),
        }
        for r in response.get("results", [])
    ]