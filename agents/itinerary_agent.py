from datetime import datetime
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from tools.web_search import research_attractions

class ItineraryDay(BaseModel):
    day: int = Field(description="Day number, starting at 1")
    theme: str = Field(description="Short theme for the day, e.g. 'Old town and food'")
    activities: list[str] = Field(description="3-4 activities in the order they should be done")

class Itinerary(BaseModel):
    days: list[ItineraryDay]

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.4) # temperature=0.4 is a moderate setting that allows for some creativity in the itinerary suggestions while still keeping them relevant and practical. A lower temperature would make the model more deterministic, while a higher temperature would introduce more randomness and variety in the output.
itinerary_llm = llm.with_structured_output(Itinerary)

''' a function that takes the current state of the itinerary planning process, 
performs research on attractions based on the user's preferences, and generates a draft itinerary for the trip. 
It calls the research_attractions tool to gather information about attractions in the destination city that match the user's interests.
It then constructs a prompt for the LLM to create a day-by-day itinerary based on the research results, the number of days in the trip, and any specified preferences. 

It returns a dictionary containing the draft itinerary and updates the status to "synthesizing" to indicate that the next step is to refine and finalize the itinerary.
'''
def itinerary_node(state: dict) -> dict:
    start = datetime.strptime(state["start_date"], "%Y-%m-%d")
    end = datetime.strptime(state["end_date"], "%Y-%m-%d")
    num_days = (end - start).days + 1

    interests = ", ".join(state.get("preferences") or [])
    research = research_attractions.invoke({
        "destination": state["destination"],
        "interests": interests,
    })

    research_text = "\n".join(
        f"- {r['title']}: {(r['summary'] or '')[:300]}" for r in research
    )

    prompt = (
        f"Plan a {num_days}-day trip to {state['destination']}.\n"
        f"Traveler interests: {interests or 'general sightseeing'}.\n"
        f"The first and last days are travel days, so keep them lighter.\n\n"
        f"Base your activities on this research:\n{research_text}"
    )

    result = itinerary_llm.invoke(prompt)

    return {
        "itinerary_draft": [d.model_dump() for d in result.days],
        "status": "synthesizing",
    }