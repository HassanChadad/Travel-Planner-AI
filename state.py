from typing import TypedDict, Optional, Literal

class TravelPlannerState(TypedDict):
    # input
    user_request: str
    destination: Optional[str]
    destination_code: Optional[str]
    origin: Optional[str]
    origin_code: Optional[str]
    start_date: Optional[str]
    end_date: Optional[str]
    budget: Optional[float]
    preferences: Optional[list[str]]

    # trip details
    trip_type: Optional[Literal["round_trip", "one_way"]]
    travelers: Optional[list[dict]] # {"age": 8, "is_minor": True}
    travelers_assumed: Optional[bool]
    include_hotels: Optional[bool]
    include_activities: Optional[bool]
    sort_by: Optional[Literal["price", "stops", "duration"]]
    clarifying_questions: Optional[list[str]]

    # guardrail
    guardrail_passed: Optional[bool]
    rejection_reason: Optional[str]

    # search results
    flight_options: Optional[list[dict]]
    hotel_options: Optional[list[dict]]

    # itinerary
    itinerary_draft: Optional[list[dict]]

    # synthesis / control flow
    final_recommendation: Optional[str]
    search_attempts: int # incremented by the search node each time it runs
    status: Literal["validating_prompt", "planning", "searching", "itinerary", "synthesizing", "needs_retry", "needs_info", "failed", "rejecting", "rejected", "done"] # these are not used in graph.py, but they are used in main.py to determine the final output. The graph.py file uses the status to determine the next node to run, but it doesn't use the specific values of the status. The main.py file uses the status to determine what to print to the user at the end of the run.