from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# Defines the exact shape we want the LLM's output forced into
# BaseModel is a Pydantic class that allows us to define data models with type validation and serialization. By using BaseModel, we can ensure that the output from the LLM adheres to the specified structure, making it easier to work with and reducing the risk of unexpected output formats.
class ExtractedRequest(BaseModel):
    origin: Optional[str] = Field(description="City the user departs from")
    destination: Optional[str] = Field(description="City or region the user wants to visit")
    start_date: Optional[str] = Field(description="Trip start date, YYYY-MM-DD if known")
    end_date: Optional[str] = Field(description="Trip end date, YYYY-MM-DD if known")
    budget: Optional[float] = Field(description="Total budget in USD, if mentioned")
    preferences: Optional[list[str]] = Field(description="Interests or constraints mentioned")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0) # temperature=0 is for deterministic output, which means the model will produce the same output for the same input every time. This is important for structured output extraction, as we want consistent results.
structured_llm = llm.with_structured_output(ExtractedRequest) # llm.with_structured_output() is a method that wraps the LLM to enforce structured output according to the ExtractedRequest model. This means that when we invoke the LLM, it will return data that conforms to the ExtractedRequest schema, making it easier to work with and reducing the risk of unexpected output formats.

def orchestrator_node(state: dict) -> dict:
    result = structured_llm.invoke(
        f"Extract travel details from this request: {state['user_request']}"
    )
    return {
        "origin": result.origin,
        "destination": result.destination,
        "start_date": result.start_date,
        "end_date": result.end_date,
        "budget": result.budget,
        "preferences": result.preferences,
        "status": "searching",
    }

REQUIRED_FIELDS = ["origin", "destination", "start_date", "end_date", "budget"]

def route_after_orchestrator(state: dict) -> str:
    missing = [field for field in REQUIRED_FIELDS if not state.get(field)]
    if missing:
        return "ask_user"
    return "search"