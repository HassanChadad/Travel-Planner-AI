import sys
from graph import build_graph
from agents.orchestrator import describe_airport
from dotenv import load_dotenv

load_dotenv()

MAX_CLARIFICATION_ROUNDS = 3

def run(user_request: str) -> dict:
    app = build_graph()

    state = {
        "user_request": user_request,
        "search_attempts": 0,
        "status": "validating_prompt",
    }

    # invoke runs the whole graph and hands back only the final result. stream yields one chunk each time a node finishes, containing that node’s name and the keys it returned. That lets us print progress, and a retry shows up as [search] appearing twice.
    for chunk in app.stream(state, stream_mode="updates"):
        for node_name, update in chunk.items():
            print(f"[{node_name}] finished and updated state is {update['status']}" )
            state.update(update)

    return state

def plan_trip(request: str) -> dict:
    state = run(request)
    rounds = 0

    while state["status"] == "needs_info" and rounds < MAX_CLARIFICATION_ROUNDS:
        rounds += 1
        print(f"\nI need a few details (round {rounds} of {MAX_CLARIFICATION_ROUNDS}):")
        print(f"{state.get('clarifying_questions', [])}")
        for question in state["clarifying_questions"]:
            answer = input(f"{question} ").strip()
            if answer:
                request += f"\nClarification: {question} Answer: {answer}"
        state = run(request)

    return state

if __name__ == "__main__":
    request = " ".join(sys.argv[1:]) or input("Describe your trip: ")
    final_state = plan_trip(request)

    print("\n" + "=" * 60)
    print(f"Status: {final_state['status']}\n")
    
    if final_state.get("origin_code") and final_state.get("destination_code"):
        print(f"Route: {describe_airport(final_state['origin_code'])} -> {describe_airport(final_state['destination_code'])}\n")

    print(final_state["final_recommendation"])