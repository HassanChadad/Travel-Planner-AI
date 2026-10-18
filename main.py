import sys
from dotenv import load_dotenv

load_dotenv()

from graph import build_graph

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

if __name__ == "__main__":
    request = " ".join(sys.argv[1:]) or input("Describe your trip: ")
    final_state = run(request)

    print("\n" + "=" * 60)
    print(f"Status: {final_state['status']}\n")
    print(final_state["final_recommendation"])