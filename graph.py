from langgraph.graph import StateGraph, START, END

from state import TravelPlannerState
from agents.orchestrator import orchestrator_node, route_after_orchestrator
from agents.search_agent import search_node
from agents.itinerary_agent import itinerary_node
from agents.synthesizer import synthesizer_node, route_after_synthesizer
from agents.exits import ask_user_node, give_up_node, reject_node
from agents.guardrail import guardrail_node, route_after_guardrail

"""
graph.py - wires the agents into one runnable LangGraph workflow.

WHAT THIS FILE DOES
-------------------
Each agent (orchestrator, search, itinerary, synthesizer) is a "node": a function
that reads the shared state (state.py) and returns only the keys it changed.
This file registers those nodes and declares which node runs after which. That
declaration is the graph. compile() then checks it (e.g. no edge points at a
node that doesn't exist) and returns an app we can run.

POSSIBLE RUNS (every run starts with the guardrail)
1. Happy path:      guardrail -> orchestrator -> search -> itinerary -> synthesizer -> END
2. Blocked input:   guardrail -> reject -> END
3. Missing info:    guardrail -> orchestrator -> ask_user -> END
4. Retry, success:  guardrail -> orchestrator -> search -> itinerary -> synthesizer
                    -> search -> itinerary -> synthesizer -> END
5. Retry, give up:  same as 4, but the last step is synthesizer -> give_up -> END

On a retry the itinerary node doesn't redo its work: it sees the draft already
exists and passes straight through.

EDGES VS. CONDITIONAL EDGES
---------------------------
- add_edge(A, B): a fixed connection. After A finishes, B always runs.
  Used where there is no decision: START -> orchestrator, search -> itinerary,
  itinerary -> synthesizer, ask_user -> END, give_up -> END.
- add_conditional_edges(A, router, mapping): a decision made at runtime. After A
  finishes, LangGraph calls the router function with the current state. The
  router returns a plain string, and the mapping translates that string into the
  next node. Used in two places:
    * after orchestrator: route_after_orchestrator -> "search" or "ask_user"
    * after synthesizer:  route_after_synthesizer  -> "end", "search" or "give_up"
- The retry LOOP is just one mapping entry ("search": "search" after the
  synthesizer) pointing back at a node that already ran. The loop is bounded by
  the search_attempts counter and MAX_SEARCH_ATTEMPTS (synthesizer.py). As a
  backstop, LangGraph stops any run exceeding its recursion limit (25 steps by
  default); our longest possible run is 8 steps.

STATUSES: WHICH ONES ACTUALLY AFFECT ROUTING
--------------------------------------------
"status" is only a field in the state. It is NOT an edge. Edges decide where the
run goes; a status is data that a router may or may not read.
- Read by a router:
    "done"        route_after_synthesizer checks it: done -> END.
- Set but never read by name (informational labels):
    "needs_retry"   Set by the synthesizer when no flight+hotel pair fits the
                    budget. route_after_synthesizer does not look for this
                    word: it treats "anything that is not done" as a possible
                    retry and then checks search_attempts < MAX_SEARCH_ATTEMPTS.
                    So the retry edge exists, but this label is not what
                    triggers it.
    "needs_info"    Set by ask_user_node. main.py prints it so you can see how
    "failed"        the run ended; "failed" is set by give_up_node the same way.
    "searching", "itinerary", "synthesizing"
                    Breadcrumbs set by nodes to mark the next stage (note: the
                    orchestrator sets "searching" even if fields are missing;
                    ask_user then overwrites it with "needs_info").
    "planning"      The starting value, set in main.py before any node runs.
- Why route_after_orchestrator uses no status: the data already answers its
  question (are required fields missing?), so a status would just repeat it.
  The synthesizer records a status because "does any pair fit the budget?" needs
  a calculation, and the status carries that verdict forward.
- Setting TravelPlannerState as StateGraph means the graph updates the fields of the state from the return values of the nodes.
"""
def build_graph():
    graph = StateGraph(TravelPlannerState)

    graph.add_node("guardrail", guardrail_node)
    graph.add_node("reject", reject_node)
    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("search", search_node)
    graph.add_node("itinerary", itinerary_node)
    graph.add_node("synthesizer", synthesizer_node)
    graph.add_node("ask_user", ask_user_node)
    graph.add_node("give_up", give_up_node)

    # the order of adding edges and conditional edges doesn't matter; the graph is compiled at the end.
    # but this way shows the flow of the run in a top-down order, which is easier to read.
    graph.add_edge(START, "guardrail")

    graph.add_conditional_edges(
        "guardrail",
        route_after_guardrail,
        {"orchestrator": "orchestrator", "reject": "reject"},
    )

    graph.add_conditional_edges(
        "orchestrator",
        route_after_orchestrator,
        {"search": "search", "ask_user": "ask_user"}, # "search": "search" (<orchestrator_return_value:node>) means if the orchestrator returns "search", go to the search node. "ask_user": "ask_user" means if the orchestrator returns "ask_user", go to the ask_user node. Where the key is the return value of the orchestrator and the value is the name of the node to go to next.
    )

    graph.add_edge("search", "itinerary")
    graph.add_edge("itinerary", "synthesizer")

    graph.add_conditional_edges(
        "synthesizer",
        route_after_synthesizer,
        {"end": END, "search": "search", "give_up": "give_up"},
    )

    graph.add_edge("ask_user", END)
    graph.add_edge("give_up", END)
    graph.add_edge("reject", END)

    return graph.compile()