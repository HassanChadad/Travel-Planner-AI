from graph import build_graph

"""
HOW TO READ THE MERMAID DIAGRAM (app.get_graph().draw_mermaid())
-----------------------------------------------------------------
The diagram is text in Mermaid syntax. Paste it into mermaid.live to see it drawn.
- "graph TD;" means a top-down flowchart.
- Lines like  search(search)  declare boxes. __start__ and __end__ are LangGraph's
  START and END markers, drawn as rounded shapes.
- "-->" (solid arrow) = a fixed edge from add_edge. Example: search --> itinerary.
- "-.->" (dotted arrow) = a conditional edge. Example: orchestrator -.-> search and
  orchestrator -.-> ask_user. Only ONE of a node's dotted arrows is followed in a
  given run; the router picks which.
- A label on an arrow (synthesizer -. end .-> __end__) appears when the router's
  string differs from the target node's name: the router returns "end" but the
  target is END. The other dotted arrows are unlabeled because the router's
  string and the node name are the same (e.g. "search" -> search).
- &nbsp; is an HTML non-breaking space, only there to pad the label.
- The classDef lines and :::first / :::last are styling only (fill colors); they
  don't affect behavior.
- The loop shows up as the path search --> itinerary --> synthesizer -.-> search.
- The diagram shows which paths are possible, not which one a run takes, and it
  can't show the attempt limit that stops the loop.
"""
app = build_graph()
print(app.get_graph().draw_mermaid())