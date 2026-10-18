from tools.web_search import research_attractions

results = research_attractions.invoke({
    "destination": "Beirut Lebanon",
    "interests": "museums, food",
})

for r in results:
    print(r["title"])
    print(r["summary"][:150], "\n")