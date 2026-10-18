from tools.flight_search import search_flights

print(search_flights.invoke({
    "departure_id": "IAD",
    "arrival_id": "BEY",
    "outbound_date": "2026-12-10",
    "return_date": "2027-01-03",
}))