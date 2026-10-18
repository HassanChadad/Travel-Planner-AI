from tools.hotel_search import search_hotels

print(search_hotels.invoke({
    "destination": "Beirut",
    "check_in_date": "2026-12-10",
    "check_out_date": "2026-12-17",
}))