import requests

OSRM_URL = "http://localhost:5000/route/v1/driving"

def find_route(src_lat, src_lon, dst_lat, dst_lon):
    url = f"{OSRM_URL}/{src_lon},{src_lat};{dst_lon},{dst_lat}?overview=full&steps=true"

    resp = requests.get(url).json()
    route = resp["routes"][0]

    return {
        "distance_km": route["distance"] / 1000,
        "duration_minutes": route["duration"] / 60,
        "polyline": route["geometry"],
        "instructions": route["legs"][0]["steps"]
    }
