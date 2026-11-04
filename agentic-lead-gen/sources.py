"""
The data layer — real businesses from OpenStreetMap, free and keyless.

Two public OSM services, no API key, no scraping:
  • Nominatim  — geocode a city name to a bounding box.
  • Overpass   — query POIs (businesses) with their tags inside that box.

The crucial tag is `website` (also `contact:website`/`url`). Its ABSENCE is
our "no website" signal — but note it's a signal, not truth: plenty of real
businesses have a site OSM simply hasn't recorded. That gap is why the pipeline
has a verification step, and why we report a false-positive rate honestly.
"""

import time

import requests

UA = {"User-Agent": "agentic-lead-gen-tutorial/1.0 (agentic-examples)"}
NOMINATIM = "https://nominatim.openstreetmap.org/search"
OVERPASS = "https://overpass-api.de/api/interpreter"
WEBSITE_TAGS = ("website", "contact:website", "url")


def geocode(place: str) -> tuple[float, float, float, float]:
    """City name -> (south, west, north, east) bounding box."""
    r = requests.get(NOMINATIM, headers=UA, timeout=30,
                     params={"q": place, "format": "json", "limit": 1})
    r.raise_for_status()
    hits = r.json()
    if not hits:
        raise ValueError(f"could not geocode {place!r}")
    s, n, w, e = hits[0]["boundingbox"]          # Nominatim: [S, N, W, E]
    return float(s), float(w), float(n), float(e)


def _overpass_filter(osm_filters: list[dict], bbox: str) -> str:
    """Build the union of node/way queries for each {key: value} filter."""
    parts = []
    for f in osm_filters:
        for k, v in f.items():
            parts.append(f'node["{k}"="{v}"]({bbox});')
            parts.append(f'way["{k}"="{v}"]({bbox});')
    return "".join(parts)


def find_businesses(bbox, osm_filters: list[dict], limit: int = 60) -> list[dict]:
    """Return businesses in the bbox matching any filter, as normalized leads."""
    s, w, n, e = bbox
    box = f"{s},{w},{n},{e}"
    q = (f"[out:json][timeout:40];({_overpass_filter(osm_filters, box)});"
         "out tags center;")
    for attempt in range(3):
        r = requests.post(OVERPASS, headers=UA, data={"data": q}, timeout=60)
        if r.status_code == 200:
            break
        time.sleep(2 * (attempt + 1))           # Overpass is rate-limited; back off
    r.raise_for_status()

    leads, seen = [], set()
    for el in r.json()["elements"]:
        t = el.get("tags", {})
        name = t.get("name")
        if not name or name.lower() in seen:
            continue
        seen.add(name.lower())
        website = next((t[k] for k in WEBSITE_TAGS if k in t), None)
        addr = " ".join(filter(None, [
            t.get("addr:housenumber"), t.get("addr:street"),
            t.get("addr:city"), t.get("addr:postcode")])) or None
        leads.append({
            "name": name,
            "phone": t.get("phone") or t.get("contact:phone"),
            "address": addr,
            "website": website,
            "has_website": bool(website),
            "lat": el.get("lat") or el.get("center", {}).get("lat"),
            "lon": el.get("lon") or el.get("center", {}).get("lon"),
        })
        if len(leads) >= limit:
            break
    return leads
