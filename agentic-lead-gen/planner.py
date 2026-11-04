"""
The planner — turn one sentence into a structured campaign.

"Find dentists in Austin without a website" becomes a spec the pipeline can
execute: what to search for (mapped to OSM tags), where, what to filter on, and
which actions to run. This is the "you didn't configure anything" step — the
agent writes the campaign config from intent.
"""

import json
import os

import anthropic
from dotenv import load_dotenv
from langsmith import traceable

load_dotenv()
MODEL = os.environ.get("CHAT_MODEL", "claude-haiku-4-5")
_client = anthropic.Anthropic()

_SYSTEM = """You turn a lead-gen request into a JSON campaign spec. Map the \
business category to OpenStreetMap tags (a list of {key: value}). Common ones:
  dentist -> [{"amenity":"dentist"},{"healthcare":"dentist"}]
  restaurant -> [{"amenity":"restaurant"}]
  cafe/coffee -> [{"amenity":"cafe"}]
  plumber -> [{"craft":"plumber"}]
  electrician -> [{"craft":"electrician"}]
  lawyer -> [{"office":"lawyer"}]
  gym/fitness -> [{"leisure":"fitness_centre"}]
  hair/salon -> [{"shop":"hairdresser"}]
  bakery -> [{"shop":"bakery"}]
  auto repair -> [{"shop":"car_repair"}]
Return ONLY JSON:
{"category": "...", "location": "City, Region",
 "osm_filters": [{"key":"value"}, ...],
 "require_no_website": true/false,
 "max_leads": 60, "top_n": 3}
require_no_website defaults true if they mention "without a website" / "no site".
top_n is how many top leads to build demo sites + pitches for (default 3)."""


@traceable(name="plan-campaign", run_type="chain")
def plan(prompt: str) -> dict:
    r = _client.messages.create(
        model=MODEL, max_tokens=400, system=_SYSTEM,
        messages=[{"role": "user", "content": prompt}])
    raw = next((b.text for b in r.content if b.type == "text"), "{}").strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1].removeprefix("json").strip()
    spec = json.loads(raw)
    spec.setdefault("max_leads", 60)
    spec.setdefault("top_n", 3)
    spec.setdefault("require_no_website", True)
    return spec
