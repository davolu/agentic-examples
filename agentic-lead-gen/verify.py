"""
Verification — the honesty step.

OSM saying a business has no `website` tag does NOT mean it has no website;
the tag is just missing. So before we spend effort building a demo site and
drafting a pitch, we web-search the business name + city and check whether an
official site actually turns up. Leads that really have a site are dropped as
false positives — and we report how many, because that false-positive rate is
the honest headline of this whole example.
"""

import os

import anthropic
from dotenv import load_dotenv
from langsmith import traceable

load_dotenv()
MODEL = os.environ.get("CHAT_MODEL", "claude-haiku-4-5")
# Basic web-search variant works on Haiku (the newer one needs programmatic calling).
WEB_SEARCH = {"type": "web_search_20250305", "name": "web_search", "max_uses": 2}
_client = anthropic.Anthropic()


@traceable(name="verify-no-website", run_type="chain")
def has_real_website(lead: dict, location: str, cost: dict) -> bool:
    """True if the business appears to actually have an official website."""
    msgs = [{"role": "user", "content":
             f"Does the business \"{lead['name']}\" in {location} have its own "
             f"official website? Search, then answer on the LAST line exactly "
             f"'VERDICT: YES' (an official site exists) or 'VERDICT: NO'."}]
    while True:
        r = _client.messages.create(
            model=MODEL, max_tokens=700, tools=[WEB_SEARCH], messages=msgs)
        cost["in"] += r.usage.input_tokens
        cost["out"] += r.usage.output_tokens
        cost["calls"] += 1
        if r.stop_reason == "pause_turn":
            msgs = msgs + [{"role": "assistant", "content": r.content}]
            continue
        text = "\n".join(b.text for b in r.content if b.type == "text")
        return "VERDICT: YES" in text.upper()
