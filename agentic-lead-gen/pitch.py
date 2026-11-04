"""
Pitch drafting — DRAFT ONLY. Nothing is ever sent.

This is a hard boundary: automatically sending unsolicited outreach to real
businesses is cold emailing/spam, with consent and anti-spam-law implications.
So Agentic Lead Gen writes a reviewable draft and stops. A human reads it and decides
whether (and how) to send — the human-in-the-loop pattern applied to the one
irreversible, outward-facing step in the whole pipeline.
"""

import os

import anthropic
from dotenv import load_dotenv
from langsmith import traceable

load_dotenv()
MODEL = os.environ.get("CHAT_MODEL", "claude-haiku-4-5")
OUT = os.path.join(os.path.dirname(__file__), "output", "pitches")
_client = anthropic.Anthropic()


@traceable(name="draft-pitch", run_type="chain")
def draft_pitch(lead: dict, category: str, location: str, site_path: str,
                cost: dict) -> str:
    """Write a short outreach draft referencing the demo site. Saved, not sent."""
    r = _client.messages.create(
        model=MODEL, max_tokens=400,
        system="Write a short, friendly, non-pushy cold-outreach email draft to a "
        "local business that has no website. Mention you noticed they don't have "
        "one, that you built a free demo they can look at, and invite a quick "
        "chat. Under 120 words. Plain text. This is a DRAFT for a human to review.",
        messages=[{"role": "user", "content":
                   f"Business: {lead['name']} ({category}) in {location}. "
                   f"Demo site built at: {site_path}"}])
    cost["in"] += r.usage.input_tokens
    cost["out"] += r.usage.output_tokens
    cost["calls"] += 1
    body = next((b.text for b in r.content if b.type == "text"), "").strip()

    from demo_site import slug
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{slug(lead['name'])}.txt")
    with open(path, "w") as f:
        f.write(f"[DRAFT — review before sending; do not auto-send]\n"
                f"To: {lead['name']} ({lead['phone'] or 'no phone on file'})\n\n{body}\n")
    return body
