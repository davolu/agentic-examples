"""
Demo-site builder — generate a REAL landing page per hot lead.

Claude writes a complete, self-contained HTML page (inline CSS, no external
deps) for the business, using the real details we have (name, phone, address,
category). A reflection pass checks it's finished and professional — no leftover
placeholders — before it's saved. These are real files you can open in a browser
and, if you like, deploy; nothing is published automatically.
"""

import os
import re

import anthropic
from dotenv import load_dotenv
from langsmith import traceable

load_dotenv()
MODEL = os.environ.get("CHAT_MODEL", "claude-haiku-4-5")
OUT = os.path.join(os.path.dirname(__file__), "output", "sites")
_client = anthropic.Anthropic()


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def _chat(system, user, cost, max_tokens=2600):
    r = _client.messages.create(
        model=MODEL, max_tokens=max_tokens, system=system,
        messages=[{"role": "user", "content": user}])
    cost["in"] += r.usage.input_tokens
    cost["out"] += r.usage.output_tokens
    cost["calls"] += 1
    return next((b.text for b in r.content if b.type == "text"), "")


def _extract_html(text: str) -> str:
    if "```" in text:
        text = text.split("```")[1].removeprefix("html").strip()
    return text.strip()


@traceable(name="build-demo-site", run_type="chain")
def build_site(lead: dict, category: str, location: str, cost: dict) -> str:
    """Generate + reflect + save a demo site; return the file path."""
    details = (f"Business: {lead['name']}\nCategory: {category}\n"
               f"Location: {location}\nPhone: {lead['phone'] or 'N/A'}\n"
               f"Address: {lead['address'] or 'N/A'}")
    html = _extract_html(_chat(
        "Build a complete, single-file HTML landing page (inline CSS, no external "
        "resources) for this local business. Modern, clean, mobile-friendly: a "
        "hero with the name and a tagline, a short about, 3–4 plausible services "
        "for the category, hours, and a contact section using the real phone/"
        "address if given. Include a small 'Demo site' note in the footer. Output "
        "only the HTML.", details, cost))
    # Reflection: finished & professional, no placeholders.
    fixed = _chat(
        "Review this landing page HTML. If it is complete and professional with "
        "no leftover placeholders (no 'lorem ipsum', no '[...]', no 'TODO'), reply "
        "exactly 'OK'. Otherwise reply with the corrected full HTML only.",
        html, cost)
    if fixed.strip() != "OK" and "<" in fixed:
        html = _extract_html(fixed)

    d = os.path.join(OUT, slug(lead["name"]))
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "index.html")
    with open(path, "w") as f:
        f.write(html)
    return path
