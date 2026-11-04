"""
Lead scoring — a cheap calibrated judgement, i.e. a decision model's job.

A lead is "hot" when there's clear opportunity and enough to act on: no website
(the gap you'd sell into), but a real name, a phone, an address — signals the
business is established and reachable. We score 0..1 with a Jev Score, LLM
fallback when no TypeSafe key is set. Scoring is per-lead and cheap, so the
expensive steps (verify, build, pitch) only run on the best ones.
"""

import os

from dotenv import load_dotenv
from langsmith import traceable

load_dotenv()
JEV_LIVE = bool(os.environ.get("TYPESAFE_API_KEY"))
JEV_MODEL = os.environ.get("JEV_MODEL", "jev-latest")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "claude-haiku-4-5")

if JEV_LIVE:
    from typesafe_sdk import Score, TypeSafeClient
import anthropic

_client = anthropic.Anthropic()


def _state(lead: dict) -> dict:
    return {"name": lead["name"], "has_phone": bool(lead["phone"]),
            "has_address": bool(lead["address"]),
            "has_website": lead["has_website"]}


@traceable(name="score-lead", run_type="chain")
def score(lead: dict) -> float:
    """0..1 opportunity score for one lead."""
    if JEV_LIVE:
        with TypeSafeClient(model=JEV_MODEL) as c:
            r = c.system_one(
                state=_state(lead),
                questions={"opportunity": Score(
                    instructions="How good a cold-outreach lead is this business "
                    "for a 'we'll build you a website' pitch? Best = no website "
                    "but clearly a real, reachable business (name + phone + "
                    "address). Worst = already has a website, or too little info.",
                    criteria=["Poor — has a site, or barely any details",
                              "Okay — some signal but thin",
                              "Strong — no site, real and reachable"])})
        return r.scores["opportunity"].score / 2.0
    out = _client.messages.create(
        model=CHAT_MODEL, max_tokens=5,
        system="Rate 0,1,2 how good a lead this business is for a 'we'll build "
        "you a website' pitch (2=no website but real & reachable, 0=has a site "
        "or too thin). Reply only the digit.",
        messages=[{"role": "user", "content": str(_state(lead))}])
    txt = next((b.text for b in out.content if b.type == "text"), "0").strip()
    return (int(txt[0]) if txt and txt[0] in "012" else 0) / 2.0
