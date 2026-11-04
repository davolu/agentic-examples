"""
The decision layer — Jev routes each incoming message before any LLM work.

Two cheap calibrated decisions (a Jev Score + a Noul), with an LLM fallback when
no TypeSafe key is set:

  complexity : simple | complex   — adaptive retrieval. Simple FAQ → one KB
               lookup; complex/multi-part → retrieve more and reason.
  account    : is this specific to THIS customer's account/history? If so the
               copilot leans on long-term memory, not just the KB.

This is the Adaptive-RAG + decision-model idea applied at the front door: don't
spend generation tokens deciding how to handle a message — classify it cheaply.
"""

import os

from dotenv import load_dotenv
from langsmith import traceable

load_dotenv()
JEV_LIVE = bool(os.environ.get("TYPESAFE_API_KEY"))
JEV_MODEL = os.environ.get("JEV_MODEL", "jev-latest")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "claude-haiku-4-5")

if JEV_LIVE:
    from typesafe_sdk import Noul, Score, TypeSafeClient


@traceable(name="route", run_type="chain")
def route(message: str, client) -> dict:
    """Return {'complexity': 'simple'|'complex', 'account': bool, 'jev': tokens}."""
    if JEV_LIVE:
        with TypeSafeClient(model=JEV_MODEL) as c:
            r = c.system_one(
                state={"message": message},
                questions={
                    "complexity": Score(
                        instructions="How complex is this support request?",
                        criteria=["Simple — one clear question",
                                  "Moderate — a couple of points",
                                  "Complex — multi-part or needs reasoning"]),
                    "account": Noul(
                        instructions="Is this specific to the customer's own "
                        "account, order, or past history (vs a general question "
                        "about the product)?")})
        toks = getattr(r.usage, "input_tokens", 0) if getattr(r, "usage", None) else 0
        return {"complexity": "complex" if r.scores["complexity"].score >= 1.5
                else "simple",
                "account": r.nouls["account"].noul >= 0.5,
                "jev": toks}
    # LLM fallback: one tiny call returning two flags.
    out = client.messages.create(
        model=CHAT_MODEL, max_tokens=10,
        system="Classify a support message. Reply exactly two tokens separated "
        "by a space: complexity (simple|complex) and account (yes|no, is it "
        "specific to this customer's own account/history).",
        messages=[{"role": "user", "content": message}])
    txt = next((b.text for b in out.content if b.type == "text"), "simple no").lower()
    parts = txt.split()
    return {"complexity": "complex" if parts and "complex" in parts[0] else "simple",
            "account": len(parts) > 1 and parts[1].startswith("y"),
            "jev": 0}
