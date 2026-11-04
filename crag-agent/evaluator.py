"""
The retrieval evaluator — CRAG's lightweight relevance judge.

The paper fine-tunes a small T5 for this; the key property is that it's CHEAP,
not the big generator. That's exactly a decision model's job, so we use a Jev
Score (0..1 confidence that the retrieved docs answer the question), with an
LLM fallback when no TypeSafe key is set.

The score is bucketed into CRAG's three tiers, which drive the whole pipeline:
    confidence ≥ UPPER   → "correct"    (docs are good → refine + use them)
    confidence ≤ LOWER   → "incorrect"  (docs are junk → discard, go to the web)
    in between           → "ambiguous"  (use both local + web)
"""

import os

from dotenv import load_dotenv

load_dotenv()
JEV_LIVE = bool(os.environ.get("TYPESAFE_API_KEY"))
JEV_MODEL = os.environ.get("JEV_MODEL", "jev-latest")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "claude-haiku-4-5")

UPPER, LOWER = 0.65, 0.35   # tier thresholds

if JEV_LIVE:
    from typesafe_sdk import Score, TypeSafeClient
import anthropic

_client = anthropic.Anthropic()


def evaluate(question: str, docs: list[dict]) -> tuple[str, float, int]:
    """Return (tier, confidence, jev_tokens)."""
    context = "\n\n".join(d["text"] for d in docs)
    if JEV_LIVE:
        with TypeSafeClient(model=JEV_MODEL) as c:
            r = c.system_one(
                state={"question": question, "retrieved_documents": context},
                questions={"relevance": Score(
                    instructions="How well do the retrieved documents actually "
                    "answer the question?",
                    criteria=["Not at all — irrelevant or off-topic",
                              "Partially — some related info but incomplete",
                              "Well — the documents clearly answer the question"])})
        conf = r.scores["relevance"].score / 2.0   # 0..2 rubric → 0..1
        toks = getattr(r, "usage", None).input_tokens if getattr(r, "usage", None) else 0
    else:
        out = _client.messages.create(
            model=CHAT_MODEL, max_tokens=5,
            system="Rate 0, 1, or 2 how well the documents answer the question "
            "(0=irrelevant, 1=partial, 2=fully). Reply with only the number.",
            messages=[{"role": "user", "content":
                       f"Q: {question}\n\nDocs:\n{context}"}])
        txt = next((b.text for b in out.content if b.type == "text"), "0").strip()
        conf = (int(txt[0]) if txt and txt[0].isdigit() else 0) / 2.0
        toks = 0

    tier = "correct" if conf >= UPPER else "incorrect" if conf <= LOWER else "ambiguous"
    return tier, conf, toks
