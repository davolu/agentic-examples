"""
The complexity router — the heart of Adaptive-RAG.

The paper trains a small Flan-T5 to classify each query's complexity into three
buckets. We can't fine-tune, so we test the practical question instead: can an
*untrained*, cheap classifier route well enough? Three routers, same interface
(question -> ('A'|'B'|'C', tokens)):

    route_jev     a Jev Score (0..2 complexity rubric → A/B/C) — a purpose-built
                  cheap calibrated classifier, the faithful analog of the paper's
                  dedicated T5. Falls back to the LLM router if no TypeSafe key.
    route_llm     ask the chat model to classify — the "just use the LLM" baseline
                  and the no-key fallback.
    route_oracle  the hand-labeled route from corpus.py — perfect routing, the
                  upper bound the two learned routers are scored against.

Routing rubric given to both learned routers:
    A  the model can answer from general knowledge, no documents needed
    B  a single company-specific fact — one lookup answers it
    C  multi-hop — the answer depends on first resolving another fact
"""

import os

from dotenv import load_dotenv

load_dotenv()
JEV_LIVE = bool(os.environ.get("TYPESAFE_API_KEY"))
JEV_MODEL = os.environ.get("JEV_MODEL", "jev-latest")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "claude-haiku-4-5")

if JEV_LIVE:
    from typesafe_sdk import Score, TypeSafeClient
import anthropic

_client = anthropic.Anthropic()
_IDX = {0: "A", 1: "B", 2: "C"}

_RUBRIC = (
    "Classify how much retrieval a question about the NovaTech product needs:\n"
    "0 = answerable from general world knowledge alone, no documents needed\n"
    "1 = one simple company-specific fact — a single document lookup answers it\n"
    "2 = multi-hop — you must first resolve one fact to know what to look up next")


def route_oracle(question: str, label: str) -> tuple[str, int]:
    return label, 0


def route_llm(question: str) -> tuple[str, int]:
    r = _client.messages.create(
        model=CHAT_MODEL, max_tokens=5,
        system=_RUBRIC + "\n\nReply with only the single digit 0, 1, or 2.",
        messages=[{"role": "user", "content": question}])
    txt = next((b.text for b in r.content if b.type == "text"), "1").strip()
    idx = int(txt[0]) if txt and txt[0] in "012" else 1
    return _IDX[idx], r.usage.input_tokens + r.usage.output_tokens


def route_jev(question: str) -> tuple[str, int]:
    if not JEV_LIVE:
        return route_llm(question)
    with TypeSafeClient(model=JEV_MODEL) as c:
        r = c.system_one(
            state={"question": question},
            questions={"complexity": Score(
                instructions=_RUBRIC,
                criteria=["Answerable from general knowledge — no retrieval",
                          "One simple company-specific lookup",
                          "Multi-hop — depends on resolving another fact first"])})
    idx = int(round(r.scores["complexity"].score))
    toks = getattr(r, "usage", None).input_tokens if getattr(r, "usage", None) else 0
    return _IDX[max(0, min(2, idx))], toks


ROUTERS = {"jev": route_jev, "llm": route_llm}
