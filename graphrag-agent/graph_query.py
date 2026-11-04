"""
GraphRAG query time — two modes.

global_search (the sensemaking win): map-reduce over community summaries. Each
community drafts a partial answer to the question; a cheap judge (Jev) rates how
helpful that partial is; the helpful ones are synthesized into a final answer.
Because it sweeps EVERY community, it sees the whole corpus — not just a top-k
slice — which is what a global question needs.

local_search: match the question to entities, pull their source tickets, answer.
Entity-centric RAG for specific questions — where plain vector RAG is just as
good and cheaper.
"""

import os

import anthropic
from dotenv import load_dotenv

from corpus import TICKETS

load_dotenv()
MODEL = os.environ.get("CHAT_MODEL", "claude-haiku-4-5")
JEV_LIVE = bool(os.environ.get("TYPESAFE_API_KEY"))
JEV_MODEL = os.environ.get("JEV_MODEL", "jev-latest")
HELPFUL = 0.34   # keep partials scoring above this (0..1)

if JEV_LIVE:
    from typesafe_sdk import Score, TypeSafeClient
_client = anthropic.Anthropic()


def _llm(system, user, cost, max_tokens=500):
    r = _client.messages.create(
        model=MODEL, max_tokens=max_tokens, system=system,
        messages=[{"role": "user", "content": user}])
    cost["in"] += r.usage.input_tokens
    cost["out"] += r.usage.output_tokens
    cost["calls"] += 1
    return next((b.text for b in r.content if b.type == "text"), "").strip()


def _helpfulness(question, partial, cost) -> float:
    """Jev rates how much a partial answer helps answer the question (0..1)."""
    if JEV_LIVE:
        with TypeSafeClient(model=JEV_MODEL) as c:
            r = c.system_one(
                state={"question": question, "partial_answer": partial},
                questions={"helpful": Score(
                    instructions="How much does this partial answer contribute "
                    "to answering the question?",
                    criteria=["Nothing — irrelevant or empty",
                              "A little — minor or tangential point",
                              "A lot — a substantive, on-point contribution"])})
        cost["jev"] += getattr(r.usage, "input_tokens", 0) if getattr(r, "usage", None) else 0
        return r.scores["helpful"].score / 2.0
    # No Jev key: ask the model for a 0/1/2 self-rating.
    txt = _llm("Rate 0, 1, or 2 how much this partial answer helps answer the "
               "question. Reply with only the digit.",
               f"Q: {question}\n\nPartial: {partial}", cost, max_tokens=5)
    return (int(txt[0]) if txt and txt[0] in "012" else 0) / 2.0


def global_search(question: str, index: dict) -> tuple[str, dict]:
    cost = {"in": 0, "out": 0, "calls": 0, "jev": 0}
    kept = []
    for comm in index["communities"]:
        partial = _llm(
            "You are given a summary of one cluster of customer-support topics. "
            "Extract what it contributes to answering the question. If it "
            "contributes nothing, reply exactly NONE.",
            f"Cluster summary:\n{comm['summary']}\n\nQuestion: {question}", cost)
        if partial.upper().startswith("NONE"):
            continue
        if _helpfulness(question, partial, cost) > HELPFUL:
            kept.append(partial)
    joined = "\n\n---\n\n".join(kept)
    final = _llm(
        "Synthesize the partial findings below into one comprehensive answer to "
        "the question. Cover every distinct problem they raise; group related "
        "points. Be specific and well-organized.",
        f"<partials>\n{joined}\n</partials>\n\nQuestion: {question}",
        cost, max_tokens=900)
    cost["kept"] = len(kept)
    cost["communities"] = len(index["communities"])
    return final, cost


def local_search(question: str, index: dict, max_tickets: int = 8) -> tuple[str, dict]:
    cost = {"in": 0, "out": 0, "calls": 0, "jev": 0}
    q = question.lower()
    # Match entities whose words appear in the question.
    hits = set()
    for ent, tids in index["entity_tickets"].items():
        words = [w for w in ent.split() if len(w) > 3]
        if any(w in q for w in words):
            hits.update(tids)
    tickets = sorted(hits)[:max_tickets] or list(TICKETS)[:max_tickets]
    ctx = "\n\n".join(f"[{t}] {TICKETS[t]['text']}" for t in tickets)
    ans = _llm("Answer the question using ONLY the provided tickets. Be specific "
               "and concise.",
               f"<tickets>\n{ctx}\n</tickets>\n\nQuestion: {question}", cost)
    cost["tickets"] = len(tickets)
    return ans, cost
