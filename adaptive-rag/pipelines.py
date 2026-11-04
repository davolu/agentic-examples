"""
The three execution pipelines Adaptive-RAG routes between.

Each has the same signature -> (answer, stats) where stats tracks the two
things that make the routing worth it: number of retrieval calls and tokens.

    no_retrieval  (route A) — answer straight from the model, zero retrieval
    single_step   (route B) — retrieve k once, answer from that context
    multi_step    (route C) — iterative self-ask: retrieve, reason about what's
                              still missing, retrieve again, up to MAX_HOPS,
                              then answer from the accumulated context
"""

import os

import anthropic
from dotenv import load_dotenv

from retriever import retrieve

load_dotenv()
MODEL = os.environ.get("CHAT_MODEL", "claude-haiku-4-5")
MAX_HOPS = 3
_client = anthropic.Anthropic()


def _new_stats():
    return {"retrievals": 0, "in": 0, "out": 0}


def _ask(system, user, stats, max_tokens=400):
    r = _client.messages.create(
        model=MODEL, max_tokens=max_tokens, system=system,
        messages=[{"role": "user", "content": user}])
    stats["in"] += r.usage.input_tokens
    stats["out"] += r.usage.output_tokens
    return next((b.text for b in r.content if b.type == "text"), "").strip()


def _fmt(docs):
    return "\n\n".join(f"[{d['id']}] {d['text']}" for d in docs)


def no_retrieval(question: str) -> tuple[str, dict]:
    stats = _new_stats()
    ans = _ask("Answer the question directly and concisely from your own "
               "knowledge.", question, stats)
    return ans, stats


def single_step(question: str, k: int = 3) -> tuple[str, dict]:
    stats = _new_stats()
    docs = retrieve(question, k)
    stats["retrievals"] += 1
    ans = _ask("Answer using ONLY the provided context. If the context does not "
               "contain the answer, say you don't have that information. Be concise.",
               f"<context>\n{_fmt(docs)}\n</context>\n\nQuestion: {question}", stats)
    return ans, stats


def multi_step(question: str, k: int = 3) -> tuple[str, dict]:
    """Self-ask loop: alternately SEARCH for the next missing fact, then ANSWER."""
    stats = _new_stats()
    gathered: list[dict] = []
    next_query = question
    for _ in range(MAX_HOPS):
        docs = retrieve(next_query, k)
        stats["retrievals"] += 1
        gathered += docs
        decision = _ask(
            "You answer multi-hop questions by searching one step at a time. "
            "Given the question and the context gathered so far, decide:\n"
            "- If you still need another fact, reply exactly: SEARCH: <a short "
            "search query for the missing fact>\n"
            "- If you can now answer, reply exactly: ANSWER: <the final answer>\n"
            "Prefer ANSWER once the context is sufficient.",
            f"<context>\n{_fmt(gathered)}\n</context>\n\nQuestion: {question}",
            stats)
        if decision.upper().startswith("ANSWER:"):
            return decision.split(":", 1)[1].strip(), stats
        if decision.upper().startswith("SEARCH:"):
            next_query = decision.split(":", 1)[1].strip()
            continue
        next_query = question
    # Out of hops — force an answer from everything gathered.
    ans = _ask("Answer using ONLY the provided context. Be concise.",
               f"<context>\n{_fmt(gathered)}\n</context>\n\nQuestion: {question}",
               stats)
    return ans, stats


PIPELINES = {"A": no_retrieval, "B": single_step, "C": multi_step}


def run_route(route: str, question: str) -> tuple[str, dict]:
    return PIPELINES[route](question)
