"""
Agentic RAG — the agent controls retrieval with judgment.

The loop (Corrective-RAG / Self-RAG flavored):

    rewrite the query  ─▶  retrieve  ─▶  grade each doc for relevance
                                              │
                   ┌── relevant docs found ───┤── nothing relevant ──┐
                   ▼                          (reformulate & retry)  │
             generate (grounded)                                     ▼
                                                   out of retries → ABSTAIN

The decision that matters is **grade relevance**: a vector DB always returns
the k nearest docs, so without this step there's no way to tell "good match"
from "nearest-but-irrelevant." Grading is what lets the agent abstain on
out-of-scope questions instead of hallucinating — done cheaply with a Jev
Noul (LLM fallback if no TypeSafe key).
"""

import os

import anthropic
from dotenv import load_dotenv

from retriever import retrieve

load_dotenv()
MODEL = os.environ.get("CHAT_MODEL", "claude-haiku-4-5")
JEV_LIVE = bool(os.environ.get("TYPESAFE_API_KEY"))
JEV_MODEL = os.environ.get("JEV_MODEL", "jev-latest")
if JEV_LIVE:
    from typesafe_sdk import Noul, TypeSafeClient

client = anthropic.Anthropic()
MAX_RETRIES = 2
ABSTAIN = "I don't have information about that in my knowledge base."


def _llm(system: str, user: str, max_tokens: int = 400, usage=None) -> str:
    r = client.messages.create(model=MODEL, max_tokens=max_tokens, system=system,
                               messages=[{"role": "user", "content": user}])
    if usage is not None:
        usage["in"] += r.usage.input_tokens
        usage["out"] += r.usage.output_tokens
        usage["llm_calls"] += 1
    return next((b.text for b in r.content if b.type == "text"), "")


def rewrite_query(question: str, usage: dict) -> str:
    """Turn a user question (maybe vague) into a good search query."""
    return _llm(
        "Rewrite the user's question as a concise search query (keywords/phrase) "
        "to retrieve relevant docs. Output only the query.",
        question, max_tokens=60, usage=usage).strip().strip('"')


def grade_relevant(question: str, doc: str, usage: dict) -> bool:
    """Is this doc actually useful for answering the question?"""
    if JEV_LIVE:
        with TypeSafeClient(model=JEV_MODEL) as c:
            r = c.system_one(
                state={"question": question, "document": doc},
                questions={"relevant": Noul(
                    instructions="The document contains information that helps "
                    "answer the question")})
        usage["jev"] += getattr(r, "usage", None).input_tokens if getattr(r, "usage", None) else 0
        usage["jev_calls"] += 1
        return r.nouls["relevant"].noul > 0.5
    verdict = _llm(
        "Does this document help answer the question? Reply only 'yes' or 'no'.",
        f"Question: {question}\n\nDocument: {doc}", max_tokens=5, usage=usage)
    return verdict.strip().lower().startswith("y")


def answer(question: str, k: int = 3) -> tuple[str, dict]:
    usage = {"in": 0, "out": 0, "jev": 0, "llm_calls": 0, "jev_calls": 0,
             "retrievals": 0, "abstained": False}
    query = rewrite_query(question, usage)

    relevant = []
    for attempt in range(MAX_RETRIES + 1):
        usage["retrievals"] += 1
        docs = retrieve(query, k)
        relevant = [d for d in docs if grade_relevant(question, d["text"], usage)]
        if relevant:
            break
        # Nothing relevant — reformulate and try once more, then give up.
        if attempt < MAX_RETRIES:
            query = _llm(
                "The previous search found nothing relevant. Rewrite the search "
                "query differently to find docs that answer the question. Output "
                "only the query.",
                f"Question: {question}\nPrevious query: {query}",
                max_tokens=60, usage=usage).strip().strip('"')

    if not relevant:
        usage["abstained"] = True
        return ABSTAIN, usage

    context = "\n\n".join(f"[{d['id']}] {d['text']}" for d in relevant)
    final = _llm(
        "Answer the question using ONLY the provided context. If the context "
        "doesn't fully answer it, say what you can and note what's missing. "
        "Be concise.",
        f"<context>\n{context}\n</context>\n\nQuestion: {question}",
        usage=usage)
    return final, usage
