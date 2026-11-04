"""
Corrective RAG (CRAG) — Yan et al., 2024.

Plain RAG trusts whatever the retriever returns. CRAG adds a corrective layer:
a lightweight evaluator scores retrieval quality, and a three-way switch takes
a different action per tier.

    retrieve ─▶ evaluate (evaluator.py) ─▶
        correct    → refine the docs (strip-level) → generate from refined
        incorrect  → discard docs → WEB SEARCH → generate from web knowledge
        ambiguous  → refine local docs + web search → generate from both

The "incorrect → web search" branch is what makes it *corrective*: when the
corpus can't answer, CRAG doesn't hallucinate or abstain — it goes and finds
the knowledge externally. Knowledge refinement (decompose→grade→recompose)
strips noise out of otherwise-relevant docs.
"""

import os

import anthropic
from dotenv import load_dotenv

from evaluator import evaluate
from retriever import retrieve

load_dotenv()
MODEL = os.environ.get("CHAT_MODEL", "claude-haiku-4-5")
client = anthropic.Anthropic()
# Basic web-search variant: the newer web_search_20260209 runs code execution
# under the hood (programmatic tool calling), which Haiku doesn't support.
WEB_SEARCH = {"type": "web_search_20250305", "name": "web_search", "max_uses": 3}


def _llm(system, user, tools=None, max_tokens=600, usage=None):
    kwargs = dict(model=MODEL, max_tokens=max_tokens, system=system,
                  messages=[{"role": "user", "content": user}])
    if tools:
        kwargs["tools"] = tools
    # web search can pause_turn; loop to completion
    msgs = kwargs["messages"]
    while True:
        r = client.messages.create(**{**kwargs, "messages": msgs})
        if usage is not None:
            usage["in"] += r.usage.input_tokens
            usage["out"] += r.usage.output_tokens
        if r.stop_reason == "pause_turn":
            msgs = msgs + [{"role": "assistant", "content": r.content}]
            continue
        return next((b.text for b in r.content if b.type == "text"), "")


def refine(question: str, docs: list[dict], usage: dict) -> str:
    """Knowledge refinement: keep only the parts of the docs that matter."""
    context = "\n\n".join(f"[{d['id']}] {d['text']}" for d in docs)
    return _llm(
        "Extract ONLY the sentences/snippets from these documents that help "
        "answer the question. Drop everything irrelevant. Output the kept snippets.",
        f"Question: {question}\n\nDocuments:\n{context}", max_tokens=400, usage=usage)


def web_knowledge(question: str, usage: dict) -> str:
    return _llm(
        "Search the web to find information that answers the question. Summarize "
        "the key facts you find.",
        question, tools=[WEB_SEARCH], max_tokens=800, usage=usage)


def crag(question: str, k: int = 3) -> dict:
    usage = {"in": 0, "out": 0, "jev": 0}
    docs = retrieve(question, k)
    tier, conf, jtoks = evaluate(question, docs)
    usage["jev"] += jtoks

    if tier == "correct":
        knowledge = refine(question, docs, usage)
    elif tier == "incorrect":
        knowledge = web_knowledge(question, usage)
    else:  # ambiguous → both
        knowledge = (refine(question, docs, usage)
                     + "\n\n--- web ---\n" + web_knowledge(question, usage))

    final = _llm(
        "Answer the question using the provided knowledge. Be concise.",
        f"<knowledge>\n{knowledge}\n</knowledge>\n\nQuestion: {question}",
        usage=usage)
    return {"tier": tier, "confidence": conf, "answer": final, "usage": usage}
