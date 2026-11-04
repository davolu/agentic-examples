"""
Naive RAG — the baseline. Retrieve top-k, stuff into context, generate.

No judgment anywhere: it trusts the retriever's top-k unconditionally. That's
fine for a clean lookup, and it's exactly how it hallucinates on an out-of-
scope question — the vector DB still hands back the k nearest docs, and the
model dutifully answers from them.
"""

import os

import anthropic
from dotenv import load_dotenv

from retriever import retrieve

load_dotenv()
MODEL = os.environ.get("CHAT_MODEL", "claude-haiku-4-5")
client = anthropic.Anthropic()


def answer(question: str, k: int = 3) -> tuple[str, dict]:
    docs = retrieve(question, k)
    context = "\n\n".join(f"[{d['id']}] {d['text']}" for d in docs)
    r = client.messages.create(
        model=MODEL, max_tokens=400,
        system="Answer the question using the provided context documents. "
        "Be concise.",
        messages=[{"role": "user", "content":
                   f"<context>\n{context}\n</context>\n\nQuestion: {question}"}])
    text = next((b.text for b in r.content if b.type == "text"), "")
    return text, {"in": r.usage.input_tokens, "out": r.usage.output_tokens,
                  "retrievals": 1, "llm_calls": 1}
