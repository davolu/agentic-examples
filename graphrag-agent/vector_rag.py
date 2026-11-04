"""
Vector-RAG baseline — ChromaDB over the tickets, top-k, answer.

This is the thing GraphRAG is compared against. On a LOCAL question it's great
and cheap. On a GLOBAL question it is structurally limited: it can only put its
top-k tickets in front of the model, so it only ever sees a slice of the corpus
and misses the themes that live in the tickets it didn't retrieve.
"""

import os

import anthropic
import chromadb
from dotenv import load_dotenv

from corpus import TICKETS

load_dotenv()
MODEL = os.environ.get("CHAT_MODEL", "claude-haiku-4-5")
_client = anthropic.Anthropic()

_chroma = chromadb.Client()
_col = _chroma.get_or_create_collection("novatech_tickets")
if _col.count() == 0:
    _col.add(documents=[t["text"] for t in TICKETS.values()],
             ids=list(TICKETS.keys()))


def answer(question: str, k: int = 6) -> tuple[str, dict]:
    res = _col.query(query_texts=[question], n_results=k)
    ctx = "\n\n".join(f"[{i}] {d}"
                      for i, d in zip(res["ids"][0], res["documents"][0]))
    r = _client.messages.create(
        model=MODEL, max_tokens=600,
        system="You answer questions about customer support tickets using ONLY "
        "the provided tickets. Be specific and concise.",
        messages=[{"role": "user", "content":
                   f"<tickets>\n{ctx}\n</tickets>\n\nQuestion: {question}"}])
    text = next((b.text for b in r.content if b.type == "text"), "")
    return text, {"in": r.usage.input_tokens, "out": r.usage.output_tokens,
                  "calls": 1, "k": k}
