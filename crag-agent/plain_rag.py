"""Plain RAG baseline — retrieve from the corpus, answer from it only."""

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
        system="Answer using ONLY the provided context. If the context does not "
        "contain the answer, say you don't have that information. Be concise.",
        messages=[{"role": "user", "content":
                   f"<context>\n{context}\n</context>\n\nQuestion: {question}"}])
    text = next((b.text for b in r.content if b.type == "text"), "")
    return text, {"in": r.usage.input_tokens, "out": r.usage.output_tokens}
