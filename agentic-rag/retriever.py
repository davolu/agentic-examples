"""
The retrieval substrate — a real vector DB (ChromaDB, embedded).

This is deliberately a *good* semantic retriever, not a weak keyword one.
ChromaDB embeds the docs with a local model (all-MiniLM-L6-v2 via onnxruntime
— no embeddings API key needed) and does cosine similarity search.

The point of the agentic-RAG example is NOT that the retriever is bad — it's
that even a good vector retriever has a blind spot: `query()` ALWAYS returns
the k nearest documents, even when none are actually relevant. Cosine
similarity has no notion of "nothing here matches." That's why the agent's
relevance grading matters.
"""

import chromadb

from corpus import DOCS

_client = chromadb.Client()                      # in-memory, embedded
_collection = _client.get_or_create_collection("novatech")

# Index the corpus once (Chroma embeds with its default local model).
if _collection.count() == 0:
    _collection.add(documents=list(DOCS.values()), ids=list(DOCS.keys()))


def retrieve(query: str, k: int = 3) -> list[dict]:
    """Return the top-k nearest docs: [{id, text, distance}]. ALWAYS returns k
    results if the corpus has them — relevant or not."""
    res = _collection.query(query_texts=[query], n_results=k)
    return [
        {"id": i, "text": t, "distance": d}
        for i, t, d in zip(res["ids"][0], res["documents"][0], res["distances"][0])
    ]
