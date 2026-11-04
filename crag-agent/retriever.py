"""ChromaDB vector retriever over the NovaTech corpus (local embeddings, no key)."""

import chromadb

from corpus import DOCS

_client = chromadb.Client()
_collection = _client.get_or_create_collection("novatech_crag")
if _collection.count() == 0:
    _collection.add(documents=list(DOCS.values()), ids=list(DOCS.keys()))


def retrieve(query: str, k: int = 3) -> list[dict]:
    res = _collection.query(query_texts=[query], n_results=k)
    return [{"id": i, "text": t, "distance": d}
            for i, t, d in zip(res["ids"][0], res["documents"][0], res["distances"][0])]
