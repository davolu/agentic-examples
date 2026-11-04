"""
Long-term memory — the part a hosted vector DB is actually for.

Unlike the KB (rebuilt each run), memory PERSISTS across sessions and is scoped
PER CUSTOMER, so the copilot can recall "you contacted us last week about your
export" on a later, separate conversation.

Pluggable behind one interface so the tutorial stays runnable for everyone:
  • ChromaStore  — default. Local, persistent (on disk), no API key. Filters by
                   customer_id in metadata.
  • PineconeStore — the production backend. Per-customer NAMESPACE, hosted
                    integrated embeddings, scales to millions of memories.
                    Selected automatically when PINECONE_API_KEY is set.

Which one you get is decided by get_store(); the rest of the app doesn't care.
"""

import os
import time
import uuid

from dotenv import load_dotenv

load_dotenv()


class ChromaStore:
    """Local, persistent, per-customer memory (default — no key needed)."""

    def __init__(self):
        import chromadb
        path = os.path.join(os.path.dirname(__file__), ".chroma_memory")
        self._client = chromadb.PersistentClient(path=path)
        self._col = self._client.get_or_create_collection("customer_memory")

    def add(self, customer_id: str, text: str, kind: str = "episodic"):
        self._col.add(
            documents=[text], ids=[str(uuid.uuid4())],
            metadatas=[{"customer_id": customer_id, "kind": kind,
                        "ts": time.time()}])

    def recall(self, customer_id: str, query: str, k: int = 3) -> list[str]:
        if self._col.count() == 0:
            return []
        res = self._col.query(query_texts=[query], n_results=k,
                              where={"customer_id": customer_id})
        return res["documents"][0] if res["documents"] else []

    @property
    def backend(self) -> str:
        return "chroma (local, persistent)"


class PineconeStore:
    """Production backend: per-customer namespace, hosted integrated embeddings.

    Not exercised by the self-contained run (needs a Pinecone account); this is
    the drop-in you'd use in production. Uses Pinecone's integrated embedding so
    you upsert/query raw text and Pinecone embeds server-side.
    """

    INDEX = os.environ.get("PINECONE_INDEX", "copilot-memory")
    MODEL = os.environ.get("PINECONE_EMBED_MODEL", "llama-text-embed-v2")

    def __init__(self):
        from pinecone import Pinecone
        self._pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        if not self._pc.has_index(self.INDEX):
            self._pc.create_index_for_model(
                name=self.INDEX, cloud="aws", region="us-east-1",
                embed={"model": self.MODEL, "field_map": {"text": "text"}})
        self._index = self._pc.Index(self.INDEX)

    def add(self, customer_id: str, text: str, kind: str = "episodic"):
        self._index.upsert_records(
            namespace=customer_id,
            records=[{"_id": str(uuid.uuid4()), "text": text,
                      "kind": kind, "ts": time.time()}])

    def recall(self, customer_id: str, query: str, k: int = 3) -> list[str]:
        res = self._index.search(
            namespace=customer_id,
            query={"inputs": {"text": query}, "top_k": k})
        return [h["fields"]["text"] for h in res["result"]["hits"]]

    @property
    def backend(self) -> str:
        return f"pinecone (namespace per customer, {self.MODEL})"


def get_store():
    """Pinecone if configured, else local Chroma — the app doesn't care which."""
    if os.environ.get("PINECONE_API_KEY"):
        return PineconeStore()
    return ChromaStore()
