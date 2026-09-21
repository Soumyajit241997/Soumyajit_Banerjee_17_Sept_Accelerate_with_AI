"""Semantic memory (ChromaDB) for cross-run knowledge.

Same graceful-degradation pattern as IDAMP: if chromadb isn't installed or
fails to initialise, memory operations become no-ops rather than crashing
the pipeline.
"""
from __future__ import annotations

from core.config import CHROMA_DIR

_client = None
_collection = None


def _get_collection():
    global _client, _collection
    if _collection is not None:
        return _collection
    try:
        import chromadb

        _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = _client.get_or_create_collection("radar_memory")
        return _collection
    except Exception:
        return None


def remember(doc_id: str, document: str, metadata: dict | None = None) -> None:
    collection = _get_collection()
    if collection is None:
        return
    try:
        collection.upsert(ids=[doc_id], documents=[document], metadatas=[metadata or {}])
    except Exception:
        pass


def recall(query: str, n_results: int = 3) -> list[dict]:
    collection = _get_collection()
    if collection is None:
        return []
    try:
        result = collection.query(query_texts=[query], n_results=n_results)
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        return [{"document": d, "metadata": m} for d, m in zip(docs, metas)]
    except Exception:
        return []
