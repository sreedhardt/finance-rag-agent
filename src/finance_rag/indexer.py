"""Vector index on ChromaDB. Embeddings are computed by the injected embedder
and passed explicitly, so the store is embedding-provider agnostic.

Upserts are idempotent: chunk IDs are `{doc_id}#{chunk_index}` and re-ingesting
the same document replaces its chunks in place (no duplicates, no full rebuild)."""

from __future__ import annotations

from pathlib import Path

import chromadb

from . import config
from .documents import Document, lineage_metadata


class VectorIndex:
    def __init__(self, embedder, persist_dir: Path | None = None,
                 collection_name: str = config.COLLECTION_NAME):
        persist_dir = persist_dir or config.CHROMA_DIR
        persist_dir.mkdir(parents=True, exist_ok=True)
        self._embedder = embedder
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._collection = self._client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

    def upsert_document(self, doc: Document, chunks: list[str]) -> int:
        # Drop stale chunks first: if the new version has fewer chunks than the
        # old one, leftover trailing chunks would otherwise survive the upsert.
        existing = self._collection.get(where={"doc_id": doc.doc_id})
        if existing["ids"]:
            self._collection.delete(ids=existing["ids"])
        vectors = self._embedder.embed_documents(chunks)
        self._collection.upsert(
            ids=[f"{doc.doc_id}#{i}" for i in range(len(chunks))],
            embeddings=vectors,
            documents=chunks,
            metadatas=[lineage_metadata(doc, i) for i in range(len(chunks))],
        )
        return len(chunks)

    def search(self, query: str, k: int = 5, doc_type: str | None = None) -> list[dict]:
        where = {"doc_type": doc_type} if doc_type else None
        result = self._collection.query(
            query_embeddings=[self._embedder.embed_query(query)],
            n_results=min(k, max(self.count(), 1)),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        for chunk_id, text, meta, dist in zip(
            result["ids"][0], result["documents"][0],
            result["metadatas"][0], result["distances"][0],
        ):
            hits.append({
                "chunk_id": chunk_id,
                "text": text,
                "doc_id": meta["doc_id"],
                "title": meta["title"],
                "doc_type": meta["doc_type"],
                "source_sha256": meta["source_sha256"],
                "relevance": round(1.0 - dist, 4),
            })
        return hits

    def get_document_text(self, doc_id: str, max_chars: int = 12000) -> str | None:
        result = self._collection.get(where={"doc_id": doc_id})
        if not result["ids"]:
            return None
        ordered = sorted(
            zip(result["metadatas"], result["documents"]),
            key=lambda pair: pair[0]["chunk_index"],
        )
        return "\n\n".join(text for _, text in ordered)[:max_chars]

    def list_documents(self) -> list[dict]:
        result = self._collection.get(include=["metadatas"])
        docs: dict[str, dict] = {}
        for meta in result["metadatas"]:
            entry = docs.setdefault(meta["doc_id"], {
                "doc_id": meta["doc_id"],
                "title": meta["title"],
                "doc_type": meta["doc_type"],
                "n_chunks": 0,
                "ingested_at": meta["ingested_at"],
            })
            entry["n_chunks"] += 1
        return sorted(docs.values(), key=lambda d: d["doc_id"])

    def count(self) -> int:
        return self._collection.count()
