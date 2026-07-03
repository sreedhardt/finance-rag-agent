"""Ingestion pipeline: extract → validate (quality gate) → embed+index →
build graph → seed structured db.

Each stage is a plain function so the same code runs from the CLI, Streamlit,
or as individual Airflow tasks (see orchestration/finance_rag_dag.py)."""

from __future__ import annotations

import time
from pathlib import Path

from . import config, findb
from .documents import Document, chunk_text, load_documents, validate_chunks
from .graph import KnowledgeGraph
from .indexer import VectorIndex


def stage_extract(raw_dir: Path | None = None) -> list[Document]:
    docs = load_documents(raw_dir)
    if not docs:
        raise RuntimeError(f"no documents found in {raw_dir or config.RAW_DATA_DIR}")
    return docs


def stage_validate(docs: list[Document]) -> dict[str, list[str]]:
    """Quality gate: chunk every document and fail fast on hard errors."""
    chunked: dict[str, list[str]] = {}
    failures = []
    for doc in docs:
        chunks = chunk_text(doc.text)
        report = validate_chunks(doc.doc_id, chunks)
        for warning in report.warnings:
            print(f"  [warn] {doc.doc_id}: {warning}")
        if not report.ok:
            failures.append(f"{doc.doc_id}: {'; '.join(report.errors)}")
        chunked[doc.doc_id] = chunks
    if failures:
        raise ValueError("quality gate failed — " + " | ".join(failures))
    return chunked


def stage_index(docs: list[Document], chunked: dict[str, list[str]],
                index: VectorIndex) -> int:
    total = 0
    for doc in docs:
        n = index.upsert_document(doc, chunked[doc.doc_id])
        print(f"  indexed {doc.doc_id}: {n} chunks")
        total += n
    return total


def stage_graph(docs: list[Document]) -> KnowledgeGraph:
    graph = KnowledgeGraph()
    for doc in docs:
        graph.add_document(doc)
    graph.save()
    return graph


def run_ingestion(raw_dir: Path | None = None, embedder=None) -> dict:
    """Full pipeline. Returns a run manifest (what was processed, counts, timing)."""
    started = time.time()
    if embedder is None:
        from .embedder import GeminiEmbedder
        embedder = GeminiEmbedder()

    print("[1/5] extract")
    docs = stage_extract(raw_dir)
    print(f"  loaded {len(docs)} documents")

    print("[2/5] validate (quality gate)")
    chunked = stage_validate(docs)

    print("[3/5] embed + index")
    index = VectorIndex(embedder)
    n_chunks = stage_index(docs, chunked, index)

    print("[4/5] knowledge graph")
    graph = stage_graph(docs)
    print(f"  {len(graph.edges)} edges")

    print("[5/5] structured financials db")
    db_path = findb.init_financials_db()
    print(f"  seeded {db_path}")

    manifest = {
        "documents": [d.doc_id for d in docs],
        "chunks_indexed": n_chunks,
        "graph_edges": len(graph.edges),
        "elapsed_s": round(time.time() - started, 2),
    }
    print(f"done in {manifest['elapsed_s']}s")
    return manifest
