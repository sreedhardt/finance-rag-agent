"""Lightweight knowledge graph (Graph-RAG layer).

Documents declare entities and typed relations; the graph links entities to the
documents that mention them plus the declared entity↔entity edges. The agent
uses it to answer "what connects X to Y" questions and to discover which
documents to search next — the core Graph-RAG retrieval pattern, kept
dependency-free (swap for Neo4j/networkx at scale)."""

from __future__ import annotations

import json
from pathlib import Path

from . import config
from .documents import Document

MENTIONS = "mentioned_in"


class KnowledgeGraph:
    def __init__(self) -> None:
        self.edges: list[tuple[str, str, str]] = []

    def add_document(self, doc: Document) -> None:
        for entity in doc.entities:
            self._add(entity, MENTIONS, f"doc:{doc.doc_id}")
        for src, rel, dst in doc.relations:
            self._add(src, rel, dst)

    def _add(self, src: str, rel: str, dst: str) -> None:
        edge = (src, rel, dst)
        if edge not in self.edges:
            self.edges.append(edge)

    def neighbors(self, entity: str) -> list[dict]:
        """Case-insensitive substring match so 'Helios' finds 'Helios Foundry'."""
        needle = entity.strip().lower()
        results = []
        for src, rel, dst in self.edges:
            if needle in src.lower() or needle in dst.lower():
                results.append({"source": src, "relation": rel, "target": dst})
        return results

    def entity_names(self) -> list[str]:
        names = set()
        for src, _, dst in self.edges:
            names.add(src)
            if not dst.startswith("doc:"):
                names.add(dst)
        return sorted(names)

    def save(self, path: Path | None = None) -> None:
        path = path or config.GRAPH_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"edges": self.edges}, indent=2))

    @classmethod
    def load(cls, path: Path | None = None) -> "KnowledgeGraph":
        path = path or config.GRAPH_PATH
        graph = cls()
        if path.exists():
            data = json.loads(path.read_text())
            graph.edges = [tuple(edge) for edge in data.get("edges", [])]
        return graph
