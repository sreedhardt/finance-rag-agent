"""Document loading, chunking, and data-quality validation.

Source documents are Markdown/text files with optional YAML-style front matter
declaring entities and relations (used to build the knowledge graph), or PDFs.
In production the entity/relation extraction step would be an LLM extraction
pass; here it is declarative so the pipeline is deterministic and testable.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from . import config

FRONT_MATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)


@dataclass
class Document:
    doc_id: str
    title: str
    doc_type: str
    text: str
    source_path: str
    sha256: str
    entities: list[str] = field(default_factory=list)
    relations: list[tuple[str, str, str]] = field(default_factory=list)


@dataclass
class QualityReport:
    doc_id: str
    n_chunks: int
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _parse_front_matter(raw: str) -> tuple[dict, str]:
    """Parse a minimal front-matter block: `key: value` lines between --- fences."""
    match = FRONT_MATTER_RE.match(raw)
    if not match:
        return {}, raw
    meta = {}
    for line in match.group(1).splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            meta[key.strip().lower()] = value.strip()
    return meta, raw[match.end():]


def _load_text_file(path: Path) -> Document:
    raw = path.read_text(encoding="utf-8")
    meta, body = _parse_front_matter(raw)
    entities = [e.strip() for e in meta.get("entities", "").split(";") if e.strip()]
    relations = []
    for rel in meta.get("relations", "").split(";"):
        parts = [p.strip() for p in rel.split("|")]
        if len(parts) == 3 and all(parts):
            relations.append(tuple(parts))
    return Document(
        doc_id=path.stem,
        title=meta.get("title", path.stem),
        doc_type=meta.get("doc_type", "document"),
        text=body.strip(),
        source_path=str(path),
        sha256=hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        entities=entities,
        relations=relations,
    )


def _load_pdf_file(path: Path) -> Document:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
    return Document(
        doc_id=path.stem,
        title=path.stem,
        doc_type="pdf",
        text=text.strip(),
        source_path=str(path),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def load_documents(raw_dir: Path | None = None) -> list[Document]:
    raw_dir = raw_dir or config.RAW_DATA_DIR
    docs = []
    for path in sorted(raw_dir.glob("*")):
        if path.suffix.lower() in {".md", ".txt"}:
            docs.append(_load_text_file(path))
        elif path.suffix.lower() == ".pdf":
            docs.append(_load_pdf_file(path))
    return docs


def chunk_text(
    text: str,
    max_chars: int | None = None,
    overlap: int | None = None,
) -> list[str]:
    """Paragraph-aware chunking: pack paragraphs up to max_chars, hard-split
    oversized paragraphs, then prepend an overlap tail from the previous chunk
    so retrieval doesn't lose context at chunk boundaries."""
    max_chars = max_chars or config.CHUNK_MAX_CHARS
    overlap = overlap if overlap is not None else config.CHUNK_OVERLAP_CHARS

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 2 <= max_chars:
            current = f"{current}\n\n{para}" if current else para
            continue
        if current:
            chunks.append(current)
        while len(para) > max_chars:
            chunks.append(para[:max_chars])
            para = para[max_chars - overlap:]
        current = para
    if current:
        chunks.append(current)

    if overlap <= 0:
        return chunks
    with_overlap = []
    for i, chunk in enumerate(chunks):
        if i > 0:
            chunk = chunks[i - 1][-overlap:] + "\n" + chunk
        with_overlap.append(chunk)
    return with_overlap


def validate_chunks(doc_id: str, chunks: list[str], min_chars: int = 40) -> QualityReport:
    """Quality gate run before anything is indexed (Great Expectations-style
    expectations, kept dependency-free). Errors block indexing; warnings don't."""
    report = QualityReport(doc_id=doc_id, n_chunks=len(chunks))
    if not chunks:
        report.errors.append("document produced zero chunks")
        return report
    seen: dict[str, int] = {}
    for i, chunk in enumerate(chunks):
        if not chunk.strip():
            report.errors.append(f"chunk {i} is empty")
            continue
        if len(chunk) < min_chars:
            report.warnings.append(f"chunk {i} is short ({len(chunk)} chars)")
        digest = hashlib.sha256(chunk.encode("utf-8")).hexdigest()
        if digest in seen:
            report.errors.append(f"chunk {i} duplicates chunk {seen[digest]}")
        else:
            seen[digest] = i
    return report


def lineage_metadata(doc: Document, chunk_index: int) -> dict:
    """Audit-ready lineage attached to every indexed chunk: where it came from,
    which exact source version (sha256) produced it, and when."""
    return {
        "doc_id": doc.doc_id,
        "title": doc.title,
        "doc_type": doc.doc_type,
        "chunk_index": chunk_index,
        "source_path": doc.source_path,
        "source_sha256": doc.sha256,
        "ingested_at": datetime.now(timezone.utc).isoformat(),
    }
