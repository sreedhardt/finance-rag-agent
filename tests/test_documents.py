from finance_rag.documents import (
    chunk_text,
    lineage_metadata,
    load_documents,
    validate_chunks,
)


def test_load_documents_parses_front_matter(sample_docs_dir):
    docs = {d.doc_id: d for d in load_documents(sample_docs_dir)}
    assert len(docs) == 3
    filing = docs["orion_10k_fy2025_excerpt"]
    assert filing.doc_type == "sec_filing"
    assert "Helios Foundry" in filing.entities
    assert ("Helios Foundry", "supplies_wafers_to", "Orion Semiconductor") in filing.relations
    assert "---" not in filing.text[:20]  # front matter stripped from body
    assert len(filing.sha256) == 64


def test_chunk_text_respects_max_and_overlap():
    paragraphs = "\n\n".join(f"Paragraph {i} " + "x" * 200 for i in range(20))
    chunks = chunk_text(paragraphs, max_chars=500, overlap=100)
    assert len(chunks) > 1
    # every chunk within max + overlap allowance
    assert all(len(c) <= 500 + 101 for c in chunks)
    # overlap: each chunk after the first starts with the tail of its predecessor
    for prev, cur in zip(chunks, chunks[1:]):
        assert cur.startswith(prev[-100:])


def test_chunk_text_hard_splits_oversized_paragraph():
    chunks = chunk_text("y" * 3000, max_chars=1000, overlap=100)
    assert len(chunks) >= 3
    assert "".join(c.replace("\n", "") for c in chunks).count("y") >= 3000


def test_validate_chunks_flags_duplicates_and_empties():
    report = validate_chunks("doc", ["same chunk of text here", "same chunk of text here"])
    assert not report.ok
    assert any("duplicates" in e for e in report.errors)

    report = validate_chunks("doc", [])
    assert not report.ok

    report = validate_chunks("doc", ["a perfectly reasonable chunk of financial text"])
    assert report.ok


def test_lineage_metadata_is_audit_complete(sample_docs_dir):
    doc = load_documents(sample_docs_dir)[0]
    meta = lineage_metadata(doc, 3)
    for key in ("doc_id", "source_path", "source_sha256", "ingested_at", "chunk_index"):
        assert meta[key] not in (None, "")
    assert meta["chunk_index"] == 3
