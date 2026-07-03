from finance_rag.documents import chunk_text, load_documents
from finance_rag.indexer import VectorIndex


def build_index(fake_embedder, sample_docs_dir, tmp_path):
    index = VectorIndex(fake_embedder, persist_dir=tmp_path / "chroma")
    for doc in load_documents(sample_docs_dir):
        index.upsert_document(doc, chunk_text(doc.text))
    return index


def test_upsert_and_search_roundtrip(fake_embedder, sample_docs_dir, tmp_path):
    index = build_index(fake_embedder, sample_docs_dir, tmp_path)
    assert index.count() > 0
    assert len(index.list_documents()) == 3

    hits = index.search("payment terms", k=3)
    assert len(hits) == 3
    for hit in hits:
        assert hit["chunk_id"].split("#")[0] == hit["doc_id"]
        assert hit["source_sha256"]  # lineage travels with every result


def test_reingest_is_idempotent(fake_embedder, sample_docs_dir, tmp_path):
    index = build_index(fake_embedder, sample_docs_dir, tmp_path)
    before = index.count()
    doc = load_documents(sample_docs_dir)[0]
    index.upsert_document(doc, chunk_text(doc.text))
    assert index.count() == before  # no duplicate chunks on re-ingest


def test_shrinking_document_leaves_no_stale_chunks(fake_embedder, sample_docs_dir, tmp_path):
    index = build_index(fake_embedder, sample_docs_dir, tmp_path)
    doc = load_documents(sample_docs_dir)[0]
    index.upsert_document(doc, ["only one chunk now remains for this document"])
    per_doc = {d["doc_id"]: d["n_chunks"] for d in index.list_documents()}
    assert per_doc[doc.doc_id] == 1


def test_doc_type_filter(fake_embedder, sample_docs_dir, tmp_path):
    index = build_index(fake_embedder, sample_docs_dir, tmp_path)
    hits = index.search("liability cap", k=5, doc_type="contract")
    assert hits
    assert all(h["doc_type"] == "contract" for h in hits)
