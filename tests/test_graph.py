from finance_rag.documents import load_documents
from finance_rag.graph import KnowledgeGraph


def build_graph(sample_docs_dir):
    graph = KnowledgeGraph()
    for doc in load_documents(sample_docs_dir):
        graph.add_document(doc)
    return graph


def test_entities_link_to_documents(sample_docs_dir):
    graph = build_graph(sample_docs_dir)
    hits = graph.neighbors("Helios Foundry")
    targets = {(h["relation"], h["target"]) for h in hits}
    # mentioned in both the 10-K and the contract
    assert ("mentioned_in", "doc:orion_10k_fy2025_excerpt") in targets
    assert ("mentioned_in", "doc:helios_supply_agreement") in targets
    # declared cross-document relation
    assert ("party_to", "Helios Supply Agreement") in targets


def test_lookup_is_case_insensitive_substring(sample_docs_dir):
    graph = build_graph(sample_docs_dir)
    assert graph.neighbors("helios")
    assert graph.neighbors("HELIOS FOUNDRY")
    assert graph.neighbors("no-such-entity") == []


def test_save_and_load_roundtrip(sample_docs_dir, tmp_path):
    graph = build_graph(sample_docs_dir)
    path = tmp_path / "graph.json"
    graph.save(path)
    loaded = KnowledgeGraph.load(path)
    assert loaded.edges == graph.edges
