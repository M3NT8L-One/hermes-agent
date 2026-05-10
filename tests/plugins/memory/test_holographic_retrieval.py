"""Tests for local Holographic fact retrieval."""

from plugins.memory.holographic.retrieval import FactRetriever
from plugins.memory.holographic.store import MemoryStore


def test_relaxed_fts_query_keeps_useful_terms_and_drops_noise():
    query = "What do you remember about M3NT8L/Tyson's trading-dashboard setup?"

    relaxed = FactRetriever._relaxed_fts_query(query)

    assert '"m3nt8l"' in relaxed
    assert '"tyson"' in relaxed
    assert '"trading"' in relaxed
    assert '"dashboard"' in relaxed
    assert '"what"' not in relaxed
    assert " OR " in relaxed


def test_search_falls_back_to_relaxed_query_for_natural_language(tmp_path):
    store = MemoryStore(db_path=tmp_path / "memory_store.db", default_trust=0.8)
    retriever = FactRetriever(store=store, hrr_weight=0.0)
    store.add_fact(
        "Trading dashboard private mobile URL uses Tailscale over phase-d.html.",
        category="project",
        tags="dashboard tailscale trading",
    )

    results = retriever.search(
        "What do you remember about Tyson's trading dashboard mobile link?",
        min_trust=0.1,
        limit=3,
    )

    assert results
    assert "Trading dashboard private mobile URL" in results[0]["content"]
