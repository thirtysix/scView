"""Offline unit tests for RAG pieces that need no DB or network:
the heuristic query router and the chunkers. (Retrieval/ingestion against
pgvector are exercised manually once RAG_DATABASE_URL is set.)
"""

from __future__ import annotations

from scview.core.rag.router import heuristic_route
from scview.core.rag.ingest import chunk_sentences, chunk_sections
from scview.core.rag.embeddings import to_pgvector


def test_router_methods_question_to_tutorials():
    r = heuristic_route("What clustering resolution should I use and why?")
    assert r.corpora == ["tutorials"]


def test_router_biology_question_to_literature():
    r = heuristic_route("What is known about the role of pDCs in lupus?")
    assert r.corpora == ["literature"]


def test_router_ambiguous_to_both():
    r = heuristic_route("Tell me about my dataset")
    assert set(r.corpora) == {"tutorials", "literature"}


def test_chunk_sentences_respects_budget_and_covers_text():
    text = " ".join(f"Sentence number {i} about cells." for i in range(60))
    chunks = chunk_sentences(text, max_chars=200, overlap_chars=40)
    assert len(chunks) > 1
    assert all(len(c) <= 260 for c in chunks)  # budget + overlap slack
    assert "Sentence number 0" in chunks[0]


def test_chunk_sections_splits_on_blank_lines():
    text = "Intro paragraph.\n\nSecond section with more detail.\n\nThird block here."
    chunks = chunk_sections(text, max_chars=40, overlap_chars=0)
    assert len(chunks) >= 2


def test_to_pgvector_format():
    assert to_pgvector([1.0, 2.5, -3.0]) == "[1.0,2.5,-3.0]"
