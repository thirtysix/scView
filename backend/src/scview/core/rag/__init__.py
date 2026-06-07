"""RAG co-pilot: dual-corpus (literature + tutorials) retrieval over pgvector.

Two routed corpora answer different question types:
  - ``literature`` — PubMed abstracts → biological / evidential questions (cite PMID).
  - ``tutorials``  — scanpy/Seurat/best-practices docs → methods / how-to questions (cite doc+section).

A query router picks the corpus(es); hybrid retrieval (vector + full-text) returns
chunks; the chunks are folded into the co-pilot's grounding context via the
``extra_context`` hook in ``core/assistant.py``. All of this degrades gracefully:
if ``RAG_DATABASE_URL`` / ``DEEPINFRA_API_KEY`` are unset the co-pilot simply
grounds in in-app facts only.
"""
