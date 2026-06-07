"""pgvector store for the RAG corpora (Neon/Postgres via asyncpg).

One table, ``scview_rag_chunks``, holds both corpora distinguished by a
``corpus`` column — simplest infra while still allowing per-corpus chunking at
ingest time and per-corpus retrieval at query time. Hybrid retrieval combines
vector cosine similarity with Postgres full-text ``ts_rank`` (weights from
settings), mirroring the WntHub ``rag-query.js`` approach.

Everything is lazy and optional: if no pool can be created (no DSN), callers
should treat RAG as disabled and fall back to in-app grounding.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

_TABLE = "scview_rag_chunks"
_pool: Any = None  # asyncpg.Pool | None


async def get_pool(dsn: str):
    """Lazily create (once) and return the asyncpg connection pool."""
    global _pool
    if _pool is None:
        import asyncpg

        _pool = await asyncpg.create_pool(dsn, min_size=1, max_size=4)
        logger.info("RAG: created asyncpg pool")
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


async def init_db(dsn: str, dim: int = 768) -> None:
    """Create the pgvector extension, table, and indexes (idempotent)."""
    pool = await get_pool(dsn)
    async with pool.acquire() as conn:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        await conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {_TABLE} (
                id          bigserial PRIMARY KEY,
                corpus      text NOT NULL,
                doc_id      text NOT NULL,
                chunk_idx   int  NOT NULL,
                content     text NOT NULL,
                embedding   vector({dim}),
                metadata    jsonb NOT NULL DEFAULT '{{}}'::jsonb,
                ts          tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
                created_at  timestamptz NOT NULL DEFAULT now(),
                UNIQUE (corpus, doc_id, chunk_idx)
            );
            """
        )
        # HNSW for fast cosine ANN; GIN for full-text; btree for corpus filter.
        await conn.execute(
            f"CREATE INDEX IF NOT EXISTS {_TABLE}_emb_hnsw "
            f"ON {_TABLE} USING hnsw (embedding vector_cosine_ops);"
        )
        await conn.execute(
            f"CREATE INDEX IF NOT EXISTS {_TABLE}_ts_gin ON {_TABLE} USING gin (ts);"
        )
        await conn.execute(
            f"CREATE INDEX IF NOT EXISTS {_TABLE}_corpus ON {_TABLE} (corpus);"
        )
    logger.info("RAG: schema ready (dim=%d)", dim)


async def upsert_chunks(dsn: str, corpus: str, rows: list[dict[str, Any]]) -> int:
    """Insert/replace chunks for one corpus.

    Each row: ``{doc_id, chunk_idx, content, embedding: list[float], metadata: dict}``.
    Returns the number of rows written.
    """
    if not rows:
        return 0
    from scview.core.rag.embeddings import to_pgvector

    pool = await get_pool(dsn)
    records = [
        (
            corpus,
            str(r["doc_id"]),
            int(r["chunk_idx"]),
            r["content"],
            to_pgvector(r["embedding"]),
            json.dumps(r.get("metadata", {})),
        )
        for r in rows
    ]
    async with pool.acquire() as conn:
        await conn.executemany(
            f"""
            INSERT INTO {_TABLE} (corpus, doc_id, chunk_idx, content, embedding, metadata)
            VALUES ($1, $2, $3, $4, $5::vector, $6::jsonb)
            ON CONFLICT (corpus, doc_id, chunk_idx)
            DO UPDATE SET content = EXCLUDED.content,
                          embedding = EXCLUDED.embedding,
                          metadata = EXCLUDED.metadata;
            """,
            records,
        )
    return len(records)


async def hybrid_search(
    dsn: str,
    *,
    corpora: list[str],
    query_vec: list[float],
    query_text: str,
    top_k: int = 6,
    vector_weight: float = 0.7,
    text_weight: float = 0.3,
) -> list[dict[str, Any]]:
    """Hybrid vector + full-text retrieval, filtered to the given corpora.

    Score = w_v · cosine_similarity + w_t · normalized_ts_rank. Returns rows with
    ``content``, ``metadata`` (dict), ``corpus``, and ``score``.
    """
    if not corpora:
        return []
    from scview.core.rag.embeddings import to_pgvector

    pool = await get_pool(dsn)
    qv = to_pgvector(query_vec)
    sql = f"""
        WITH scored AS (
            SELECT
                corpus,
                content,
                metadata,
                1 - (embedding <=> $1::vector)                              AS vsim,
                ts_rank(ts, plainto_tsquery('english', $2))                AS trank
            FROM {_TABLE}
            WHERE corpus = ANY($3::text[])
              AND embedding IS NOT NULL
        ),
        norm AS (
            SELECT *,
                   COALESCE(trank / NULLIF(MAX(trank) OVER (), 0), 0) AS tnorm
            FROM scored
        )
        SELECT corpus, content, metadata,
               ($4 * vsim + $5 * tnorm) AS score
        FROM norm
        ORDER BY score DESC
        LIMIT $6;
    """
    async with pool.acquire() as conn:
        recs = await conn.fetch(
            sql, qv, query_text, corpora, vector_weight, text_weight, top_k
        )
    out = []
    for r in recs:
        md = r["metadata"]
        if isinstance(md, str):
            try:
                md = json.loads(md)
            except json.JSONDecodeError:
                md = {}
        out.append(
            {
                "corpus": r["corpus"],
                "content": r["content"],
                "metadata": md or {},
                "score": float(r["score"]),
            }
        )
    return out


async def corpus_counts(dsn: str) -> dict[str, int]:
    """Return chunk counts per corpus (for health/status)."""
    pool = await get_pool(dsn)
    async with pool.acquire() as conn:
        recs = await conn.fetch(f"SELECT corpus, count(*) AS n FROM {_TABLE} GROUP BY corpus;")
    return {r["corpus"]: int(r["n"]) for r in recs}
