"""DeepInfra embeddings for the RAG corpora (OpenAI-compatible client).

Matches the WntHub corpus pipeline: ``BAAI/bge-base-en-v1.5`` (768-d). Used for
both ingestion (embedding chunks) and query-time embedding, so the same model
must be used for both or cosine distances are meaningless.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

DEEPINFRA_BASE_URL = "https://api.deepinfra.com/v1/openai"


async def embed_texts(
    texts: list[str], api_key: str, model: str = "BAAI/bge-base-en-v1.5"
) -> list[list[float]]:
    """Embed a batch of texts. Returns one vector per input, order-preserved."""
    if not texts:
        return []
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=api_key, base_url=DEEPINFRA_BASE_URL)
    resp = await client.embeddings.create(model=model, input=texts)
    # The API returns items with an `index`; sort to guarantee input order.
    items = sorted(resp.data, key=lambda d: d.index)
    return [list(d.embedding) for d in items]


async def embed_query(query: str, api_key: str, model: str = "BAAI/bge-base-en-v1.5") -> list[float]:
    """Embed a single query string."""
    vecs = await embed_texts([query], api_key, model)
    return vecs[0] if vecs else []


def to_pgvector(vec: list[float]) -> str:
    """Format a Python float list as a pgvector text literal: ``[1,2,3]``.

    pgvector accepts this cast to ``::vector`` — lets us use plain asyncpg
    without a vector-type adapter dependency.
    """
    return "[" + ",".join(repr(float(x)) for x in vec) + "]"
