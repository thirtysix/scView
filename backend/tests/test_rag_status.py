"""RAG corpus coverage for the co-pilot's Sources note: read-only, cached, and
never echoes a database error to the browser. No real database is touched."""

from __future__ import annotations

import contextlib
from datetime import datetime, timezone

import pytest

import scview.api.v1.assistant as api
from scview.core.rag import store

URL = "/api/v1/assistant/rag-status"


@pytest.fixture(autouse=True)
def _fresh_cache():
    api._rag_status_cache.clear()
    yield
    api._rag_status_cache.clear()


async def test_disabled_says_why(client, test_settings):
    body = (await client.get(URL)).json()
    assert body == {"enabled": False, "counts": {}, "corpora": {}, "reason": "no_database"}
    test_settings.RAG_DATABASE_URL = "postgresql://example/db"
    assert (await client.get(URL)).json()["reason"] == "no_api_key"


async def test_enabled_reports_documents_and_is_cached(client, test_settings, monkeypatch):
    calls = []
    stats = {
        "literature": {"chunks": 900, "documents": 310, "last_added": "2026-06-07T10:00:00+00:00"},
        "tutorials": {"chunks": 120, "documents": 14, "last_added": None},
    }

    async def fake_stats(dsn):
        calls.append(dsn)
        return stats

    monkeypatch.setattr(store, "corpus_stats", fake_stats)
    test_settings.RAG_DATABASE_URL = "postgresql://example/db"
    test_settings.DEEPINFRA_API_KEY = "k"
    body = (await client.get(URL)).json()
    assert body["enabled"] is True and body["corpora"] == stats
    assert body["counts"] == {"literature": 900, "tutorials": 120}
    await client.get(URL)
    assert len(calls) == 1  # the second open doesn't query the database


async def test_database_errors_stay_in_the_log(client, test_settings, monkeypatch):
    async def broken(dsn):
        raise RuntimeError("connect failed for postgresql://user:hunter2@host/db")

    monkeypatch.setattr(store, "corpus_stats", broken)
    test_settings.RAG_DATABASE_URL = "postgresql://user:hunter2@host/db"
    test_settings.DEEPINFRA_API_KEY = "k"
    r = await client.get(URL)
    assert r.json() == {"enabled": True, "counts": {}, "corpora": {}, "error": "database unreachable"}
    assert "hunter2" not in r.text
    assert api._rag_status_cache == {}  # a failure isn't cached


async def test_corpus_stats_is_one_read_only_select(monkeypatch):
    seen: list[str] = []
    when = datetime(2026, 6, 7, 10, 0, tzinfo=timezone.utc)

    class _Conn:
        async def fetch(self, sql):
            seen.append(sql)
            return [{"corpus": "literature", "chunks": 9, "docs": 3, "last": when},
                    {"corpus": "tutorials", "chunks": 2, "docs": 1, "last": None}]

    class _Pool:
        @contextlib.asynccontextmanager
        async def acquire(self):
            yield _Conn()

    async def fake_pool(dsn):
        return _Pool()

    monkeypatch.setattr(store, "get_pool", fake_pool)
    out = await store.corpus_stats("dsn")
    assert out == {
        "literature": {"chunks": 9, "documents": 3, "last_added": when.isoformat()},
        "tutorials": {"chunks": 2, "documents": 1, "last_added": None},
    }
    (sql,) = seen
    assert sql.lstrip().upper().startswith("SELECT") and "DISTINCT doc_id" in sql


async def test_llm_status_follows_the_api_key(client, test_settings):
    url = "/api/v1/assistant/llm-status"
    test_settings.DEEPINFRA_API_KEY = ""
    assert (await client.get(url)).json() == {"available": False}
    test_settings.DEEPINFRA_API_KEY = "k"
    assert (await client.get(url)).json() == {"available": True}
