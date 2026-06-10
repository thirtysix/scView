"""Abuse guards: input caps, the shared-secret gate, and deployment self-check."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from scview.config import Settings
from scview.core.guards import enforce_query_limits, require_access


def _req(headers: dict) -> SimpleNamespace:
    return SimpleNamespace(headers=headers)


def test_enforce_query_limits_caps_chars_words_history():
    s = Settings(MAX_QUERY_CHARS=100, MAX_QUERY_WORDS=10, MAX_HISTORY_MESSAGES=2)
    enforce_query_limits("a normal question", None, s)  # ok
    with pytest.raises(HTTPException) as e1:
        enforce_query_limits("x" * 200, None, s)
    assert e1.value.status_code == 413
    with pytest.raises(HTTPException):
        enforce_query_limits(" ".join(["w"] * 20), None, s)
    with pytest.raises(HTTPException):
        enforce_query_limits("ok", [1, 2, 3], s)


@pytest.mark.asyncio
async def test_require_access_is_noop_without_token():
    await require_access(_req({}), Settings())  # no ACCESS_TOKEN → allowed


@pytest.mark.asyncio
async def test_require_access_enforces_token_when_set():
    s = Settings(ACCESS_TOKEN="secret123")
    with pytest.raises(HTTPException) as e:
        await require_access(_req({}), s)
    assert e.value.status_code == 401
    with pytest.raises(HTTPException):
        await require_access(_req({"x-access-token": "wrong"}), s)
    # correct token via either header form is accepted (no raise)
    await require_access(_req({"authorization": "Bearer secret123"}), s)
    await require_access(_req({"x-access-token": "secret123"}), s)


def test_deployment_self_check():
    assert Settings().deployment_warnings() == []  # private → silent
    warns = Settings(DEPLOYMENT_MODE="public").deployment_warnings()
    assert any("ACCESS_TOKEN" in w for w in warns)
    assert any("localhost" in w for w in warns)
    assert Settings(DEPLOYMENT_MODE="public").is_public is True
    assert Settings().is_public is False
