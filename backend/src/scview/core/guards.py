"""Lightweight abuse guards: input caps + an optional shared-secret gate.

These are the cheap, always-correct defenses (input validation, a coarse access
token). They are NOT a substitute for per-user auth + rate limiting + quotas,
which a real public/multi-user deployment still needs — see SECURITY.md. The
deployment-mode self-check (config.deployment_warnings) makes that gap loud.
"""

from __future__ import annotations

import hmac

from fastapi import Depends, HTTPException, Request

from scview.config import Settings
from scview.dependencies import get_settings_dep


def enforce_query_limits(
    query: str,
    history: list | None,
    settings: Settings,
) -> None:
    """Reject oversized LLM-bound input before it inflates token cost. Always on
    (both deployment modes) — this is hygiene, not a public-only concern."""
    q = query or ""
    if len(q) > settings.MAX_QUERY_CHARS:
        raise HTTPException(
            status_code=413,
            detail=f"Query too long (max {settings.MAX_QUERY_CHARS} characters).",
        )
    if len(q.split()) > settings.MAX_QUERY_WORDS:
        raise HTTPException(
            status_code=413,
            detail=f"Query has too many words (max {settings.MAX_QUERY_WORDS}).",
        )
    if history is not None and len(history) > settings.MAX_HISTORY_MESSAGES:
        raise HTTPException(
            status_code=413,
            detail=f"Conversation history too long (max {settings.MAX_HISTORY_MESSAGES} messages).",
        )


def _present_token(request: Request) -> str:
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return request.headers.get("x-access-token", "").strip()


async def require_access(
    request: Request,
    settings: Settings = Depends(get_settings_dep),
) -> None:
    """Coarse shared-secret gate. A no-op unless ACCESS_TOKEN is configured; when
    set, every guarded request must present it. One secret for everyone (pair with
    a reverse proxy for real auth), but it keeps an exposed instance from being
    wide open. Constant-time compare to avoid a timing oracle."""
    expected = settings.ACCESS_TOKEN
    if not expected:
        return
    if not hmac.compare_digest(_present_token(request), expected):
        raise HTTPException(status_code=401, detail="Missing or invalid access token.")
