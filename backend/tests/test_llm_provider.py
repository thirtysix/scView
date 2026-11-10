"""The configurable language-model provider: settings fallbacks, and every chat
client built against the configured base URL with the configured model."""

from __future__ import annotations

import pytest

from scview.config import DEEPINFRA_BASE_URL, Settings
from scview.core import llm

PM_URL = "http://privatemode-proxy:8080/v1"


def _s(**kw) -> Settings:
    base = dict(DEEPINFRA_API_KEY="", LLM_BASE_URL="", LLM_API_KEY="", LLM_CHAT_MODEL="",
                LLM_PROVIDER_NAME="", RAG_DATABASE_URL="")
    return Settings(**{**base, **kw})


def test_defaults_are_deepinfra_as_before():
    s = _s(DEEPINFRA_API_KEY="di")
    assert s.llm_base_url == DEEPINFRA_BASE_URL and s.llm_is_deepinfra
    assert s.llm_api_key == "di"
    assert s.chat_model == s.RAG_CHAT_MODEL
    assert s.llm_provider == "DeepInfra"


def test_rag_chat_model_still_sets_the_model():
    assert _s(RAG_CHAT_MODEL="zai-org/GLM-5.2").chat_model == "zai-org/GLM-5.2"
    assert _s(RAG_CHAT_MODEL="x", LLM_CHAT_MODEL=" glm-5.3-flash ").chat_model == "glm-5.3-flash"


def test_another_provider_never_gets_the_deepinfra_key():
    s = _s(DEEPINFRA_API_KEY="di", LLM_BASE_URL=PM_URL + "/")
    assert s.llm_base_url == PM_URL and not s.llm_is_deepinfra
    assert s.llm_api_key == ""
    assert s.llm_provider == "privatemode-proxy"
    s = _s(DEEPINFRA_API_KEY="di", LLM_BASE_URL=PM_URL, LLM_API_KEY="pm", LLM_PROVIDER_NAME="Privatemode")
    assert s.llm_api_key == "pm" and s.llm_provider == "Privatemode"


def test_llm_api_key_wins_on_deepinfra_too():
    assert _s(DEEPINFRA_API_KEY="di", LLM_API_KEY="other").llm_api_key == "other"


def test_a_lookalike_host_is_not_deepinfra():
    assert not _s(LLM_BASE_URL="https://deepinfra.com.evil.example/v1").llm_is_deepinfra
    assert _s(LLM_BASE_URL="https://api.deepinfra.com/v1/openai/").llm_is_deepinfra


@pytest.mark.parametrize("kw, reason", [
    (dict(), "no_database"),
    (dict(RAG_DATABASE_URL="pg://x"), "no_api_key"),
    (dict(RAG_DATABASE_URL="pg://x", DEEPINFRA_API_KEY="di"), None),
    # RAG embeds and reranks on DeepInfra: off with another chat provider, key or not.
    (dict(RAG_DATABASE_URL="pg://x", DEEPINFRA_API_KEY="di", LLM_BASE_URL=PM_URL, LLM_API_KEY="pm"),
     "provider"),
])
def test_rag_needs_deepinfra_as_the_provider(kw, reason):
    s = _s(**kw)
    assert s.rag_disabled_reason == reason and s.rag_enabled is (reason is None)


@pytest.mark.parametrize("model, body", [
    ("glm-5.3-flash", {"reasoning_effort": "low"}),
    ("glm-5.3", {"reasoning_effort": "low"}),
    ("glm-5.2", {"chat_template_kwargs": {"enable_thinking": False}}),
    ("gpt-oss-120b", {"reasoning_effort": "low"}),
    ("openai/gpt-oss-120b", {"reasoning_effort": "low"}),
])
def test_privatemode_model_names_get_their_rule(model, body):
    """Privatemode names its models without an org/ prefix: same rules as DeepInfra's."""
    assert llm.chat_extra(model) == {"extra_body": body}
    assert (llm.reply_room(model) > 0) is (body == {"reasoning_effort": "low"})


def _capture(monkeypatch, name):
    import openai

    made = []

    class _C:
        def __init__(self, **kw):
            made.append(kw)

    monkeypatch.setattr(openai, name, _C)
    return made


def test_clients_use_the_configured_base_url(monkeypatch):
    monkeypatch.setattr("scview.config.get_settings", lambda: _s(LLM_BASE_URL=PM_URL, LLM_API_KEY="pm"))
    a = _capture(monkeypatch, "AsyncOpenAI")
    llm.async_client("pm")
    o = _capture(monkeypatch, "OpenAI")
    llm.sync_client("pm")
    llm.sync_client("pm", "http://elsewhere/v1")
    assert a == [{"api_key": "pm", "base_url": PM_URL}]
    assert o == [{"api_key": "pm", "base_url": PM_URL}, {"api_key": "pm", "base_url": "http://elsewhere/v1"}]


def test_no_chat_call_site_names_a_provider_url():
    """Chat clients come from llm.async_client / sync_client only; the RAG embeddings
    (DeepInfra by design) import the URL from config."""
    from pathlib import Path

    import scview

    root = Path(scview.__file__).parent
    hits = [str(p.relative_to(root)) for p in root.rglob("*.py")
            if "deepinfra.com/v1/openai" in p.read_text() and p.name != "config.py"]
    assert hits == []
    users = [str(p.relative_to(root)) for p in root.rglob("*.py")
             if "AsyncOpenAI(" in p.read_text() or "OpenAI(" in p.read_text()]
    assert sorted(users) == ["core/llm.py", "core/rag/embeddings.py"]


class _Stream:
    """An async stream of chat chunks, as openai returns with stream=True."""

    def __init__(self, text):
        self.parts = [text]

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.parts:
            raise StopAsyncIteration
        from types import SimpleNamespace as N

        t = self.parts.pop()
        return N(choices=[N(delta=N(content=t, reasoning_content=None), finish_reason="stop")], usage=None)


async def test_routes_send_the_configured_model_and_key(client, test_settings, monkeypatch):
    """The co-pilot (no dataset) asks the configured provider with LLM_CHAT_MODEL and
    LLM_API_KEY, never the DeepInfra key."""
    import openai
    from types import SimpleNamespace as N

    test_settings.DEEPINFRA_API_KEY = "di"
    test_settings.LLM_BASE_URL = PM_URL
    test_settings.LLM_API_KEY = "pm"
    test_settings.LLM_CHAT_MODEL = "glm-5.3-flash"
    monkeypatch.setattr("scview.config.get_settings", lambda: test_settings)
    seen = []

    class _Completions:
        async def create(self, **kw):
            seen.append(kw)
            if kw.get("stream"):
                return _Stream("Hello.")
            return N(choices=[N(message=N(content='{"sources": ["app"]}'), finish_reason="stop")])

    class _Client:
        def __init__(self, **kw):
            seen.append(kw)
            self.chat = N(completions=_Completions())

    monkeypatch.setattr(openai, "AsyncOpenAI", _Client)
    r = await client.post("/api/v1/assistant/chat-stream", json={"query": "what can scView do?"})
    assert r.status_code == 200
    clients = [k for k in seen if "base_url" in k]
    calls = [k for k in seen if "model" in k]
    assert clients and all(k == {"api_key": "pm", "base_url": PM_URL} for k in clients)
    assert calls and all(k["model"] == "glm-5.3-flash" for k in calls)
    # GLM-5.3(-Flash) gets reasoning_effort low (enable_thinking false backfires).
    assert all(k.get("extra_body") == {"reasoning_effort": "low"} for k in calls)
