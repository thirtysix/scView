"""The language-model client and per-model request settings for the chat calls.

``async_client`` / ``sync_client`` build every chat client against the configured
provider (``Settings.llm_base_url``: DeepInfra by default, or e.g. a Privatemode
proxy); nothing else names a provider URL. The RAG embeddings and reranker are the
exception: they stay on DeepInfra (core/rag/).

scView's calls are short, grounded and latency-sensitive, so thinking is turned off
wherever the model allows it. Many models a user may set as ``LLM_CHAT_MODEL`` or
``annotation_llm_model`` think by default: 10-100x the output tokens, and a tight
``max_tokens`` can be spent before the answer starts (Qwen3.5-397B returned empty
content). How each model turns it off differs, and the wrong switch can make it worse,
so the settings come from probes, not from a guess: ``_RULES``.

Probed on DeepInfra 2026-11-06/07 (research/batch42/annotation-bench NOTES.md and
bench/models.py; GLM-5.3-Flash and gpt-oss re-probed 2026-11-07 with a 25-cluster
annotation prompt at the annotator's 1,200-token budget, $0.004 in all):

- ``chat_template_kwargs.enable_thinking: false`` gives a clean answer: DeepSeek V4.x,
  GLM-5.2, Kimi-K2.6, Qwen3.5 / 3.6 / 3.8, MiniMax-M3, MiMo-V2.6-Pro, Nemotron-3-Ultra,
  Seed-2.0-pro / mini, Hy4-preview (Hy4 and Qwen3.6 otherwise returned EMPTY content).
- Nothing to send: Qwen3-235B-A22B-Instruct-2507 (instruct only), DeepSeek-V3.2,
  gemma-4 (none of them think), Claude, Gemini (no visible thinking).
- GLM-5.3: enable_thinking false (or reasoning_effort none) LEAKS the reasoning into
  the content; reasoning_effort low gave a clean answer.
- GLM-5.3-Flash: recorded as not thinking, but it does on longer prompts (a 25-cluster
  lung reply spent 2,070 reasoning chars and was cut at 1,200 tokens). enable_thinking
  false and reasoning_effort none make it WORSE: 3/3 and 1/1 probes spent the whole
  1,200 tokens reasoning (~3,000 chars) and returned no content. reasoning_effort low:
  0-5 reasoning chars, 25/25 lines, 3/3 probes (the default: 5-12 chars on this prompt).
- gpt-oss-120b: thinking cannot be turned off (enable_thinking -> 429); reasoning_effort
  low still reasons (~2,500-2,700 chars): 22/25 lines at 1,200 tokens (cut), 25/25 with
  2,000 more (1,314 tokens used). Hence ``reply_room``.

Privatemode (through privatemode-proxy) serves glm-5.3-flash, glm-5.3, glm-5.2 and
gpt-oss-120b under bare names, and the same prefixes match them. Smoke test 2026-11-11
with these settings: glm-5.3-flash streamed chat + router + follow-ups, advisor JSON
(14 suggestions), annotation (5/5 clusters with CL ids) and claims JSON all clean;
glm-5.2 (enable_thinking false) and gpt-oss-120b (low) gave clean claims JSON.

Unknown models get nothing extra, as before. Every setting goes in ``extra_body``, so
the request is the same with any openai SDK version.
"""

from __future__ import annotations

import copy
import time
from collections.abc import Callable
from typing import TypeVar

_T = TypeVar("_T")

_OFF = {"chat_template_kwargs": {"enable_thinking": False}}
_LOW = {"reasoning_effort": "low"}

# (prefix of the lower-cased model name after the provider "org/", request body
# additions, extra max_tokens for a model that still reasons). First match wins, so
# the exceptions come before their family.
_RULES: tuple[tuple[str, dict, int], ...] = (
    ("deepseek-v3.2", {}, 0),                 # does not think
    ("qwen3-235b-a22b-instruct", {}, 0),      # instruct only: never send enable_thinking
    ("glm-5.3", _LOW, 1000),                  # GLM-5.3 and -Flash: enable_thinking false backfires
    ("gpt-oss", _LOW, 2000),                  # cannot be turned off
    ("deepseek", _OFF, 0),                    # V4.x (and every other DeepSeek, as before)
    ("glm-", _OFF, 0),
    ("kimi-", _OFF, 0),
    ("qwen3.", _OFF, 0),                      # Qwen3.5 / 3.6 / 3.8
    ("minimax-m", _OFF, 0),
    ("mimo-", _OFF, 0),
    ("nvidia-nemotron-3", _OFF, 0),
    ("seed-2", _OFF, 0),
    ("hy4", _OFF, 0),
)


def async_client(api_key: str, base_url: str | None = None):
    """An ``openai.AsyncOpenAI`` for the configured provider (``base_url`` overrides it)."""
    from openai import AsyncOpenAI

    return AsyncOpenAI(api_key=api_key, base_url=base_url or _base_url())


def sync_client(api_key: str, base_url: str | None = None):
    """An ``openai.OpenAI`` for the configured provider (``base_url`` overrides it)."""
    from openai import OpenAI

    return OpenAI(api_key=api_key, base_url=base_url or _base_url())


def _base_url() -> str:
    from scview.config import get_settings

    return get_settings().llm_base_url


def _rule(model: str | None) -> tuple[dict, int]:
    name = (model or "").strip().lower().rsplit("/", 1)[-1]
    if name:
        for prefix, body, room in _RULES:
            if name.startswith(prefix):
                return body, room
    return {}, 0


def chat_extra(model: str) -> dict:
    """Extra keyword arguments for ``chat.completions.create`` with ``model``: the
    model's thinking switch (see the module docstring), or {} for a model without one."""
    body, _room = _rule(model)
    return {"extra_body": copy.deepcopy(body)} if body else {}


def reply_room(model: str) -> int:
    """Extra ``max_tokens`` for a model that still reasons with ``chat_extra``'s settings
    (gpt-oss, GLM-5.3): add it to a call's budget, or the reasoning can use up the
    budget before the answer. 0 for every other model."""
    return _rule(model)[1]


# Waits (seconds) before each retry of a 429 in a step that runs in the background
# (the annotator). DeepInfra answers 429 "Model busy" for a model under load
# (DeepSeek-V4.1-Flash, 2026-11-08); the openai client's own 2 retries are over within
# about 2 s, and the step then failed.
BUSY_WAITS: tuple[float, ...] = (5.0, 15.0, 30.0)


def is_busy(exc: BaseException) -> bool:
    """A 429 from the provider ("Model busy" or a rate limit): worth asking again later."""
    return getattr(exc, "status_code", None) == 429


def busy_retry(call: Callable[[], _T], waits: tuple[float, ...] | None = None,
               sleep: Callable[[float], None] | None = None) -> tuple[_T, int]:
    """``call()``, asked again after each wait in ``waits`` (default ``BUSY_WAITS``)
    while it raises a 429. Returns (result, retries used); the last 429 is raised.
    Only for calls off the event loop that nobody waits on interactively (it can
    sleep ~50 s)."""
    sleep = sleep or time.sleep
    for i, wait in enumerate((*(BUSY_WAITS if waits is None else waits), None)):
        try:
            return call(), i
        except Exception as exc:
            if wait is None or not is_busy(exc):
                raise
            sleep(wait)
    raise AssertionError("unreachable")
