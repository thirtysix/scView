"""Per-model request settings for the DeepInfra chat calls.

scView's calls are short, grounded and latency-sensitive, so thinking is turned off
wherever the model allows it. Many models a user may set as ``RAG_CHAT_MODEL`` or
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

Unknown models get nothing extra, as before. Every setting goes in ``extra_body``, so
the request is the same with any openai SDK version.
"""

from __future__ import annotations

import copy

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
