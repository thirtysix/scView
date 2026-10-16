"""Per-model request settings for the DeepInfra chat calls."""

from __future__ import annotations


def chat_extra(model: str) -> dict:
    """Extra keyword arguments for ``chat.completions.create`` with ``model``.

    DeepSeek's V4 models can think before answering; scView's calls are short,
    grounded and latency-sensitive, so thinking is turned off explicitly rather
    than left to the provider's default. Other models get nothing extra.
    """
    if "deepseek" in (model or "").lower():
        return {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
    return {}
