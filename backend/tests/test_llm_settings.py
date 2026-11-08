"""Per-model request settings for the DeepInfra chat calls (core/llm.py: probed 2026-11-07)."""

import pytest

from scview.core.llm import chat_extra, reply_room

OFF = {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
LOW = {"extra_body": {"reasoning_effort": "low"}}


def test_deepseek_thinking_is_turned_off():
    for m in ("deepseek-ai/DeepSeek-V4-Flash-0731", "deepseek-ai/DeepSeek-V4.1-Flash"):
        assert chat_extra(m) == OFF


@pytest.mark.parametrize("model", [
    "deepseek-ai/DeepSeek-V4-Pro", "deepseek-ai/DeepSeek-V4.1-Flash-1017", "zai-org/GLM-5.2",
    "moonshotai/Kimi-K2.6", "Qwen/Qwen3.5-397B-A17B", "Qwen/Qwen3.6-35B-A3B", "Qwen/Qwen3.8-Flash",
    "Qwen/Qwen3.8-Max", "MiniMaxAI/MiniMax-M3", "XiaomiMiMo/MiMo-V2.6-Pro",
    "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B", "ByteDance/Seed-2.0-pro", "ByteDance/Seed-2.0-mini",
    "tencent/Hy4-preview",
])
def test_models_that_think_by_default_get_enable_thinking_false(model):
    assert chat_extra(model) == OFF
    assert reply_room(model) == 0


@pytest.mark.parametrize("model", [
    "Qwen/Qwen3-235B-A22B-Instruct-2507",  # instruct only: never send enable_thinking
    "deepseek-ai/DeepSeek-V3.2", "google/gemma-4-31B-it", "anthropic/claude-sonnet-5-5",
    "google/gemini-3.5-flash", "meta-llama/Meta-Llama-3.1-8B-Instruct", "", None,
])
def test_models_without_thinking_and_unknown_models_get_nothing(model):
    assert chat_extra(model) == {}
    assert reply_room(model) == 0


@pytest.mark.parametrize("model", ["zai-org/GLM-5.3", "zai-org/GLM-5.3-Flash", "openai/gpt-oss-120b",
                                   "openai/gpt-oss-20b"])
def test_glm53_and_gpt_oss_get_low_effort_and_reply_room(model):
    # GLM-5.3(-Flash): enable_thinking false leaks reasoning into the content or spends
    # the whole budget reasoning; gpt-oss cannot stop reasoning at all.
    assert chat_extra(model) == LOW
    assert reply_room(model) >= 1000


def test_match_is_case_insensitive_and_ignores_the_provider():
    assert chat_extra("ZAI-ORG/glm-5.3-flash") == LOW
    assert chat_extra("GLM-5.2") == chat_extra("zai-org/GLM-5.2") == OFF
    assert reply_room("OPENAI/GPT-OSS-120B") == reply_room("openai/gpt-oss-120b") > 0


def test_settings_are_copies():
    a = chat_extra("openai/gpt-oss-120b")
    a["extra_body"]["reasoning_effort"] = "high"
    assert chat_extra("openai/gpt-oss-120b") == LOW


class _Busy(Exception):
    status_code = 429


def test_busy_retry_waits_and_asks_again_on_429_only():
    from scview.core.llm import busy_retry

    slept: list[float] = []
    answers = iter([_Busy(), _Busy(), "ok"])

    def call():
        a = next(answers)
        if isinstance(a, Exception):
            raise a
        return a

    assert busy_retry(call, waits=(5, 15, 30), sleep=slept.append) == ("ok", 2)
    assert slept == [5, 15]

    def other():
        raise ValueError("bad request")

    with pytest.raises(ValueError):
        busy_retry(other, waits=(5,), sleep=slept.append)
    assert slept == [5, 15]  # no wait for an error that is not a 429


def test_busy_retry_raises_the_last_429_once_the_waits_are_used():
    from scview.core.llm import busy_retry

    slept: list[float] = []

    def call():
        raise _Busy()

    with pytest.raises(_Busy):
        busy_retry(call, waits=(1, 2), sleep=slept.append)
    assert slept == [1, 2]
