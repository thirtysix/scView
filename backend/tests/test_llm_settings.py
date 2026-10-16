"""Per-model request settings for the DeepInfra chat calls."""

from scview.core.llm import chat_extra


def test_deepseek_thinking_is_turned_off():
    for m in ("deepseek-ai/DeepSeek-V4-Flash-0731", "deepseek-ai/DeepSeek-V4.1-Flash"):
        assert chat_extra(m) == {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}


def test_other_models_get_nothing_extra():
    assert chat_extra("meta-llama/Meta-Llama-3.1-8B-Instruct") == {}
    assert chat_extra("") == {}
