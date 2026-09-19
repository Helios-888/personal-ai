"""llm_client: llama-swap（OpenAI 互換 API）の chat/completions を 1 往復だけ呼ぶ薄いラッパー。"""
import pytest
import requests

from scripts.lib.llm_client import ChatResult, chat_completion


class FakeResponse:
    def __init__(self, status_code, payload, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


def completion_payload(content, *, finish="stop", tokens=42, reasoning=None):
    message = {"role": "assistant", "content": content}
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    return {
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": {"prompt_tokens": 100, "completion_tokens": tokens},
    }


def call(fake_post, **overrides):
    kwargs = dict(
        base_url="http://llm:8080",
        model="kukai",
        system="S",
        user="U",
        temperature=0.7,
        max_tokens=800,
        http_post=fake_post,
    )
    kwargs.update(overrides)
    return chat_completion(**kwargs)


def test_chat_completion_posts_a_single_turn_to_the_openai_compatible_endpoint():
    calls = []

    def fake_post(url, json, timeout):
        calls.append((url, json))
        return FakeResponse(200, completion_payload("若者よ"))

    call(fake_post)

    url, body = calls[0]
    assert url == "http://llm:8080/v1/chat/completions"
    assert body["model"] == "kukai"
    assert body["messages"] == [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]
    assert body["temperature"] == 0.7
    assert body["max_tokens"] == 800
    assert body["stream"] is False


def test_chat_completion_returns_content_finish_reason_and_completion_tokens():
    result = call(lambda url, json, timeout: FakeResponse(200, completion_payload("若者よ", tokens=12)))

    assert result.content == "若者よ"
    assert result.finish_reason == "stop"
    assert result.completion_tokens == 12
    assert result.reasoning == ""


def test_chat_completion_keeps_reasoning_text_apart_from_the_answer():
    payload = completion_payload("本文", reasoning="We need to answer in Japanese")

    result = call(lambda url, json, timeout: FakeResponse(200, payload))

    assert result.content == "本文"
    assert result.reasoning == "We need to answer in Japanese"


def test_chat_completion_treats_missing_answer_as_empty_string_not_none():
    # 思考だけで出力枠を使い切ると content が null で返る（Phase 2 入口の実測）
    payload = completion_payload(None, finish="length", reasoning="思考のみ")

    result = call(lambda url, json, timeout: FakeResponse(200, payload))

    assert result.content == ""
    assert result.finish_reason == "length"


def test_chat_completion_measures_elapsed_seconds_with_the_injected_clock():
    ticks = iter([10.0, 16.2])

    result = call(lambda url, json, timeout: FakeResponse(200, completion_payload("x")), clock=lambda: next(ticks))

    assert result.elapsed_seconds == pytest.approx(6.2)


def test_chat_completion_raises_on_http_error_instead_of_returning_an_empty_answer():
    with pytest.raises(requests.HTTPError):
        call(lambda url, json, timeout: FakeResponse(503, {"error": "loading"}))


def test_chat_completion_names_the_missing_field_when_the_response_shape_is_unexpected():
    with pytest.raises(ValueError) as excinfo:
        call(lambda url, json, timeout: FakeResponse(200, {"object": "chat.completion"}))

    assert "choices" in str(excinfo.value)


def test_chat_completion_uses_a_long_timeout_by_default_because_answers_take_tens_of_seconds():
    seen = {}

    def fake_post(url, json, timeout):
        seen["timeout"] = timeout
        return FakeResponse(200, completion_payload("x"))

    call(fake_post)

    assert seen["timeout"] >= 300


# --- レビュー指摘への回帰テスト（2026-09-18） ---


def test_chat_completion_puts_status_and_server_body_in_the_http_error_message():
    with pytest.raises(requests.HTTPError) as excinfo:
        call(lambda url, json, timeout: FakeResponse(400, {"error": "x"}, text='{"error":"model not found"}'))

    assert "400" in str(excinfo.value)
    assert "model not found" in str(excinfo.value)


def test_chat_completion_joins_text_parts_when_the_server_returns_content_as_a_list():
    payload = completion_payload([{"type": "text", "text": "若者"}, {"type": "text", "text": "よ"}])

    result = call(lambda url, json, timeout: FakeResponse(200, payload))

    assert result.content == "若者よ"


# --- Open WebUI 経路（Phase 3 の評価実行、2026-09-18） ---


def test_chat_completion_sends_only_the_user_turn_when_system_is_none():
    # Open WebUI のカスタムモデルは登録済みの system を自分で先頭に足す。こちらからも送ると二重になる
    bodies = []

    def fake_post(url, json, timeout):
        bodies.append(json)
        return FakeResponse(200, completion_payload("x"))

    call(fake_post, system=None)

    assert bodies[0]["messages"] == [{"role": "user", "content": "U"}]


def test_chat_completion_can_target_another_endpoint_with_headers():
    calls = []

    def fake_post(url, json, timeout, headers):
        calls.append((url, headers))
        return FakeResponse(200, completion_payload("x"))

    call(
        fake_post,
        base_url="http://owui:3000/",
        endpoint="/api/chat/completions",
        headers={"Authorization": "Bearer k"},
    )

    assert calls == [("http://owui:3000/api/chat/completions", {"Authorization": "Bearer k"})]


def test_chat_completion_reports_prompt_tokens_to_compare_what_each_route_actually_sent():
    result = call(lambda url, json, timeout: FakeResponse(200, completion_payload("x")))

    assert result.prompt_tokens == 100


def test_chat_result_prompt_tokens_defaults_to_zero_for_existing_callers():
    result = ChatResult(content="c", reasoning="", finish_reason="stop", completion_tokens=1, elapsed_seconds=0.1)

    assert result.prompt_tokens == 0


def test_chat_completion_keeps_the_passages_open_webui_retrieved():
    # 付け足し・書き換えの判定には、回答ごとに検索された箇所が要る（faithfulness.yaml）。Open WebUI は応答の sources で返す
    sources = [{"source": {"name": "primary__即身成仏義.md"}, "document": ["T2428_.77.0382c22 四種曼荼羅者"], "distances": [0.4]}]
    payload = {**completion_payload("x"), "sources": sources}

    result = call(lambda url, json, timeout: FakeResponse(200, payload))

    assert result.sources == tuple(sources)


def test_chat_result_has_no_sources_without_retrieval():
    result = call(lambda url, json, timeout: FakeResponse(200, completion_payload("x")))

    assert result.sources == ()
