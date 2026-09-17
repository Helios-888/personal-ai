"""tokens: llama-swap の tokenize でトークン数を数え、届かないときは概算に切り替える。"""
from scripts.lib.tokens import count_tokens


class FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def test_count_tokens_asks_llama_swap_tokenize_endpoint_for_the_model():
    calls = []

    def fake_post(url, json, timeout):
        calls.append((url, json))
        return FakeResponse(200, {"tokens": [11, 22, 33]})

    result = count_tokens("六大", base_url="http://llm:8080", model="kukai", http_post=fake_post)

    assert result.count == 3
    assert result.exact is True
    assert calls == [("http://llm:8080/upstream/kukai/tokenize", {"content": "六大"})]


def test_count_tokens_falls_back_to_estimate_when_server_is_unreachable():
    def fake_post(url, json, timeout):
        raise ConnectionError("down")

    result = count_tokens("a" * 150, base_url="http://llm:8080", model="kukai", http_post=fake_post)

    assert result.exact is False
    assert result.count == 100  # 150 文字 ÷ 1.5


def test_count_tokens_falls_back_to_estimate_on_http_error():
    def fake_post(url, json, timeout):
        return FakeResponse(503, {"error": "loading"})

    result = count_tokens("abc", base_url="http://llm:8080", model="kukai", http_post=fake_post)

    assert result.exact is False
    assert result.count == 2  # ceil(3 / 1.5)
