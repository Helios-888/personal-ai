"""トークン数の計測。

本番と同じ数え方にするため llama-swap（llama-server）の tokenize を使う。
届かないときは文字数からの概算に切り替え、exact=False で返す。
"""
import math
from dataclasses import dataclass
from typing import Callable

import requests

CHARS_PER_TOKEN_ESTIMATE = 1.5
_FALLBACK_ERRORS = (OSError, ValueError, KeyError, TypeError, RuntimeError, requests.RequestException)


@dataclass(frozen=True)
class TokenCount:
    count: int
    exact: bool


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / CHARS_PER_TOKEN_ESTIMATE)


def count_tokens(
    text: str,
    *,
    base_url: str,
    model: str,
    http_post: Callable = requests.post,
    timeout: float = 30,
) -> TokenCount:
    url = f"{base_url.rstrip('/')}/upstream/{model}/tokenize"
    try:
        response = http_post(url, json={"content": text}, timeout=timeout)
        response.raise_for_status()
        return TokenCount(count=len(response.json()["tokens"]), exact=True)
    except _FALLBACK_ERRORS:
        return TokenCount(count=estimate_tokens(text), exact=False)
