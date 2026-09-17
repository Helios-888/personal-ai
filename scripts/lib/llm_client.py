"""llama-swap（OpenAI 互換 API）の chat/completions を 1 往復だけ呼ぶ薄いラッパー。

ストリーミングはしない。思考（reasoning_content）は本文と分けて返す。
"""
import time
from dataclasses import dataclass
from typing import Callable

import requests

DEFAULT_TIMEOUT_SECONDS = 600  # 思考 ON だと 1 問に数分かかることがある


@dataclass(frozen=True)
class ChatResult:
    content: str
    reasoning: str
    finish_reason: str
    completion_tokens: int
    elapsed_seconds: float


def chat_completion(
    base_url: str,
    model: str,
    *,
    system: str,
    user: str,
    temperature: float,
    max_tokens: int,
    http_post: Callable = requests.post,
    clock: Callable[[], float] = time.monotonic,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> ChatResult:
    body = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    started = clock()
    response = http_post(f"{base_url.rstrip('/')}/v1/chat/completions", json=body, timeout=timeout)
    response.raise_for_status()
    elapsed = clock() - started
    return _parse(response.json(), elapsed)


def _parse(payload, elapsed: float) -> ChatResult:
    choices = payload.get("choices") if isinstance(payload, dict) else None
    if not choices:
        raise ValueError(f"chat/completions の応答に choices がありません: {str(payload)[:200]}")
    choice = choices[0]
    message = choice.get("message") or {}
    usage = payload.get("usage") or {}
    return ChatResult(
        content=message.get("content") or "",
        reasoning=message.get("reasoning_content") or "",
        finish_reason=str(choice.get("finish_reason") or ""),
        completion_tokens=int(usage.get("completion_tokens") or 0),
        elapsed_seconds=elapsed,
    )
