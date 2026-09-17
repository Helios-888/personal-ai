"""llama-swap（OpenAI 互換 API）の chat/completions を 1 往復だけ呼ぶ薄いラッパー。

ストリーミングはしない。思考（reasoning_content）は本文と分けて返す。
"""
import time
from dataclasses import dataclass
from typing import Callable

import requests

DEFAULT_TIMEOUT_SECONDS = 600  # 思考 ON だと 1 問に数分かかることがある
_BODY_PREVIEW_CHARS = 200


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
    if response.status_code >= 400:
        # 失敗理由（モデル未登録、コンテキスト超過など）はサーバの本文にしか無いので添える
        raise requests.HTTPError(
            f"chat/completions が HTTP {response.status_code} を返しました: "
            f"{str(response.text)[:_BODY_PREVIEW_CHARS]}",
            response=response,
        )
    elapsed = clock() - started
    return _parse(response.json(), elapsed)


def _parse(payload, elapsed: float) -> ChatResult:
    choices = payload.get("choices") if isinstance(payload, dict) else None
    if not choices:
        raise ValueError(f"chat/completions の応答に choices がありません: {str(payload)[:_BODY_PREVIEW_CHARS]}")
    choice = choices[0]
    message = choice.get("message") or {}
    usage = payload.get("usage") or {}
    return ChatResult(
        content=_as_text(message.get("content")),
        reasoning=_as_text(message.get("reasoning_content")),
        finish_reason=str(choice.get("finish_reason") or ""),
        completion_tokens=int(usage.get("completion_tokens") or 0),
        elapsed_seconds=elapsed,
    )


def _as_text(raw) -> str:
    """content は文字列が普通だが、OpenAI 互換のパーツ配列（[{type, text}]）で返る実装もある。必ず str にする。"""
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        return "".join(str(part.get("text", "")) for part in raw if isinstance(part, dict))
    return str(raw)
