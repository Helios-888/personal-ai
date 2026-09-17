"""プローブ（語調・判断の確認）の質問読み込みと、記録（Markdown）の組み立て。純粋関数。"""
from dataclasses import dataclass
from pathlib import Path

import yaml

from scripts.lib.llm_client import ChatResult


@dataclass(frozen=True)
class Question:
    kind: str
    text: str


@dataclass(frozen=True)
class TranscriptHeader:
    title: str
    date: str
    conditions: str
    temperature: float
    max_tokens: int
    system_tokens: int
    tokens_exact: bool


@dataclass(frozen=True)
class ProbeRecord:
    question: Question
    result: ChatResult


def load_questions(path: Path) -> list[Question]:
    """`questions:` の一覧を順に読む。text の無い項目は番号を添えて失敗する。"""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    entries = data.get("questions") if isinstance(data, dict) else None
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{path}: questions は 1 件以上の一覧である必要があります")
    questions: list[Question] = []
    for index, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict) or not str(entry.get("text") or "").strip():
            raise ValueError(f"{path}: {index} 番目の質問に text がありません")
        questions.append(Question(kind=str(entry.get("kind") or ""), text=str(entry["text"]).strip()))
    return questions


def render_transcript(header: TranscriptHeader, records: list[ProbeRecord]) -> str:
    """既存の記録（evaluations/kukai/phase2-voice-check/）と同じ体裁の Markdown を返す。"""
    estimate = "" if header.tokens_exact else "（概算）"
    lines = [
        f"# {header.title}（{header.date}、{header.conditions}、"
        f"temperature {header.temperature}、max_tokens {header.max_tokens}）",
        "",
        f"システムプロンプト {header.system_tokens} トークン{estimate}。",
    ]
    for record in records:
        lines.extend(["", *_section(record)])
    return "\n".join(lines) + "\n"


def _section(record: ProbeRecord) -> list[str]:
    result = record.result
    body = result.content.strip()
    stats = [f"{result.elapsed_seconds:.1f} 秒", f"{len(body)} 字", f"{result.completion_tokens} トークン"]
    if result.reasoning:
        stats.append(f"思考 {len(result.reasoning)} 字")
    stats.append(f"finish={result.finish_reason}")
    return [
        f"## [{record.question.kind}] {record.question.text}",
        "",
        "（" + "、".join(stats) + "）",
        "",
        body or "（本文なし）",
    ]
