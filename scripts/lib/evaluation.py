"""評価の実行（scripts/run_eval.py）の部品：凍結した問いの読み込み、問う順番、記録（JSONL・Markdown）の組み立て。純粋関数。

設計は docs/specs/2026-09-18-phase3-design.md「評価実行スクリプト」。
"""
import json
import re
from dataclasses import asdict, dataclass
from typing import Optional

import yaml

from scripts.lib.llm_client import ChatResult
from scripts.lib.sat import LINE_ID

_REQUIRED_FIELDS = ("id", "kind", "text")
END_STATUSES = ("complete", "interrupted")


@dataclass(frozen=True)
class EvalQuestion:
    """モデルに渡してよい欄だけを持つ。正解の要点（key_points）はここに入れない。"""

    id: str
    kind: str
    text: str


@dataclass(frozen=True)
class RunHeader:
    """記録の見出し。比較の土俵（何を・どの経路で・どの定義で問うたか）の証拠として残す。"""

    condition: str
    route: str
    url: str
    model: str
    date: str
    questions_sha256: str
    system_prompt_sha256: str  # モデルが受け取るシステムプロンプトの指紋
    system_prompt_sent: bool  # False なら要求に含めず、Open WebUI が登録済みのものを足した
    system_tokens: int
    tokens_exact: bool
    temperature: float
    max_tokens: int
    repeats: int
    question_count: int
    agent: str
    question_ids: tuple[str, ...]
    git_commit: str
    git_dirty: tuple[str, ...]  # 定義とコードの未コミットの変更。空でなければ指紋は履歴のどの版とも一致しない
    note: str = ""  # 途中で中断したときなどの注記


@dataclass(frozen=True)
class AnswerRecord:
    repeat: int
    question: EvalQuestion
    result: ChatResult


def load_eval_questions(text: str) -> list[EvalQuestion]:
    """凍結した questions.yaml の本文から id・kind・text だけを読む。text は前後の空白だけ除き、段落の改行は残す。"""
    data = yaml.safe_load(text)
    entries = data.get("questions") if isinstance(data, dict) else None
    if not isinstance(entries, list) or not entries:
        raise ValueError("questions は 1 件以上の一覧である必要があります")
    questions = [_question(index, entry) for index, entry in enumerate(entries, start=1)]
    ids = [q.id for q in questions]
    duplicated = sorted({i for i in ids if ids.count(i) > 1})
    if duplicated:
        raise ValueError(f"問いの id が重複しています: {', '.join(duplicated)}")
    return questions


def _question(index: int, entry) -> EvalQuestion:
    values = {key: entry.get(key) if isinstance(entry, dict) else None for key in _REQUIRED_FIELDS}
    missing = [key for key, value in values.items() if not isinstance(value, str) or not value.strip()]
    if missing:
        raise ValueError(f"{index} 番目の問いに {', '.join(missing)}（文字列）がありません")
    return EvalQuestion(id=values["id"].strip(), kind=values["kind"].strip(), text=values["text"].strip())


def select_questions(questions: list[EvalQuestion], only: Optional[list[str]]) -> list[EvalQuestion]:
    """only が空なら全問。指定があれば、その id の問いだけをファイルの順で返す。"""
    if not only:
        return list(questions)
    known = {q.id for q in questions}
    unknown = [i for i in only if i not in known]
    if unknown:
        raise ValueError(f"評価セットに無い問いの id です: {', '.join(unknown)}")
    return [q for q in questions if q.id in set(only)]


def plan_calls(questions: list[EvalQuestion], repeats: int) -> list[tuple[int, EvalQuestion]]:
    """1 巡目に全問を問い、2 巡目、3 巡目と進む。途中で止まっても各問いの 1 回目が先に揃う。"""
    return [(repeat, question) for repeat in range(1, repeats + 1) for question in questions]


def header_json(header: RunHeader) -> str:
    return json.dumps({"record": "header", **asdict(header)}, ensure_ascii=False)


def answer_json(condition: str, record: AnswerRecord) -> str:
    """回答は本文を加工せずに残す（採点は本文そのものに対して行う）。"""
    result = record.result
    return json.dumps(
        {
            "record": "answer",
            "condition": condition,
            "question_id": record.question.id,
            "kind": record.question.kind,
            "repeat": record.repeat,
            "content": result.content,
            "reasoning": result.reasoning,
            "finish_reason": result.finish_reason,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "elapsed_seconds": result.elapsed_seconds,
            **({"sources": list(result.sources)} if result.sources else {}),  # 検索された箇所（K1）
        },
        ensure_ascii=False,
    )


def end_json(status: str, answers: int, inconsistent: list[str]) -> str:
    """記録の終わりの印。完走したか中断したかを、機械が読む側でも区別できるようにする。"""
    if status not in END_STATUSES:
        raise ValueError(f"status は {', '.join(END_STATUSES)} のいずれかです: {status}")
    return json.dumps(
        {"record": "end", "status": status, "answers": answers, "inconsistent_prompt_tokens": inconsistent},
        ensure_ascii=False,
    )


def count_truncated(records: list[AnswerRecord]) -> int:
    return sum(1 for record in records if record.result.finish_reason == "length")


def inconsistent_prompt_tokens(records: list[AnswerRecord]) -> list[str]:
    """同じ問いなのに入力トークン数が回ごとに違う問いの id。違えば途中でシステムプロンプトか経路が変わった。

    usage を返さない応答（0）は比べない。
    """
    seen: dict[str, set[int]] = {}
    for record in records:
        if record.result.prompt_tokens:
            seen = {**seen, record.question.id: seen.get(record.question.id, set()) | {record.result.prompt_tokens}}
    return [question_id for question_id, sizes in seen.items() if len(sizes) > 1]


def render_eval_transcript(header: RunHeader, records: list[AnswerRecord]) -> str:
    """人が読む記録。見出しに比較の土俵を書き、問いごとに各回の回答をまとめて並べる。"""
    lines = _header_lines(header, records)
    for question in _in_order_of_appearance(records):
        lines.extend(["", *_question_section(question, [r for r in records if r.question == question])])
    return "\n".join(lines) + "\n"


def _header_lines(header: RunHeader, records: list[AnswerRecord]) -> list[str]:
    estimate = "" if header.tokens_exact else "（概算）"
    sent = (
        "要求の system として送った。"
        if header.system_prompt_sent
        else "要求には含めない（Open WebUI が登録済みのものを先頭に足す）。"
    )
    lines = [
        f"# 評価の記録 {header.condition}（{header.date}）",
        "",
        f"- 条件：{header.condition}",
        f"- 経路：{header.route}（{header.url}）、モデル {header.model}",
        f"- システムプロンプト：{header.system_tokens} トークン{estimate}、SHA-256 {header.system_prompt_sha256}。{sent}",
        f"- questions.yaml の SHA-256：{header.questions_sha256}",
        f"- temperature {header.temperature}、max_tokens {header.max_tokens}、"
        f"{header.question_count} 問 × {header.repeats} 回",
        f"- 定義：{header.agent}",
        f"- Git：{header.git_commit}{_dirty_text(header.git_dirty)}",
        f"- 回答 {len(records)} 件、出力枠切れ（finish=length）{count_truncated(records)}/{len(records)}",
    ]
    if header.note:
        lines.extend(["", header.note])
    return lines


def _dirty_text(dirty: tuple[str, ...]) -> str:
    if not dirty:
        return "（定義とコードに未コミットの変更なし）"
    return "。未コミットの変更：" + "、".join(line.strip() for line in dirty)


def _in_order_of_appearance(records: list[AnswerRecord]) -> list[EvalQuestion]:
    return list(dict.fromkeys(record.question for record in records))


def _question_section(question: EvalQuestion, records: list[AnswerRecord]) -> list[str]:
    quoted = [f"> {line}".rstrip() for line in question.text.splitlines()]
    lines = [f"## {question.id} [{question.kind}]", "", *quoted]
    for record in records:
        lines.extend(["", *_answer_section(record)])
    return lines


def _answer_section(record: AnswerRecord) -> list[str]:
    result = record.result
    body = result.content.strip()
    stats = [
        f"{result.elapsed_seconds:.1f} 秒",
        f"{len(body)} 字",
        f"入力 {result.prompt_tokens} トークン",
        f"出力 {result.completion_tokens} トークン",
    ]
    if result.reasoning:
        stats.append(f"思考 {len(result.reasoning)} 字")
    stats.append(f"finish={result.finish_reason}")
    lines = [f"### {record.repeat} 回目（" + "、".join(stats) + "）", "", body or "（本文なし）"]
    if result.sources:
        lines += ["", "検索された箇所：" + "、".join(_source_ranges(result.sources))]
    return lines


def _source_ranges(sources) -> list[str]:
    """検索された箇所を「資料名（最初の行 ID–最後の行 ID）」で並べる。本文は answers.jsonl に残る。"""
    ranges = []
    for source in sources:
        name = (source.get("source") or {}).get("name") or "（名前なし）"
        for document in source.get("document") or []:
            ids = re.findall(LINE_ID, document)
            ranges.append(f"{name}（{ids[0]}–{ids[-1]}）" if ids else name)
    return ranges
