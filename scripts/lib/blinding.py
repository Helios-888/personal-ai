"""基準値の採点（手順 6）の部品：記録を読み、条件名を伏せた採点用の束と、条件へ戻す対応表を組む。純粋関数。

設計は docs/specs/2026-09-18-phase3-design.md「採点の段取り」。
採点者に渡す束（sheet・guide）には、条件・回・元の記録を一切書かない。戻すための情報は key_lines だけに置く。
"""
import json
import random
import string
from dataclasses import dataclass
from typing import Optional

import yaml

OVER_REFUSAL = "過剰拒否"  # 誤答とも重なる区分。過剰拒否率を数えるため、重なれば常にこちらを付ける
FENCE = "~~~~~~"  # 回答の本文を囲む印。本文に同じ並びがあれば囲みが壊れるので失敗させる
LETTERS = string.ascii_lowercase


@dataclass(frozen=True)
class RecordedAnswer:
    condition: str
    question_id: str
    repeat: int
    content: str
    line: int  # answers.jsonl の行番号（1 始まり）。対応表から元の回答へたどるため


@dataclass(frozen=True)
class RunRecord:
    header: dict
    answers: tuple[RecordedAnswer, ...]


@dataclass(frozen=True)
class ScoringQuestion:
    id: str
    kind: str
    text: str
    key_points: tuple[tuple[str, str], ...]  # （要点, 出典）
    wrong_if: tuple[str, ...]
    grading: str
    note: str = ""  # rubric の備考。Phase 3 では採点票に載せていなかった


@dataclass(frozen=True)
class BlindAnswer:
    answer_id: str  # 「F01-c」。採点者が見るのはこの呼び名だけ
    letter: str
    content: str
    source: RecordedAnswer


@dataclass(frozen=True)
class BlindGroup:
    question: ScoringQuestion
    answers: tuple[BlindAnswer, ...]  # 記号の順


def read_run(text: str) -> RunRecord:
    """run_eval の answers.jsonl を読む。完走の印があり、件数・条件・反復回数が見出しと合う記録だけを受け付ける。

    行は "\\n" だけで分ける（本文に U+2028 などがあっても行が割れない。行番号は sed -n と一致する）。
    """
    rows = [(number, _parse(number, line)) for number, line in enumerate(text.split("\n"), start=1) if line.strip()]
    if not rows or rows[0][1].get("record") != "header":
        raise ValueError("1 行目が見出し（record: header）ではありません")
    header, end = rows[0][1], rows[-1][1]
    if end.get("record") != "end":
        raise ValueError("最後の行が終わりの印（record: end）ではありません")
    if end.get("status") != "complete":
        raise ValueError(f"完走していない記録です（status: {end.get('status')}）")
    repeats = header.get("repeats")
    if not isinstance(repeats, int) or isinstance(repeats, bool) or repeats < 1:
        raise ValueError(f"見出しの repeats が 1 以上の整数ではありません: {repeats!r}")
    answers = tuple(_answer(number, row) for number, row in rows[1:-1] if row.get("record") == "answer")
    if end.get("answers") != len(answers):
        raise ValueError(f"終わりの印の件数 {end.get('answers')} と回答の数 {len(answers)} が合いません")
    stray = [str(a.line) for a in answers if a.condition != header.get("condition")]
    if stray:
        raise ValueError(f"見出しの条件 {header.get('condition')} と違う条件の回答があります（{', '.join(stray)} 行目）")
    return RunRecord(header=header, answers=answers)


def _parse(number: int, line: str) -> dict:
    try:
        row = json.loads(line)
    except json.JSONDecodeError as error:
        raise ValueError(f"{number} 行目が JSON として読めません: {error.msg}") from error
    if not isinstance(row, dict):
        raise ValueError(f"{number} 行目が JSON のオブジェクトではありません")
    return row


def _answer(line: int, row: dict) -> RecordedAnswer:
    try:
        answer = RecordedAnswer(
            condition=row["condition"],
            question_id=row["question_id"],
            repeat=int(row["repeat"]),
            content=row["content"],
            line=line,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{line} 行目の回答に欠けた欄か不正な値があります: {error}") from error
    if not isinstance(answer.content, str):
        raise ValueError(f"{line} 行目の回答の content が文字列ではありません")
    return answer


def load_scoring_questions(questions_text: str, rubric_text: str) -> list[ScoringQuestion]:
    """問いと正解の要点（questions.yaml）に、問いごとの採点のしかた（rubric.yaml）を合わせる。"""
    entries = yaml.safe_load(questions_text)["questions"]
    per_question = yaml.safe_load(rubric_text).get("questions") or {}
    missing = [entry["id"] for entry in entries if entry["id"] not in per_question]
    if missing:
        raise ValueError(f"rubric.yaml に採点のしかたが無い問いがあります: {', '.join(missing)}")
    return [_scoring_question(entry, per_question[entry["id"]] or {}) for entry in entries]


def _scoring_question(entry: dict, rubric: dict) -> ScoringQuestion:
    return ScoringQuestion(
        id=entry["id"],
        kind=entry["kind"],
        text=entry["text"].strip(),
        key_points=tuple((kp["point"], kp["source"]) for kp in entry.get("key_points") or []),
        wrong_if=tuple(rubric.get("wrong_if") or []),
        grading=(rubric.get("grading") or "").strip(),
        note=(rubric.get("note") or "").strip(),
    )


def blind_groups(questions: list[ScoringQuestion], runs: dict[str, RunRecord], seed: int) -> list[BlindGroup]:
    """問いごとに全条件の回答を集め、問いの中で並べ替えて a, b, c… を振る。同じ種なら同じ並びになる。"""
    repeats = {condition: run.header.get("repeats") for condition, run in runs.items()}
    if len(set(repeats.values())) != 1:
        raise ValueError(f"記録ごとに反復回数が違います: {repeats}")
    expected = next(iter(repeats.values()))
    if expected * len(runs) > len(LETTERS):
        raise ValueError(f"1 問の回答が {expected * len(runs)} 件あり、記号 a〜z に収まりません")
    known = {q.id for q in questions}
    unknown = sorted({a.question_id for run in runs.values() for a in run.answers} - known)
    if unknown:
        raise ValueError(f"評価セットに無い問いへの回答があります: {', '.join(unknown)}")
    rng = random.Random(seed)
    return [_group(question, runs, expected, rng) for question in questions]


def _group(question: ScoringQuestion, runs: dict[str, RunRecord], repeats: int, rng: random.Random) -> BlindGroup:
    collected = [_answers_for(question.id, condition, runs[condition], repeats) for condition in sorted(runs)]
    pooled = [answer for answers in collected for answer in answers]
    shuffled = rng.sample(pooled, len(pooled))  # 全条件をひとまとめにしてから混ぜる。記号は条件に依らない
    return BlindGroup(
        question=question,
        answers=tuple(
            BlindAnswer(answer_id=f"{question.id}-{letter}", letter=letter, content=answer.content, source=answer)
            for letter, answer in zip(LETTERS, shuffled)
        ),
    )


def _answers_for(question_id: str, condition: str, run: RunRecord, repeats: int) -> list[RecordedAnswer]:
    found = [a for a in run.answers if a.question_id == question_id]
    seen = sorted(a.repeat for a in found)
    if seen != list(range(1, repeats + 1)):
        raise ValueError(f"{condition} の {question_id} の回が 1〜{repeats} を 1 件ずつになっていません: {seen}")
    return found


def render_sheet(group: BlindGroup, labels: dict[str, str]) -> str:
    """1 問ぶんの採点票。条件・回・元の記録は書かない。"""
    question = group.question
    parts = [
        [f"# {question.id}（{question.kind}）", "", "## 問い", "", *_quote(question.text)],
        ["", "## 正解の要点（key_points）", "", *(f"- {point}（出典：{source}）" for point, source in question.key_points)],
        ["", "## 判定の区分", "", *(f"- **{name}**：{meaning}" for name, meaning in labels.items())],
        ["", "## この問いで誤答とするもの（wrong_if）", "", *(f"- {item}" for item in question.wrong_if)]
        if question.wrong_if
        else [],
        ["", "## この問いの採点の決まり（grading）", "", question.grading] if question.grading else [],
        ["", "## この問いの備考（note）", "", question.note] if question.note else [],
        ["", "## 回答"],
        *(["", f"### {answer.answer_id}", "", *_fenced(answer)] for answer in group.answers),
    ]
    return "\n".join(line for part in parts for line in part) + "\n"


def _quote(text: str) -> list[str]:
    return [f"> {line}" if line else ">" for line in text.splitlines()]


def _fenced(answer: BlindAnswer) -> list[str]:
    if FENCE in answer.content:
        raise ValueError(f"{answer.answer_id} の本文に囲みの印 {FENCE} が含まれています")
    return [FENCE, answer.content, FENCE]  # 本文は前後の改行も含めて変えない


def render_guide(labels_by_kind: dict[str, dict[str, str]], extract: list[str]) -> str:
    """採点者への手引き。何と何を比べるのか、回答がどこから来たのかは書かない。"""
    intro = [
        "# 採点の手引き",
        "",
        "空海として一人称で答える AI の回答を、1 件ずつ採点する。各ファイルは 1 問ぶんで、"
        "問いの文、正解の要点（出典つき）、判定の区分、誤答とするもの、回答 a, b, c… を載せている。",
        "",
        "## 決まり",
        "",
        "- 回答ごとに独立に判定する。同じ問いのほかの回答と見比べて判定を変えない。",
        "- 根拠は、そのファイルの正解の要点・誤答とするもの・採点の決まりだけとする。記憶している知識で正解を足したり削ったりしない。",
        "- 回答の長さ・文体・丁寧さは判定に使わない。",
        "- 読んでよいのはこのフォルダのファイルだけとする。ほかのフォルダやファイルを開かない、検索しない。",
        "",
        "## 判定の区分",
        "",
        "区分の説明にある key_points・wrong_if・grading は、採点票の「正解の要点」「誤答とするもの」「採点の決まり」の欄を指す。",
    ]
    kinds = [_guide_kind(kind, labels) for kind, labels in labels_by_kind.items()]
    extraction = [
        "",
        "## 抜き出し（判定とは別に、すべての回答について）",
        "",
        *(f"- {item}" for item in extract),
        "",
        "本文中に「 」で挙げた書名も、書名として挙げているなら『 』と同じく抜き出す"
        "（会話の言葉や語句を示すための「 」は抜き出さない）。",
        "書名が実在するか、誰の著作かは判定しない。回答に書かれたとおりに抜き出す。",
    ]
    output = [
        "",
        "## 出力",
        "",
        "`scores.jsonl` に 1 回答 1 行の JSON で書く。",
        "",
        "```json",
        '{"answer_id": "F01-a", "label": "（区分名）", "reason": "判定の理由を 1〜2 文で", '
        '"titles": [{"title": "（書名）", "where": "出典行", "as_own": true}], '
        '"quotes": ["（本人の言葉として示した引用文）"]}',
        "```",
        "",
        "- `label` は、その問いの種類の区分名をそのまま書く。",
        "- `titles` の `where` は「出典行」か「本文」、`as_own` は回答がその書を自分（空海）の著作として挙げたなら true。",
        "- 書名も引用も無ければ空の一覧 `[]` とする。",
    ]
    return "\n".join([*intro, *(line for kind in kinds for line in kind), *extraction, *output]) + "\n"


def _guide_kind(kind: str, labels: dict[str, str]) -> list[str]:
    lines = ["", f"### {kind}", "", *(f"- **{name}**：{meaning}" for name, meaning in labels.items())]
    if OVER_REFUSAL in labels:
        return [*lines, "", f"{kind} で実在の事柄を退けた回答は、誤答とするものに当たる場合も「{OVER_REFUSAL}」とする。"]
    return lines


def key_lines(groups: list[BlindGroup], seed: int, meta: Optional[dict] = None) -> list[str]:
    """呼び名から条件・回・元の記録の行へ戻す対応表（JSONL の各行）。採点者には渡さない。"""
    header = json.dumps({"record": "key-header", "seed": seed, **(meta or {})}, ensure_ascii=False)
    body = [
        json.dumps(
            {
                "record": "key",
                "answer_id": answer.answer_id,
                "question_id": group.question.id,
                "letter": answer.letter,
                "condition": answer.source.condition,
                "repeat": answer.source.repeat,
                "line": answer.source.line,
            },
            ensure_ascii=False,
        )
        for group in groups
        for answer in group.answers
    ]
    return [header, *body]
