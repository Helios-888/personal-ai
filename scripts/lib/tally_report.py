"""基準値の集計結果（tally.md）を組む。純粋関数。

集計値だけで終わらせず、数えた問い・回答の呼び名と、元の記録へのたどり方を必ず載せる（設計書の成功基準 5）。
正式な値は 1 人目（一次採点）の集計で、2 人目は結論が変わらないかの確かめに使う（一致 9 割以上で一次採点を正式とする決め）。
"""
from scripts.lib.tally import (
    CEILING,
    CONDITIONS,
    QUESTION_MARGIN,
    QUESTION_METRICS,
    RATE_CONCLUSIONS,
    GraderTally,
    ceiling_reached,
    label_agreement,
    margin_for,
    metric_conclusion,
)

METRIC_NAMES = {
    "factual_correct": "factual 正答（多数決）",
    "factual_correct_answers": "factual 正答（回答単位）",
    "trap_refused": "trap 正しく退けた（多数決）",
    "trap_refused_answers": "trap 正しく退けた（回答単位）",
    "attribution_correct": "attribution 正（多数決）",
    "holdout_agree": "ホールドアウト 一致（回答単位）",
    "over_refusal": "過剰拒否（factual の回答単位）",
    "fictitious_strict": "架空引用（厳しい数え方）",
    "fictitious_lenient": "架空引用（緩い数え方、参考）",
}
CATEGORY_NAMES = {
    "misnamed": "書名の取り違え",
    "other_author": "他者の著作",
    "later": "後世のもの",
    "not_found": "見当たらない",
}


def _row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _table(head: list[str], rows: list[list[str]]) -> list[str]:
    return [_row(head), _row(["---"] * len(head)), *(_row(r) for r in rows)]


def render_report(header: dict, tallies: list[GraderTally], notes: list[str],
                  inputs: list[tuple[str, str]], commit: str) -> str:
    parts = [
        _intro(header, tallies, notes, inputs, commit),
        _metrics(tallies),
        _conclusions(tallies),
        _ceiling(tallies),
        _agreement(tallies),
        _per_question(tallies[0]),
        *(_majority_differences(tallies[0], other) for other in tallies[1:]),
        *(_fictitious(t) for t in tallies),
        _appendix(tallies),
        _how_to_trace(),
    ]
    return "\n".join(line for part in parts for line in part) + "\n"


def _intro(header: dict, tallies: list[GraderTally], notes: list[str], inputs: list[tuple[str, str]], commit: str) -> list[str]:
    sources = header.get("sources") or {}
    return [
        "# 基準値の集計（Phase 3）",
        "",
        "`scripts/tally.py` が作成した。条件は `key.jsonl` で戻した。定義は `docs/specs/2026-09-18-phase3-design.md` の"
        "「採点の段取り」5 と「指標」、架空引用の数え方は `titles.yaml` の冒頭に従う。",
        "",
        f"- **正式な値は {tallies[0].name} の集計**（2 人の区分の一致が 9 割以上なら一次採点を正式とする決め）。"
        + (f"{', '.join(t.name for t in tallies[1:])} は結論が変わらないかの確かめに使う。" if len(tallies) > 1 else ""),
        *(f"- 記録 {c}：`{(sources.get(c) or {}).get('path', '（見出しに無し）')}`" for c in CONDITIONS),
        *(f"- {note}" for note in notes),
        f"- 集計したときの Git コミット：`{commit}`（入力と集計のコードに未コミットの変更が無いことを確かめた）",
        "",
        "入力の指紋（SHA-256）：",
        "",
        *_table(["ファイル", "SHA-256"], [[f"`{path}`", f"`{digest}`"] for path, digest in inputs]),
    ]


def _metrics(tallies: list[GraderTally]) -> list[str]:
    head = ["指標", *(f"{c}（{t.name}）" for t in tallies for c in CONDITIONS)]
    rows = [[label, *(t.metrics[c][name].text() for t in tallies for c in CONDITIONS)] for name, label in METRIC_NAMES.items()]
    return ["", "## 指標", "", *_table(head, rows), "",
            "架空引用の行は、採点者の区分ではなく、書名の組（一次採点の抜き出し＋その列の採点者の組の補助抜き出し）で決まる。"]


def _conclusions(tallies: list[GraderTally]) -> list[str]:
    rows = []
    for name in (*QUESTION_METRICS, *RATE_CONCLUSIONS):
        total = tallies[0].metrics["B0"][name].total
        margin = f"{margin_for(name, total)} {'問' if name in QUESTION_METRICS else '回答'}"
        verdicts = [metric_conclusion(t, name) for t in tallies]
        rows.append([METRIC_NAMES[name], margin, *verdicts, "同じ" if len(set(verdicts)) == 1 else "**食い違い**"])
    return ["", "## 結論の照合", "",
            f"多数決の問い数は {QUESTION_MARGIN} 問以上の差（設計書「明確に上回る」）、回答単位の率は分母の 1 割以上の回答数の差を"
            "「明確」とする（率の規則は 2026-09-19、条件を戻す前に決めた）。ホールドアウトは参考値なので結論に使わない。",
            "", *_table(["指標", "明確とする差", *(t.name for t in tallies), "採点者間"], rows)]


def _ceiling(tallies: list[GraderTally]) -> list[str]:
    lines = [f"- {t.name}：B1 の trap 正しく退けた {t.metrics['B1']['trap_refused'].text()} → "
             f"{'天井効果に当たる' if ceiling_reached(t) else '当たらない'}" for t in tallies]
    return ["", f"## 天井効果（設計書「撤退条件」：B1 の trap 正しい拒否が {CEILING}/10 以上）", "",
            f"判定は正式な値（{tallies[0].name}）で行う。", "", *lines]


def _agreement(tallies: list[GraderTally]) -> list[str]:
    if len(tallies) < 2:
        return []
    first, second = ({j.entry.answer_id: j for j in t.judged} for t in tallies[:2])
    same, total, differ = label_agreement({a: j.label for a, j in first.items()}, {a: j.label for a, j in second.items()})
    rows = [[a, first[a].entry.condition, f"{first[a].entry.repeat} 回目", first[a].label, second[a].label] for a in differ]
    body = _table(["回答", "条件", "回", tallies[0].name, tallies[1].name], rows) if rows else ["（なし）"]
    return ["", "## 採点者間の一致", "", f"区分の一致：{same}/{total}（{same / total:.1%}）", "", *body]


def _cell(result) -> str:
    detail = "・".join(f"{j.entry.answer_id[-1]} {j.label}" for j in result.answers)
    return f"{result.majority}{'※' if result.tie else ''}（{detail}）"


def _order(question_id: str) -> tuple[int, str]:
    return ("FATH".find(question_id[0]), question_id)  # factual・attribution・trap・holdout の順


def _per_question(t: GraderTally) -> list[str]:
    by_key = {(r.question_id, r.condition): r for r in t.results}
    questions = sorted({(r.question_id, r.kind) for r in t.results}, key=lambda q: _order(q[0]))
    rows = [[q, kind, *(_cell(by_key[(q, c)]) for c in CONDITIONS)] for q, kind in questions]
    return ["", f"## 問いごとの判定（{t.name}）", "",
            "多数決の区分のあとに、1〜3 回目の回答の記号と区分を並べた。※は 3 回とも割れて rubric の majority_tie で決めたもの。",
            "", *_table(["問い", "種類", *CONDITIONS], rows)]


def _majority_differences(first: GraderTally, other: GraderTally) -> list[str]:
    theirs = {(r.question_id, r.condition): r for r in other.results}
    rows = [[r.question_id, r.condition, _cell(r), _cell(theirs[(r.question_id, r.condition)])]
            for r in sorted(first.results, key=lambda r: (_order(r.question_id), r.condition))
            if r.majority != theirs[(r.question_id, r.condition)].majority]
    body = _table(["問い", "条件", first.name, other.name], rows) if rows else ["（なし）"]
    return ["", f"## 多数決が {first.name} と {other.name} で食い違った問い", "", *body]


def _fictitious(t: GraderTally) -> list[str]:
    entries = {j.entry.answer_id: j.entry for j in t.judged}
    rows = [
        [a, c, f"{entries[a].repeat} 回目", str(entries[a].line),
         "、".join(f"{title}（{CATEGORY_NAMES[cat]}）" for title, cat in hits)]
        for c in CONDITIONS for a, hits in t.fictitious["strict"][c]
    ]
    body = _table(["回答", "条件", "回", "answers.jsonl の行", "自分の作として挙げた書名（区分）"], rows) if rows else ["（なし）"]
    return ["", f"## 架空引用の内訳（厳しい数え方、{t.name} の組）", "",
            "緩い数え方では、書名の取り違えだけに当たる回答を除く。", "", *body]


def _appendix(tallies: list[GraderTally]) -> list[str]:
    lines = ["", "## 付録：数えた問い・回答の一覧", ""]
    for t in tallies:
        lines += [f"### {t.name}", ""]
        for name, label in METRIC_NAMES.items():
            for c in CONDITIONS:
                count = t.metrics[c][name]
                lines.append(f"- {label}・{c}（{count.text()}）：{'、'.join(count.hits) or 'なし'}")
        lines.append("")
    return lines


def _how_to_trace() -> list[str]:
    return ["## 元の回答へのたどり方", "",
            "呼び名（例 F01-c）の条件・回は上の表と `key.jsonl` にある。本文は各記録の `transcript.md` の"
            "「## <問い id>」の下の「### <回> 回目」、または `answers.jsonl` の行（`key.jsonl` の line）で読める。"]
