"""採点の集計結果（tally.md）を組む。純粋関数。

集計値だけで終わらせず、数えた問い・回答の呼び名と、元の記録へのたどり方を必ず載せる（Phase 3 設計書の成功基準 5）。
正式な値は 1 人目（一次採点）の集計で、2 人目は結論が変わらないかの確かめに使う（一致 9 割以上で一次採点を正式とする決め）。
条件名は各集計の conditions（基準値が先）。2 人目の採点範囲が限られていれば、範囲の外の欄は「—」とする。
"""
from scripts.lib.tally import (
    CEILING,
    CONTENT_QUESTIONS,
    QUESTION_MARGIN,
    QUESTION_METRICS,
    RATE_CONCLUSIONS,
    GraderTally,
    ceiling_reached,
    gate,
    label_agreement,
    margin_for,
    metric_conclusion,
)

METRIC_NAMES = {  # 経典の内容を問う 10 問を先に（Phase 4 設計書「採点」の集計）
    "content_correct": "内容 10 問 正答（多数決）",
    "content_correct_answers": "内容 10 問 正答（回答単位）",
    "content_over_refusal": "内容 10 問 過剰拒否（回答単位）",
    "content_unfaithful": "内容 10 問 付け足し・書き換えあり（回答単位）",
    "factual_correct": "factual 正答（多数決）",
    "factual_correct_answers": "factual 正答（回答単位）",
    "trap_refused": "trap 正しく退けた（多数決）",
    "trap_refused_answers": "trap 正しく退けた（回答単位）",
    "attribution_correct": "attribution 正（多数決）",
    "holdout_agree": "ホールドアウト 一致（回答単位）",
    "over_refusal": "過剰拒否（factual の回答単位）",
    "fictitious_strict": "架空引用（厳しい数え方）",
    "fictitious_lenient": "架空引用（緩い数え方、参考）",
    "unfaithful_answers": "付け足し・書き換えあり（回答単位、全問）",
}
CATEGORY_NAMES = {
    "misnamed": "書名の取り違え",
    "other_author": "他者の著作",
    "later": "後世のもの",
    "not_found": "見当たらない",
}
DASH = "—"


def _row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _table(head: list[str], rows: list[list[str]]) -> list[str]:
    return [_row(head), _row(["---"] * len(head)), *(_row(r) for r in rows)]


def _plain(text: str) -> str:
    """表の升に入れる文。改行と縦棒は表を壊すので置き換える。"""
    return text.replace("\n", " ").replace("|", "｜")


def _names(t: GraderTally) -> list[str]:
    """その集計にある指標（METRIC_NAMES の順）。"""
    present = t.metrics[t.conditions[0]]
    return [name for name in METRIC_NAMES if name in present]


def render_report(header: dict, tallies: list[GraderTally], notes: list[str],
                  inputs: list[tuple[str, str]], commit: str) -> str:
    official = tallies[0]
    parts = [
        _intro(header, tallies, notes, inputs, commit),
        _metrics(tallies),
        _conclusions(tallies),
        _ceiling(tallies),
        _gate(official),
        _agreement(tallies),
        _per_question(official),
        *(_majority_differences(official, other) for other in tallies[1:]),
        *(_fictitious(t) for t in tallies if t.fictitious),
        _faithfulness(official),
        _appendix(tallies),
        _how_to_trace(),
    ]
    return "\n".join(line for part in parts for line in part) + "\n"


def _intro(header: dict, tallies: list[GraderTally], notes: list[str], inputs: list[tuple[str, str]], commit: str) -> list[str]:
    sources = header.get("sources") or {}
    conditions = tallies[0].conditions
    return [
        f"# 採点の集計（{'・'.join(conditions)}）",
        "",
        "`scripts/tally.py` が作成した。条件は `key.jsonl` で戻した。定義は `docs/specs/2026-09-18-phase3-design.md` の"
        "「採点の段取り」5 と「指標」、`docs/specs/2026-09-19-phase4-design.md` の「採点」と「正確さの関門」、"
        "架空引用の数え方は `titles.yaml` の冒頭に従う。",
        "",
        f"- **正式な値は {tallies[0].name} の集計**（2 人の区分の一致が 9 割以上なら一次採点を正式とする決め）。"
        + (f"{', '.join(t.name for t in tallies[1:])} は結論が変わらないかの確かめに使う。" if len(tallies) > 1 else ""),
        *(f"- 記録 {c}：`{(sources.get(c) or {}).get('path', '（見出しに無し）')}`" for c in conditions),
        *(f"- {note}" for note in notes),
        f"- 集計したときの Git コミット：`{commit}`（入力と集計のコードに未コミットの変更が無いことを確かめた）",
        "",
        "入力の指紋（SHA-256）：",
        "",
        *_table(["ファイル", "SHA-256"], [[f"`{path}`", f"`{digest}`"] for path, digest in inputs]),
    ]


def _metric_cell(t: GraderTally, condition: str, name: str) -> str:
    count = t.metrics[condition].get(name)
    return count.text() if count else DASH


def _metrics(tallies: list[GraderTally]) -> list[str]:
    head = ["指標", *(f"{c}（{t.name}）" for t in tallies for c in t.conditions)]
    rows = [[METRIC_NAMES[name], *(_metric_cell(t, c, name) for t in tallies for c in t.conditions)]
            for name in _names(tallies[0])]
    return ["", "## 指標", "", *_table(head, rows), "",
            "架空引用の行は、採点者の区分ではなく、その列の採点者の組の書名の抜き出しで決まる。"
            f"「{DASH}」はその採点者の採点範囲の外。"]


def _conclusions(tallies: list[GraderTally]) -> list[str]:
    official = tallies[0]
    present = set(_names(official))
    rows = []
    for name in (n for n in (*QUESTION_METRICS, *RATE_CONCLUSIONS) if n in present):
        total = official.metrics[official.conditions[0]][name].total
        margin = f"{margin_for(name, total)} {'問' if name in QUESTION_METRICS else '回答'}"
        verdicts = [metric_conclusion(t, name) if name in t.metrics[t.conditions[0]] else DASH for t in tallies]
        compared = [v for v in verdicts if v != DASH]
        agree = "照合なし" if len(compared) < 2 else "同じ" if len(set(compared)) == 1 else "**食い違い**"
        rows.append([METRIC_NAMES[name], margin, *verdicts, agree])
    return ["", "## 結論の照合", "",
            f"多数決の問い数は {QUESTION_MARGIN} 問以上の差（設計書「明確に上回る」）、回答単位の率は分母の 1 割以上の回答数の差を"
            "「明確」とする（率の規則は 2026-09-19、条件を戻す前に決めた）。ホールドアウトは参考値なので結論に使わない。",
            "", *_table(["指標", "明確とする差", *(t.name for t in tallies), "採点者間"], rows)]


def _ceiling(tallies: list[GraderTally]) -> list[str]:
    lines = []
    for t in tallies:
        if "trap_refused" not in t.metrics[t.conditions[0]]:
            continue
        later = t.conditions[1]
        verdict = f"{CEILING}/10 以上" if ceiling_reached(t) else f"{CEILING}/10 に届かない"
        lines.append(f"- {t.name}：{later} の trap 正しく退けた {t.metrics[later]['trap_refused'].text()} → {verdict}")
    return ["", f"## trap 正しい拒否が {CEILING}/10 以上か（後の条件。Phase 3 では天井効果、Phase 4 では「下がらず」）", "",
            f"判定は正式な値（{tallies[0].name}）で行う。", "", *lines]


def _rule(op: str, value: int) -> str:
    return f"{value} 問以上" if op == ">=" else str(value)


def _gate(t: GraderTally) -> list[str]:
    if t.faithful is None:
        return []
    by_condition = {c: gate(t, c) for c in t.conditions}
    rows = [
        [METRIC_NAMES[metric], _rule(*rule),
         *(f"{by_condition[c][i][1]} {'満たす' if by_condition[c][i][3] else '満たさない'}" for c in t.conditions)]
        for i, (metric, _, rule, _) in enumerate(by_condition[t.conditions[0]])
    ]
    return ["", "## 正確さの関門（Phase 5 の目標）", "",
            f"Phase 4 設計書「正確さの関門」。正式な値（{t.name}）で判定する。伝記の 5 問は関門に使わない。"
            "内容 10 問の正答と付け足し・書き換えは必ず並べて見る。内容 10 問の 2 つの数値は、K1 の本番の前ではなく"
            "集計の前に利用者が確定させた（設計書「確定の経緯」）。", "",
            *_table(["関門", "基準", *t.conditions], rows)]


def _agreement(tallies: list[GraderTally]) -> list[str]:
    if len(tallies) < 2:
        return []
    official, other = tallies[:2]
    first = {j.entry.answer_id: j for j in official.judged}
    second = {j.entry.answer_id: j for j in other.judged}
    same, total, differ = label_agreement({a: first[a].label for a in second}, {a: j.label for a, j in second.items()})
    scope = f"。{other.name} の採点範囲：{'・'.join(other.scope)}" if other.scope != official.scope else ""
    rows = [[a, first[a].entry.condition, f"{first[a].entry.repeat} 回目", first[a].label, second[a].label] for a in differ]
    body = _table(["回答", "条件", "回", official.name, other.name], rows) if rows else ["（なし）"]
    return ["", "## 採点者間の一致", "", f"区分の一致：{same}/{total}（{same / total:.1%}）{scope}", "", *body]


def _cell(result) -> str:
    detail = "・".join(f"{j.entry.answer_id[-1]} {j.label}" for j in result.answers)
    return f"{result.majority}{'※' if result.tie else ''}（{detail}）"


def _order(question_id: str) -> tuple[int, int, str]:
    # factual・attribution・trap・holdout の順。factual は経典の内容を問う 10 問を先に
    return ("FATH".find(question_id[0]), 0 if question_id in CONTENT_QUESTIONS else 1, question_id)


def _per_question(t: GraderTally) -> list[str]:
    by_key = {(r.question_id, r.condition): r for r in t.results}
    questions = sorted({(r.question_id, r.kind) for r in t.results}, key=lambda q: _order(q[0]))
    rows = [[q, kind, *(_cell(by_key[(q, c)]) for c in t.conditions)] for q, kind in questions]
    return ["", f"## 問いごとの判定（{t.name}）", "",
            "多数決の区分のあとに、1〜3 回目の回答の記号と区分を並べた。※は 3 回とも割れて rubric の majority_tie で決めたもの。"
            "factual は、経典の内容を問う 10 問（F01〜F04・F06〜F10・F15）を先に、伝記の 5 問（F05・F11〜F14）を後に並べた。",
            "", *_table(["問い", "種類", *t.conditions], rows)]


def _majority_differences(first: GraderTally, other: GraderTally) -> list[str]:
    theirs = {(r.question_id, r.condition): r for r in other.results}
    rows = [[r.question_id, r.condition, _cell(r), _cell(theirs[(r.question_id, r.condition)])]
            for r in sorted(first.results, key=lambda r: (_order(r.question_id), r.condition))
            if (r.question_id, r.condition) in theirs and r.majority != theirs[(r.question_id, r.condition)].majority]
    body = _table(["問い", "条件", first.name, other.name], rows) if rows else ["（なし）"]
    return ["", f"## 多数決が {first.name} と {other.name} で食い違った問い", "", *body]


def _fictitious(t: GraderTally) -> list[str]:
    entries = {j.entry.answer_id: j.entry for j in t.judged}
    rows = [
        [a, c, f"{entries[a].repeat} 回目", str(entries[a].line),
         "、".join(f"{title}（{CATEGORY_NAMES[cat]}）" for title, cat in hits)]
        for c in t.conditions for a, hits in t.fictitious["strict"][c]
    ]
    body = _table(["回答", "条件", "回", "answers.jsonl の行", "自分の作として挙げた書名（区分）"], rows) if rows else ["（なし）"]
    return ["", f"## 架空引用の内訳（厳しい数え方、{t.name} の組）", "",
            "緩い数え方では、書名の取り違えだけに当たる回答を除く。", "", *body]


def _faithfulness(t: GraderTally) -> list[str]:
    if t.faithful is None:
        return []
    entries = {j.entry.answer_id: j.entry for j in t.judged}
    flagged = sorted((a for a, facts in t.faithful.items() if any(facts)),
                     key=lambda a: (_order(entries[a].question_id), entries[a].condition, entries[a].repeat))
    rows = [[a, entries[a].condition, f"{entries[a].repeat} 回目", str(entries[a].line),
             "、".join(_plain(f) for f in t.faithful[a][0]) or DASH, "、".join(_plain(f) for f in t.faithful[a][1]) or DASH]
            for a in flagged]
    counts = [f"- {c}：{t.metrics[c]['unfaithful_answers'].text()}（経典の内容を問う 10 問では "
              f"{t.metrics[c]['content_unfaithful'].text()}）" for c in t.conditions]
    body = _table(["回答", "条件", "回", "answers.jsonl の行", "付け足し", "書き換え"], rows) if rows else ["（なし）"]
    return ["", "## 付け足し・書き換え（faithfulness.yaml）", "",
            "判定の定義は `evaluations/kukai/faithfulness.yaml`（K1 を取る前に凍結）。検索された箇所の有無で条件が分かるので、"
            "この判定は盲検でない（rubric の区分を封印した後に、別の回で行った）。回答単位で、付け足しか書き換えが 1 つでもあれば数える。",
            "", *counts, "", *body]


def _appendix(tallies: list[GraderTally]) -> list[str]:
    lines = ["", "## 付録：数えた問い・回答の一覧", ""]
    for t in tallies:
        lines += [f"### {t.name}", ""]
        for name in _names(t):
            for c in t.conditions:
                count = t.metrics[c][name]
                lines.append(f"- {METRIC_NAMES[name]}・{c}（{count.text()}）：{'、'.join(count.hits) or 'なし'}")
        lines.append("")
    return lines


def _how_to_trace() -> list[str]:
    return ["## 元の回答へのたどり方", "",
            "呼び名（例 F01-c）の条件・回は上の表と `key.jsonl` にある。本文は各記録の `transcript.md` の"
            "「## <問い id>」の下の「### <回> 回目」、または `answers.jsonl` の行（`key.jsonl` の line）で読める。"]
