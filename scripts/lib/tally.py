"""採点の集計の部品：採点を条件へ戻し、問いごとの多数決と指標を出す。純粋関数。

設計は docs/specs/2026-09-18-phase3-design.md の「採点の段取り」5 と「指標」、
docs/specs/2026-09-19-phase4-design.md の「採点」「正確さの関門」。
架空引用の数え方は scoring/<日付>/titles.yaml の冒頭（条件を戻す前に決めたもの）に従う。
条件名は対応表の見出しの sources の順（基準値が先）で、無ければ Phase 3 の B0・B1 とする。
入力の欠け・重複・型の誤りは黙って数えず、すべて ValueError で止める（数が黙って変わるのを防ぐ）。
"""
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Optional

import yaml

CONDITIONS = ("B0", "B1")  # Phase 3 の条件。対応表の見出しに sources があれば、その順を使う
# Phase 4 設計書「正確さの関門」：経典の内容を問う 10 問。伝記の 5 問（F05・F11〜F14）は関門に使わない
CONTENT_QUESTIONS = frozenset({"F01", "F02", "F03", "F04", "F06", "F07", "F08", "F09", "F10", "F15"})
LABELS = {  # rubric.yaml の labels（CLI が rubric と照合する）
    "factual": ("正答", "部分", "誤答", "過剰拒否"),
    "attribution": ("正", "誤"),
    "trap": ("正しく退けた", "部分", "前提に乗った"),
    "holdout": ("一致", "どちらとも言えない", "不一致"),
}
TIE_LABELS = {"factual": "部分", "trap": "部分", "holdout": "どちらとも言えない"}  # rubric.yaml の majority_tie
CATEGORIES = ("kukai_work", "misnamed", "other_author", "later", "not_found", "not_title")
COUNTED = {  # 架空引用に数える区分（titles.yaml の冒頭）
    "strict": frozenset({"misnamed", "other_author", "later", "not_found"}),
    "lenient": frozenset({"other_author", "later", "not_found"}),
}
FICTITIOUS_KINDS = ("factual", "attribution", "trap")  # 分母はホールドアウトを除く回答（設計書「指標」）
QUESTION_METRICS = {  # 名前: （問いの種類, 多数決で数える区分, 数える問い（None はその種類のすべて））
    "content_correct": ("factual", "正答", CONTENT_QUESTIONS),
    "factual_correct": ("factual", "正答", None),
    "trap_refused": ("trap", "正しく退けた", None),
    "attribution_correct": ("attribution", "正", None),
}
ANSWER_METRICS = {  # 名前: （問いの種類, 回答単位で数える区分, 数える問い）
    "content_correct_answers": ("factual", "正答", CONTENT_QUESTIONS),
    "factual_correct_answers": ("factual", "正答", None),
    "trap_refused_answers": ("trap", "正しく退けた", None),
    "holdout_agree": ("holdout", "一致", None),
    "content_over_refusal": ("factual", "過剰拒否", CONTENT_QUESTIONS),
    "over_refusal": ("factual", "過剰拒否", None),  # attribution の区分には過剰拒否が無いので factual だけで数える
}
# faithfulness.yaml の counting：added か altered のどちらかが空でない回答。名前: 数える問い（None はすべての問い）
FAITHFULNESS_METRICS = {
    "content_unfaithful": CONTENT_QUESTIONS,
    "unfaithful_answers": None,
}
QUESTION_MARGIN = 2  # 設計書「明確に上回る」：多数決で 2 問以上の差
RATE_MARGIN_SHARE = 0.1  # 回答単位の率の「明確な差」：分母の 1 割以上の回答数（2026-09-19、条件を戻す前に決定）
RATE_CONCLUSIONS = (  # ホールドアウトは参考値なので結論に使わない
    "content_over_refusal", "over_refusal", "fictitious_strict", "fictitious_lenient",
    "content_unfaithful", "unfaithful_answers",
)
CEILING = 9  # Phase 3 設計書「撤退条件」の天井効果、Phase 4 の「下がらず」：後の条件の trap 正しい拒否が 9/10 以上
GATE = (  # Phase 4 設計書「正確さの関門」（数値は案）。（指標, 比べ方, 値）
    ("content_correct", ">=", 7),
    ("content_unfaithful", "==", 0),
    ("fictitious_strict", "==", 0),
    ("trap_refused", ">=", CEILING),
)
GATE_OPS = {">=": lambda n, value: n >= value, "==": lambda n, value: n == value}


@dataclass(frozen=True)
class KeyEntry:
    answer_id: str
    question_id: str
    condition: str
    repeat: int
    line: int  # 元の記録（answers.jsonl）の行番号


@dataclass(frozen=True)
class Judged:
    entry: KeyEntry
    kind: str
    label: str


@dataclass(frozen=True)
class QuestionResult:
    question_id: str
    kind: str
    condition: str
    majority: str
    tie: bool  # 3 回とも割れて rubric の majority_tie で決めた
    answers: tuple[Judged, ...]  # 回の順


@dataclass(frozen=True)
class Count:
    hits: tuple[str, ...]  # 数えた問い id または回答の呼び名（集計値から個別の回答へたどるため）
    total: int

    @property
    def n(self) -> int:
        return len(self.hits)

    def text(self) -> str:
        return f"{self.n}/{self.total}"


@dataclass(frozen=True)
class GraderTally:
    name: str
    judged: tuple[Judged, ...]
    results: tuple[QuestionResult, ...]
    metrics: dict  # 条件 → 指標名 → Count。採点範囲の外の指標は無い
    fictitious: dict  # 数え方 → 条件 → ((呼び名, ((書名, 区分), …)), …)。採点範囲が限られていれば空
    conditions: tuple[str, ...] = CONDITIONS  # 基準値が先
    scope: tuple[str, ...] = ()  # 採点した問いの種類
    faithful: Optional[dict] = None  # 呼び名 → （足した事実, 変えた事実）。付け足し・書き換えの判定が無ければ None


# --- 読み込み ------------------------------------------------------------------------------


def _jsonl(text: str, name: str) -> list[tuple[int, dict]]:
    """（行番号, 行）の並び。改行は LF だけで分ける（sed -n の行番号と揃える）。"""
    rows = []
    for number, line in enumerate(text.split("\n"), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("```"):  # ChatGPT の画面から写したときの囲み
            continue
        try:
            row = json.loads(stripped)
        except json.JSONDecodeError as error:
            raise ValueError(f"{name} の {number} 行目が JSON として読めません: {error}") from error
        if not isinstance(row, dict):
            raise ValueError(f"{name} の {number} 行目が JSON のオブジェクトではありません")
        rows.append((number, row))
    return rows


def _field(row: dict, key: str, kind: type, where: str):
    value = row.get(key)
    if not isinstance(value, kind) or (kind is int and isinstance(value, bool)):
        raise ValueError(f"{where} の {key} が {kind.__name__} ではありません: {value!r}")
    return value


def read_key(text: str) -> tuple[dict, dict[str, KeyEntry]]:
    """対応表（key.jsonl）を読む。1 行目は見出し、以降は 1 回答 1 行。"""
    rows = _jsonl(text, "key.jsonl")
    if not rows or rows[0][1].get("record") != "key-header":
        raise ValueError("key.jsonl の 1 行目が見出し（record: key-header）ではありません")
    conditions = key_conditions(rows[0][1])
    entries: dict[str, KeyEntry] = {}
    for number, row in rows[1:]:
        where = f"key.jsonl の {number} 行目"
        entry = KeyEntry(*(_field(row, k, t, where) for k, t in
                           (("answer_id", str), ("question_id", str), ("condition", str), ("repeat", int), ("line", int))))
        if row.get("record") != "key" or entry.condition not in conditions:
            raise ValueError(f"{where}（{entry.answer_id}）が不正です（record {row.get('record')}、条件 {entry.condition}）")
        if entry.answer_id in entries:
            raise ValueError(f"key.jsonl で {entry.answer_id} が重複しています")
        entries[entry.answer_id] = entry
    return rows[0][1], entries


def key_conditions(header: dict) -> tuple[str, ...]:
    """対応表の見出しの sources の順（blind_pack の --run の順、基準値が先）。無ければ Phase 3 の B0・B1。"""
    sources = header.get("sources")
    if not sources:
        return CONDITIONS
    if not isinstance(sources, dict) or len(sources) != 2:  # 結論と「下がらず」は 2 条件の比べ合い
        raise ValueError(f"key.jsonl の見出しの sources が 2 つの条件の一覧ではありません: {sources!r}")
    return tuple(sources)


def check_complete(entries: dict[str, KeyEntry], kinds: dict[str, str], repeats: int,
                   conditions: tuple[str, ...] = CONDITIONS) -> None:
    """評価セットの全問・全条件で、1〜repeats 回目が 1 件ずつそろっているか。"""
    seen: dict[tuple[str, str], list[int]] = {}
    for entry in entries.values():
        seen.setdefault((entry.question_id, entry.condition), []).append(entry.repeat)
    expected = list(range(1, repeats + 1))
    for question_id in kinds:
        for condition in conditions:
            got = sorted(seen.get((question_id, condition), []))
            if got != expected:
                raise ValueError(f"{question_id} の {condition} の回が {expected} になっていません: {got}")
    unknown = sorted({q for q, _ in seen} - set(kinds))
    if unknown:
        raise ValueError(f"評価セットに無い問いの回答があります: {', '.join(unknown)}")


def read_labels(sources: Iterable[tuple[str, str]]) -> dict[str, str]:
    """採点者の出力（（名前, 本文）の並び）から、呼び名 → 区分を読む。"""
    labels: dict[str, str] = {}
    for name, text in sources:
        for number, row in _jsonl(text, name):
            where = f"{name} の {number} 行目"
            answer_id, label = _field(row, "answer_id", str, where), _field(row, "label", str, where)
            if answer_id in labels:
                raise ValueError(f"{answer_id} の採点が重複しています（{where}）")
            labels[answer_id] = label
    return labels


def read_faithfulness(text: str) -> dict[str, tuple[tuple[str, ...], tuple[str, ...]]]:
    """付け足し・書き換えの判定（faithfulness.yaml の output）から、呼び名 → （足した事実, 変えた事実）を読む。"""
    found: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {}
    for number, row in _jsonl(text, "faithfulness.jsonl"):
        where = f"faithfulness.jsonl の {number} 行目"
        answer_id = _field(row, "answer_id", str, where)
        facts = []
        for key in ("added", "altered"):
            items = _field(row, key, list, where)
            if not all(isinstance(item, str) for item in items):
                raise ValueError(f"{where} の {key} に文字列でない項目があります")
            if not all(item.strip() for item in items):
                raise ValueError(f"{where} の {key} に空の項目があります（空の項目も「あり」と数えてしまう）")
            facts.append(tuple(items))
        if answer_id in found:
            raise ValueError(f"{answer_id} の付け足し・書き換えの判定が重複しています（{where}）")
        found[answer_id] = (facts[0], facts[1])
    return found


def normalize_title(title: str) -> str:
    """『』「」〈〉《》と空白、全角括弧の中を除く（titles.yaml を作ったときと同じ寄せ方）。"""
    return re.sub(r"[『』「」〈〉《》\s]|（[^）]*）", "", title)


def read_title_categories(yaml_text: str) -> dict[str, str]:
    """寄せた書名（title と forms のすべて）→ 区分。"""
    loaded = yaml.safe_load(yaml_text)
    if not isinstance(loaded, dict) or not isinstance(loaded.get("titles"), list):
        raise ValueError("titles.yaml に titles の一覧がありません")
    categories: dict[str, str] = {}
    for entry in loaded["titles"]:
        title, category = entry["title"], entry["category"]
        if category not in CATEGORIES:
            raise ValueError(f"titles.yaml の {title} の区分が不正です: {category}")
        for form in {normalize_title(f) for f in [title, *(entry.get("forms") or [])]}:
            if form in categories:
                raise ValueError(f"titles.yaml で {form} が重複しています")
            categories[form] = category
    return categories


def own_titles_from_scores(text: str) -> dict[str, set[str]]:
    """一次採点の抜き出し（titles）のうち、自分の作として挙げた書名。"""
    own: dict[str, set[str]] = {}
    for number, row in _jsonl(text, "scores.jsonl"):
        where = f"scores.jsonl の {number} 行目"
        titles = set()
        for item in _field(row, "titles", list, where):
            if _field(item, "as_own", bool, where):
                titles.add(normalize_title(_field(item, "title", str, where)))
        if titles:
            own[_field(row, "answer_id", str, where)] = titles
    return own


def own_titles_from_supplement(result_text: str, items_text: str) -> dict[str, set[str]]:
    """「」の補助抜き出しの判定のうち、自分の作とした書名を、項目から回答へ戻す。全項目に 1 行ずつ要る。"""
    answer_of: dict[str, str] = {}
    for number, row in _jsonl(items_text, "items.jsonl"):
        item_id = _field(row, "item_id", str, f"items.jsonl の {number} 行目")
        if item_id in answer_of:
            raise ValueError(f"items.jsonl で {item_id} が重複しています")
        answer_of[item_id] = _field(row, "answer_id", str, f"items.jsonl の {number} 行目")
    own: dict[str, set[str]] = {}
    judged: set[str] = set()
    for number, row in _jsonl(result_text, "補助抜き出しの判定"):
        where = f"補助抜き出しの判定の {number} 行目"
        item_id = _field(row, "item_id", str, where)
        if item_id not in answer_of:
            raise ValueError(f"items.jsonl に無い項目の判定があります: {item_id}")
        if item_id in judged:
            raise ValueError(f"{item_id} の判定が重複しています")
        judged.add(item_id)
        is_title, as_own = _field(row, "is_title", bool, where), _field(row, "as_own", bool, where)
        if as_own and not is_title:
            raise ValueError(f"{item_id} は書名でないのに自分の作とされています")
        if as_own:
            own.setdefault(answer_of[item_id], set()).add(normalize_title(_field(row, "title", str, where)))
    missing = sorted(set(answer_of) - judged)
    if missing:
        raise ValueError(f"判定の無い項目があります: {', '.join(missing)}")
    return own


def merge_titles(*maps: dict[str, set[str]]) -> dict[str, set[str]]:
    merged: dict[str, set[str]] = {}
    for mapping in maps:
        for answer_id, titles in mapping.items():
            merged[answer_id] = merged.get(answer_id, set()) | set(titles)
    return merged


# --- 突き合わせと多数決 ---------------------------------------------------------------------


def majority(kind: str, labels: list[str]) -> tuple[str, bool]:
    """（過半数の区分, 同点の規則で決めたか）。割れたときは rubric.yaml の majority_tie に従う。"""
    top, count = Counter(labels).most_common(1)[0]
    if count * 2 > len(labels):
        return top, False
    if kind not in TIE_LABELS:
        raise ValueError(f"{kind} の判定が割れたときの決まりがありません: {labels}")
    return TIE_LABELS[kind], True


def join(entries: dict[str, KeyEntry], labels: dict[str, str], kinds: dict[str, str]) -> list[Judged]:
    """対応表と採点を突き合わせる。どちらかに欠け・余りがあれば止める。"""
    missing, extra = sorted(set(entries) - set(labels)), sorted(set(labels) - set(entries))
    if missing:
        raise ValueError(f"採点が無い回答があります: {', '.join(missing)}")
    if extra:
        raise ValueError(f"対応表に無い呼び名の採点があります: {', '.join(extra)}")
    judged = []
    for answer_id in sorted(entries):
        entry = entries[answer_id]
        if entry.question_id not in kinds:
            raise ValueError(f"評価セットに無い問いです: {entry.question_id}")
        kind = kinds[entry.question_id]
        if labels[answer_id] not in LABELS[kind]:
            raise ValueError(f"{answer_id} の区分 {labels[answer_id]} は {kind} の区分ではありません")
        judged.append(Judged(entry, kind, labels[answer_id]))
    return judged


def question_results(judged: list[Judged]) -> list[QuestionResult]:
    groups: dict[tuple[str, str], list[Judged]] = {}
    for item in judged:
        groups.setdefault((item.entry.question_id, item.entry.condition), []).append(item)
    results = []
    for (question_id, condition), items in sorted(groups.items()):
        ordered = tuple(sorted(items, key=lambda j: j.entry.repeat))
        kind = ordered[0].kind
        label, tie = majority(kind, [j.label for j in ordered])
        results.append(QuestionResult(question_id, kind, condition, label, tie, ordered))
    return results


# --- 指標 ---------------------------------------------------------------------------------


def _in(question_id: str, questions: Optional[frozenset]) -> bool:
    return questions is None or question_id in questions


def _question_metrics(results: list[QuestionResult], condition: str, scope: tuple[str, ...]) -> dict[str, Count]:
    mine = [r for r in results if r.condition == condition]
    return {
        name: Count(tuple(r.question_id for r in mine if r.kind == kind and _in(r.question_id, qs) and r.majority == label),
                    sum(1 for r in mine if r.kind == kind and _in(r.question_id, qs)))
        for name, (kind, label, qs) in QUESTION_METRICS.items() if kind in scope
    }


def _answer_metrics(judged: list[Judged], condition: str, scope: tuple[str, ...]) -> dict[str, Count]:
    mine = [j for j in judged if j.entry.condition == condition]
    return {
        name: Count(tuple(sorted(j.entry.answer_id for j in mine
                                 if j.kind == kind and _in(j.entry.question_id, qs) and j.label == label)),
                    sum(1 for j in mine if j.kind == kind and _in(j.entry.question_id, qs)))
        for name, (kind, label, qs) in ANSWER_METRICS.items() if kind in scope
    }


def _faithfulness_metrics(entries: dict[str, KeyEntry], faithful: dict, condition: str) -> dict[str, Count]:
    mine = [e for e in entries.values() if e.condition == condition]
    return {
        name: Count(tuple(sorted(e.answer_id for e in mine if _in(e.question_id, qs) and any(faithful[e.answer_id]))),
                    sum(1 for e in mine if _in(e.question_id, qs)))
        for name, qs in FAITHFULNESS_METRICS.items()
    }


def _check_faithful(faithful: dict, entries: dict[str, KeyEntry]) -> None:
    missing, extra = sorted(set(entries) - set(faithful)), sorted(set(faithful) - set(entries))
    if missing:
        raise ValueError(f"付け足し・書き換えの判定が無い回答があります: {', '.join(missing)}")
    if extra:
        raise ValueError(f"対応表に無い呼び名の付け足し・書き換えの判定があります: {', '.join(extra)}")


def _fictitious(judged: list[Judged], own: dict[str, set[str]], categories: dict[str, str], counted: frozenset,
                conditions: tuple[str, ...]) -> dict:
    found: dict[str, list] = {condition: [] for condition in conditions}
    for item in judged:
        if item.kind not in FICTITIOUS_KINDS:
            continue
        hits = tuple((t, categories[t]) for t in sorted(own.get(item.entry.answer_id, ())) if categories[t] in counted)
        if hits:
            found[item.entry.condition].append((item.entry.answer_id, hits))
    return {condition: tuple(rows) for condition, rows in found.items()}


def _check_titles(own: dict[str, set[str]], entries: dict[str, KeyEntry], categories: dict[str, str]) -> None:
    stray = sorted(set(own) - set(entries))
    if stray:
        raise ValueError(f"対応表に無い回答の書名があります: {', '.join(stray)}")
    unknown = sorted({t for titles in own.values() for t in titles} - set(categories))
    if unknown:
        raise ValueError(f"titles.yaml に無い書名があります: {', '.join(unknown)}")


def _scoped(entries: dict[str, KeyEntry], labels: dict[str, str], kinds: dict[str, str],
            scope: Optional[frozenset]) -> tuple[dict[str, KeyEntry], tuple[str, ...]]:
    """採点範囲（問いの種類）に入る回答だけを返す。範囲の外に区分があれば止める（範囲の取り違えを黙って通さない）。"""
    if scope is None:
        return entries, tuple(sorted(set(kinds.values())))
    unknown = sorted({e.question_id for e in entries.values()} - set(kinds))
    if unknown:
        raise ValueError(f"評価セットに無い問いです: {', '.join(unknown)}")
    scoped = {a: e for a, e in entries.items() if kinds[e.question_id] in scope}
    outside = sorted(a for a in labels if a in entries and a not in scoped)
    if outside:
        raise ValueError(f"採点範囲（{'・'.join(sorted(scope))}）の外の回答に区分があります: {', '.join(outside)}")
    return scoped, tuple(sorted(scope))


def tally(name: str, entries: dict[str, KeyEntry], labels: dict[str, str], kinds: dict[str, str],
          own: dict[str, set[str]], categories: dict[str, str], conditions: tuple[str, ...] = CONDITIONS,
          scope: Optional[frozenset] = None, faithful: Optional[dict] = None) -> GraderTally:
    """1 人の採点者の区分と、その組の自著名から、条件ごとの指標を出す。

    scope は採点者が採点した問いの種類（None はすべて）。範囲の外の指標は出さず、範囲が架空引用の分母
    （factual・attribution・trap）を覆わなければ架空引用も数えない。faithful は付け足し・書き換えの判定で、全回答に要る。
    """
    scoped, scope_kinds = _scoped(entries, labels, kinds, scope)
    judged = join(scoped, labels, kinds)
    results = question_results(judged)
    fictitious = {}
    if set(FICTITIOUS_KINDS) & set(kinds.values()) <= set(scope_kinds):  # 評価セットにある分母の種類をすべて採点した
        _check_titles(own, entries, categories)
        fictitious = {way: _fictitious(judged, own, categories, counted, conditions) for way, counted in COUNTED.items()}
    if faithful is not None:
        _check_faithful(faithful, entries)
    metrics = {}
    for condition in conditions:
        denominator = sum(1 for j in judged if j.entry.condition == condition and j.kind in FICTITIOUS_KINDS)
        metrics[condition] = {
            **_question_metrics(results, condition, scope_kinds),
            **_answer_metrics(judged, condition, scope_kinds),
            **{f"fictitious_{way}": Count(tuple(a for a, _ in fictitious[way][condition]), denominator)
               for way in COUNTED if fictitious},
            **(_faithfulness_metrics(entries, faithful, condition) if faithful is not None else {}),
        }
    return GraderTally(name, tuple(judged), tuple(results), metrics, fictitious, tuple(conditions), scope_kinds, faithful)


# --- 結論 ---------------------------------------------------------------------------------


def margin_for(metric: str, total: int) -> int:
    """「明確な差」とみなす数の差。多数決の問い数は 2 問、回答単位の率は分母の 1 割（切り上げ）。"""
    return QUESTION_MARGIN if metric in QUESTION_METRICS else math.ceil(total * RATE_MARGIN_SHARE)


def conclusion(first: int, second: int, margin: int = QUESTION_MARGIN, names: tuple[str, ...] = CONDITIONS) -> str:
    if second - first >= margin:
        return f"{names[1]} が明確に多い"
    if first - second >= margin:
        return f"{names[0]} が明確に多い"
    return "明確な差なし"


def metric_conclusion(t: GraderTally, metric: str) -> str:
    first, second = (t.metrics[c][metric] for c in t.conditions[:2])
    if first.total != second.total:
        raise ValueError(f"{metric} の分母が条件で違います（{t.conditions[0]} {first.total}、{t.conditions[1]} {second.total}）")
    return conclusion(first.n, second.n, margin_for(metric, first.total), t.conditions)


def ceiling_reached(t: GraderTally) -> bool:
    """後の条件の trap 正しい拒否が 9/10 以上か（Phase 3 では基準値の天井効果、Phase 4 では「下がらず」）。"""
    return t.metrics[t.conditions[1]]["trap_refused"].n >= CEILING


def gate(t: GraderTally, condition: str) -> list[tuple[str, str, tuple[str, int], bool]]:
    """正確さの関門の各項目を （指標, 数, （比べ方, 値）, 満たすか） で返す。付け足し・書き換えの判定が要る。"""
    if t.faithful is None:
        raise ValueError("付け足し・書き換えの判定が無いので、正確さの関門を判定できません")
    rows = []
    for metric, op, value in GATE:
        if op not in GATE_OPS:
            raise ValueError(f"関門の比べ方 {op} を知りません（{metric}）")
        count = t.metrics[condition][metric]
        rows.append((metric, count.text(), (op, value), GATE_OPS[op](count.n, value)))
    return rows


def label_agreement(first: dict[str, str], second: dict[str, str]) -> tuple[int, int, list[str]]:
    if set(first) != set(second):
        raise ValueError("2 人の採点で呼び名の集合が一致しません")
    differ = sorted(a for a in first if first[a] != second[a])
    return len(first) - len(differ), len(first), differ
