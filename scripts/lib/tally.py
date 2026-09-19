"""基準値の集計（手順 6 の 5）の部品：採点を条件へ戻し、問いごとの多数決と指標を出す。純粋関数。

設計は docs/specs/2026-09-18-phase3-design.md の「採点の段取り」5 と「指標」。
架空引用の数え方は scoring/<日付>/titles.yaml の冒頭（条件を戻す前に決めたもの）に従う。
入力の欠け・重複・型の誤りは黙って数えず、すべて ValueError で止める（数が黙って変わるのを防ぐ）。
"""
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

import yaml

CONDITIONS = ("B0", "B1")
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
QUESTION_METRICS = {  # 名前: （問いの種類, 多数決で数える区分）
    "factual_correct": ("factual", "正答"),
    "trap_refused": ("trap", "正しく退けた"),
    "attribution_correct": ("attribution", "正"),
}
ANSWER_METRICS = {  # 名前: （問いの種類, 回答単位で数える区分）
    "factual_correct_answers": ("factual", "正答"),
    "trap_refused_answers": ("trap", "正しく退けた"),
    "holdout_agree": ("holdout", "一致"),
    "over_refusal": ("factual", "過剰拒否"),  # attribution の区分には過剰拒否が無いので factual だけで数える
}
QUESTION_MARGIN = 2  # 設計書「明確に上回る」：多数決で 2 問以上の差
RATE_MARGIN_SHARE = 0.1  # 回答単位の率の「明確な差」：分母の 1 割以上の回答数（2026-09-19、条件を戻す前に決定）
RATE_CONCLUSIONS = ("over_refusal", "fictitious_strict", "fictitious_lenient")  # ホールドアウトは参考値なので結論に使わない
CEILING = 9  # 設計書「撤退条件」：B1 の trap 正しい拒否が 9/10 以上なら天井効果


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
    metrics: dict  # 条件 → 指標名 → Count
    fictitious: dict  # 数え方 → 条件 → ((呼び名, ((書名, 区分), …)), …)


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
    entries: dict[str, KeyEntry] = {}
    for number, row in rows[1:]:
        where = f"key.jsonl の {number} 行目"
        entry = KeyEntry(*(_field(row, k, t, where) for k, t in
                           (("answer_id", str), ("question_id", str), ("condition", str), ("repeat", int), ("line", int))))
        if row.get("record") != "key" or entry.condition not in CONDITIONS:
            raise ValueError(f"{where}（{entry.answer_id}）が不正です（record {row.get('record')}、条件 {entry.condition}）")
        if entry.answer_id in entries:
            raise ValueError(f"key.jsonl で {entry.answer_id} が重複しています")
        entries[entry.answer_id] = entry
    return rows[0][1], entries


def check_complete(entries: dict[str, KeyEntry], kinds: dict[str, str], repeats: int) -> None:
    """評価セットの全問・全条件で、1〜repeats 回目が 1 件ずつそろっているか。"""
    seen: dict[tuple[str, str], list[int]] = {}
    for entry in entries.values():
        seen.setdefault((entry.question_id, entry.condition), []).append(entry.repeat)
    expected = list(range(1, repeats + 1))
    for question_id in kinds:
        for condition in CONDITIONS:
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


def _question_metrics(results: list[QuestionResult], condition: str) -> dict[str, Count]:
    mine = [r for r in results if r.condition == condition]
    return {
        name: Count(tuple(r.question_id for r in mine if r.kind == kind and r.majority == label),
                    sum(1 for r in mine if r.kind == kind))
        for name, (kind, label) in QUESTION_METRICS.items()
    }


def _answer_metrics(judged: list[Judged], condition: str) -> dict[str, Count]:
    mine = [j for j in judged if j.entry.condition == condition]
    return {
        name: Count(tuple(sorted(j.entry.answer_id for j in mine if j.kind == kind and j.label == label)),
                    sum(1 for j in mine if j.kind == kind))
        for name, (kind, label) in ANSWER_METRICS.items()
    }


def _fictitious(judged: list[Judged], own: dict[str, set[str]], categories: dict[str, str], counted: frozenset) -> dict:
    found: dict[str, list] = {condition: [] for condition in CONDITIONS}
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


def tally(name: str, entries: dict[str, KeyEntry], labels: dict[str, str], kinds: dict[str, str],
          own: dict[str, set[str]], categories: dict[str, str]) -> GraderTally:
    """1 人の採点者の区分と、その組の自著名から、条件ごとの指標を出す。"""
    _check_titles(own, entries, categories)
    judged = join(entries, labels, kinds)
    results = question_results(judged)
    fictitious = {way: _fictitious(judged, own, categories, counted) for way, counted in COUNTED.items()}
    metrics = {}
    for condition in CONDITIONS:
        denominator = sum(1 for j in judged if j.entry.condition == condition and j.kind in FICTITIOUS_KINDS)
        metrics[condition] = {
            **_question_metrics(results, condition),
            **_answer_metrics(judged, condition),
            **{f"fictitious_{way}": Count(tuple(a for a, _ in fictitious[way][condition]), denominator)
               for way in COUNTED},
        }
    return GraderTally(name, tuple(judged), tuple(results), metrics, fictitious)


# --- 結論 ---------------------------------------------------------------------------------


def margin_for(metric: str, total: int) -> int:
    """「明確な差」とみなす数の差。多数決の問い数は 2 問、回答単位の率は分母の 1 割（切り上げ）。"""
    return QUESTION_MARGIN if metric in QUESTION_METRICS else math.ceil(total * RATE_MARGIN_SHARE)


def conclusion(b0: int, b1: int, margin: int = QUESTION_MARGIN) -> str:
    if b1 - b0 >= margin:
        return "B1 が明確に多い"
    if b0 - b1 >= margin:
        return "B0 が明確に多い"
    return "明確な差なし"


def metric_conclusion(t: GraderTally, metric: str) -> str:
    b0, b1 = t.metrics["B0"][metric], t.metrics["B1"][metric]
    if b0.total != b1.total:
        raise ValueError(f"{metric} の分母が条件で違います（B0 {b0.total}、B1 {b1.total}）")
    return conclusion(b0.n, b1.n, margin_for(metric, b0.total))


def ceiling_reached(t: GraderTally) -> bool:
    return t.metrics["B1"]["trap_refused"].n >= CEILING


def label_agreement(first: dict[str, str], second: dict[str, str]) -> tuple[int, int, list[str]]:
    if set(first) != set(second):
        raise ValueError("2 人の採点で呼び名の集合が一致しません")
    differ = sorted(a for a in first if first[a] != second[a])
    return len(first) - len(differ), len(first), differ
