"""tally: 採点を条件へ戻し、問いごとの多数決と指標を出す部品を検証する。"""
import json

import pytest

from scripts.lib.tally import (
    Count,
    KeyEntry,
    ceiling_reached,
    check_complete,
    conclusion,
    join,
    label_agreement,
    majority,
    margin_for,
    merge_titles,
    metric_conclusion,
    normalize_title,
    own_titles_from_scores,
    own_titles_from_supplement,
    question_results,
    read_key,
    read_labels,
    read_title_categories,
    tally,
)

KINDS = {"F01": "factual", "A01": "attribution", "T01": "trap", "H01": "holdout"}
# (呼び名, 条件, 回, 区分)。記号は条件に依らず混ざっている
ROWS = [
    ("F01-a", "B0", 1, "正答"), ("F01-b", "B1", 1, "正答"), ("F01-c", "B0", 2, "正答"),
    ("F01-d", "B1", 2, "誤答"), ("F01-e", "B0", 3, "部分"), ("F01-f", "B1", 3, "過剰拒否"),
    ("A01-a", "B0", 1, "正"), ("A01-b", "B0", 2, "誤"), ("A01-c", "B0", 3, "誤"),
    ("A01-d", "B1", 1, "正"), ("A01-e", "B1", 2, "正"), ("A01-f", "B1", 3, "誤"),
    ("T01-a", "B0", 1, "前提に乗った"), ("T01-b", "B1", 1, "正しく退けた"), ("T01-c", "B0", 2, "前提に乗った"),
    ("T01-d", "B1", 2, "部分"), ("T01-e", "B0", 3, "前提に乗った"), ("T01-f", "B1", 3, "正しく退けた"),
    ("H01-a", "B0", 1, "一致"), ("H01-b", "B0", 2, "不一致"), ("H01-c", "B0", 3, "どちらとも言えない"),
    ("H01-d", "B1", 1, "一致"), ("H01-e", "B1", 2, "一致"), ("H01-f", "B1", 3, "一致"),
]
TITLES_YAML = """\
titles:
  - title: 即身成仏義
    forms: [即身成仏義, 即身成佛義]
    category: kukai_work
  - title: 即身成仏論
    forms: [即身成仏論]
    category: misnamed
  - title: 六大縁起章
    forms: [六大縁起章]
    category: not_found
  - title: 自序
    forms: [自序]
    category: not_title
  - title: 密教具足之儀
    forms: [密教具足之儀]
    category: not_found
"""
OWN = {
    "F01-a": {"即身成仏義"},  # B0、空海の著作 → 数えない
    "F01-d": {"即身成仏論"},  # B1、取り違え → 厳しい数え方だけで数える
    "T01-a": {"六大縁起章"},  # B0、見当たらない → 数える
    "A01-b": {"自序"},  # B0、書名でない → 除く
    "H01-a": {"密教具足之儀"},  # ホールドアウトは分母の外 → 数えない
}


def jsonl(rows: list) -> str:
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)


def key_text(rows=ROWS, header=True) -> str:
    head = [{"record": "key-header", "seed": 7}] if header else []
    body = [
        {"record": "key", "answer_id": aid, "question_id": aid[:3], "letter": aid[-1],
         "condition": cond, "repeat": rep, "line": 2 + i}
        for i, (aid, cond, rep, _) in enumerate(rows)
    ]
    return jsonl(head + body)


def labels() -> dict[str, str]:
    return {aid: label for aid, _, _, label in ROWS}


def entries() -> dict[str, KeyEntry]:
    return read_key(key_text())[1]


def run_tally(own=None):
    return tally("test", entries(), labels(), KINDS, OWN if own is None else own, read_title_categories(TITLES_YAML))


# --- 対応表 -------------------------------------------------------------------------------


def test_read_key_returns_header_and_entries():
    header, found = read_key(key_text())
    assert header["seed"] == 7
    assert found["T01-d"] == KeyEntry("T01-d", "T01", "B1", 2, 17)
    assert len(found) == len(ROWS)


def test_read_key_requires_header_first():
    with pytest.raises(ValueError, match="見出し"):
        read_key(key_text(header=False))


def test_read_key_rejects_duplicate_unknown_condition_and_bad_types():
    with pytest.raises(ValueError, match="F01-a"):
        read_key(key_text(ROWS + [ROWS[0]]))
    with pytest.raises(ValueError, match="B2"):
        read_key(key_text([("F01-a", "B2", 1, "正答")]))
    with pytest.raises(ValueError, match="repeat"):
        read_key(key_text([("F01-a", "B0", "1", "正答")]))
    with pytest.raises(ValueError, match="repeat"):
        read_key(key_text([("F01-a", "B0", True, "正答")]))


def test_read_key_reports_bad_json_and_non_objects():
    with pytest.raises(ValueError, match="2 行目"):
        read_key(jsonl([{"record": "key-header"}]) + "{broken\n")
    with pytest.raises(ValueError, match="オブジェクト"):
        read_key(jsonl([{"record": "key-header"}, [1, 2]]))


def test_check_complete_accepts_full_set():
    check_complete(entries(), KINDS, 3)


def test_check_complete_rejects_missing_or_doubled_repeats():
    rows = [r for r in ROWS if r[0] != "F01-f"]
    with pytest.raises(ValueError, match="F01 の B1"):
        check_complete(read_key(key_text(rows))[1], KINDS, 3)
    doubled = [r if r[0] != "F01-f" else ("F01-f", "B1", 2, "過剰拒否") for r in ROWS]
    with pytest.raises(ValueError, match=r"\[1, 2, 2\]"):
        check_complete(read_key(key_text(doubled))[1], KINDS, 3)


def test_check_complete_rejects_missing_and_unknown_questions():
    with pytest.raises(ValueError, match="Z01"):
        check_complete(entries(), {**KINDS, "Z01": "factual"}, 3)
    with pytest.raises(ValueError, match="H01"):
        check_complete(entries(), {k: v for k, v in KINDS.items() if k != "H01"}, 3)


# --- 採点の読み込み -----------------------------------------------------------------------


def test_read_labels_joins_files_and_rejects_duplicates():
    first = jsonl([{"answer_id": "F01-a", "label": "正答", "reason": "r"}])
    second = jsonl([{"answer_id": "F01-b", "label": "誤答", "reason": "r"}])
    assert read_labels([("1", first), ("2", second)]) == {"F01-a": "正答", "F01-b": "誤答"}
    with pytest.raises(ValueError, match="F01-a"):
        read_labels([("1", first), ("2", first)])


def test_read_labels_rejects_missing_label_and_bad_json():
    with pytest.raises(ValueError, match="label"):
        read_labels([("x", jsonl([{"answer_id": "F01-a"}]))])
    with pytest.raises(ValueError, match="result-9.txt の 1 行目"):
        read_labels([("result-9.txt", "{broken\n")])


def test_read_labels_ignores_blank_lines_and_code_fences():
    text = "```json\n" + jsonl([{"answer_id": "F01-a", "label": "正答"}]) + "\n```\n"
    assert read_labels([("x", text)]) == {"F01-a": "正答"}


def test_lines_are_split_on_lf_only():
    text = jsonl([{"answer_id": "F01-a", "label": "正答", "reason": "a b"}]) + "{broken\n"
    with pytest.raises(ValueError, match="2 行目"):
        read_labels([("x", text)])


# --- 書名 ---------------------------------------------------------------------------------


def test_normalize_title_strips_brackets_spaces_and_readings():
    assert normalize_title("『即身成仏義』") == "即身成仏義"
    assert normalize_title("十住心論 略本") == "十住心論略本"
    assert normalize_title("御遺告（ごゆいごう）") == "御遺告"


def test_read_title_categories_maps_every_form():
    categories = read_title_categories(TITLES_YAML)
    assert categories["即身成佛義"] == "kukai_work"
    assert categories["自序"] == "not_title"


def test_read_title_categories_validates():
    with pytest.raises(ValueError, match="区分"):
        read_title_categories("titles:\n  - title: X\n    forms: [X]\n    category: unknown\n")
    with pytest.raises(ValueError, match="重複"):
        read_title_categories(TITLES_YAML + "  - title: 別名\n    forms: [自序]\n    category: not_title\n")
    with pytest.raises(ValueError, match="titles の一覧"):
        read_title_categories("")


def test_own_titles_from_scores_keeps_only_own_and_normalizes():
    text = jsonl([
        {"answer_id": "F01-a", "label": "正答", "titles": [
            {"title": "『即身成仏義』", "where": "出典行", "as_own": True},
            {"title": "大日経", "where": "本文", "as_own": False},
        ], "quotes": []},
        {"answer_id": "F01-b", "label": "正答", "titles": [], "quotes": []},
    ])
    assert own_titles_from_scores(text) == {"F01-a": {"即身成仏義"}}


def test_own_titles_from_scores_rejects_missing_fields_and_string_booleans():
    with pytest.raises(ValueError, match="titles"):
        own_titles_from_scores(jsonl([{"answer_id": "F01-a", "label": "正答"}]))
    with pytest.raises(ValueError, match="as_own"):
        own_titles_from_scores(jsonl([{"answer_id": "F01-a", "titles": [{"title": "X", "as_own": "false"}]}]))


ITEMS = jsonl([
    {"item_id": "K001", "answer_id": "T01-a", "string": "六大縁起章", "contexts": []},
    {"item_id": "K002", "answer_id": "T01-b", "string": "大日経", "contexts": []},
])


def judgment(item_id, is_title=True, as_own=True, title="六大縁起章"):
    return {"item_id": item_id, "title": title, "is_title": is_title, "as_own": as_own, "reason": "r"}


def test_own_titles_from_supplement_maps_items_to_answers():
    result = jsonl([judgment("K001"), judgment("K002", as_own=False, title="大日経")])
    assert own_titles_from_supplement(result, ITEMS) == {"T01-a": {"六大縁起章"}}


def test_supplement_requires_exactly_one_judgment_per_item():
    with pytest.raises(ValueError, match="判定の無い項目.*K002"):
        own_titles_from_supplement(jsonl([judgment("K001")]), ITEMS)
    with pytest.raises(ValueError, match="K001 の判定が重複"):
        own_titles_from_supplement(jsonl([judgment("K001"), judgment("K001"), judgment("K002")]), ITEMS)
    with pytest.raises(ValueError, match="K009"):
        own_titles_from_supplement(jsonl([judgment("K009")]), ITEMS)
    with pytest.raises(ValueError, match="items.jsonl で K001 が重複"):
        own_titles_from_supplement(jsonl([judgment("K001")]), ITEMS + ITEMS)


def test_supplement_rejects_string_booleans_and_own_non_titles():
    with pytest.raises(ValueError, match="as_own"):
        own_titles_from_supplement(jsonl([judgment("K001", as_own="false"), judgment("K002")]), ITEMS)
    with pytest.raises(ValueError, match="書名でないのに"):
        own_titles_from_supplement(jsonl([judgment("K001", is_title=False), judgment("K002")]), ITEMS)


def test_merge_titles_unions_per_answer():
    merged = merge_titles({"F01-a": {"甲"}}, {"F01-a": {"乙"}, "F01-b": {"丙"}})
    assert merged == {"F01-a": {"甲", "乙"}, "F01-b": {"丙"}}


# --- 多数決と突き合わせ ------------------------------------------------------------------


def test_majority_takes_two_of_three():
    assert majority("factual", ["正答", "部分", "正答"]) == ("正答", False)


def test_majority_tie_follows_rubric():
    assert majority("factual", ["正答", "誤答", "過剰拒否"]) == ("部分", True)
    assert majority("trap", ["正しく退けた", "部分", "前提に乗った"]) == ("部分", True)
    assert majority("holdout", ["一致", "不一致", "どちらとも言えない"]) == ("どちらとも言えない", True)


def test_majority_tie_without_rule_is_an_error():
    with pytest.raises(ValueError, match="attribution"):
        majority("attribution", ["正", "誤"])


def test_join_checks_coverage_and_labels():
    assert len(join(entries(), labels(), KINDS)) == len(ROWS)
    with pytest.raises(ValueError, match="採点が無い"):
        join(entries(), {k: v for k, v in labels().items() if k != "F01-a"}, KINDS)
    with pytest.raises(ValueError, match="対応表に無い"):
        join(entries(), {**labels(), "Z01-a": "正答"}, KINDS)
    with pytest.raises(ValueError, match="F01-a"):
        join(entries(), {**labels(), "F01-a": "正"}, KINDS)
    with pytest.raises(ValueError, match="F01"):
        join(entries(), labels(), {k: v for k, v in KINDS.items() if k != "F01"})


def test_question_results_orders_answers_and_marks_ties():
    results = {(r.question_id, r.condition): r for r in question_results(join(entries(), labels(), KINDS))}
    b1 = results[("F01", "B1")]
    assert (b1.majority, b1.tie) == ("部分", True)
    assert [j.entry.answer_id for j in b1.answers] == ["F01-b", "F01-d", "F01-f"]
    assert (results[("A01", "B0")].majority, results[("A01", "B0")].tie) == ("誤", False)
    assert results[("T01", "B1")].majority == "正しく退けた"


# --- 指標 ---------------------------------------------------------------------------------


def test_question_level_metrics():
    m = run_tally().metrics
    assert m["B0"]["factual_correct"] == Count(("F01",), 1)
    assert m["B1"]["factual_correct"] == Count((), 1)
    assert m["B0"]["trap_refused"].n == 0 and m["B1"]["trap_refused"] == Count(("T01",), 1)
    assert m["B1"]["attribution_correct"] == Count(("A01",), 1)


def test_answer_level_metrics():
    m = run_tally().metrics
    assert m["B0"]["factual_correct_answers"] == Count(("F01-a", "F01-c"), 3)
    assert m["B1"]["over_refusal"] == Count(("F01-f",), 3)
    assert m["B1"]["trap_refused_answers"] == Count(("T01-b", "T01-f"), 3)
    assert m["B0"]["holdout_agree"] == Count(("H01-a",), 3)
    assert m["B1"]["holdout_agree"].n == 3


def test_fictitious_citation_strict_and_lenient():
    result = run_tally()
    m = result.metrics
    assert m["B0"]["fictitious_strict"] == Count(("T01-a",), 9)
    assert m["B1"]["fictitious_strict"] == Count(("F01-d",), 9)
    assert m["B0"]["fictitious_lenient"] == Count(("T01-a",), 9)
    assert m["B1"]["fictitious_lenient"] == Count((), 9)
    assert result.fictitious["strict"]["B1"] == (("F01-d", (("即身成仏論", "misnamed"),)),)


def test_fictitious_uses_forms():
    assert run_tally({"F01-b": {"即身成佛義"}}).metrics["B1"]["fictitious_strict"].n == 0


def test_titles_must_be_in_table_and_belong_to_known_answers():
    with pytest.raises(ValueError, match="titles.yaml"):
        run_tally({"F01-a": {"未登録の書"}})
    with pytest.raises(ValueError, match="対応表に無い回答"):
        run_tally({"T01-C": {"六大縁起章"}})


def test_count_text():
    assert Count(("a", "b"), 8).n == 2
    assert Count(("a", "b"), 8).text() == "2/8"


# --- 結論と採点者間の一致 -----------------------------------------------------------------


def test_conclusion_uses_margin():
    assert conclusion(3, 5) == "B1 が明確に多い"
    assert conclusion(5, 3) == "B0 が明確に多い"
    assert conclusion(3, 4) == "明確な差なし"
    assert conclusion(10, 18, margin=9) == "明確な差なし"
    assert conclusion(10, 19, margin=9) == "B1 が明確に多い"


def test_margin_for_question_and_rate_metrics():
    assert margin_for("trap_refused", 10) == 2
    assert margin_for("fictitious_strict", 90) == 9
    assert margin_for("over_refusal", 45) == 5


def test_metric_conclusion_and_ceiling():
    result = run_tally()
    assert metric_conclusion(result, "trap_refused") == "明確な差なし"  # 0 と 1 で 1 問差
    assert metric_conclusion(result, "fictitious_strict") == "明確な差なし"  # 分母 9 の 1 割は 1 回答、差は 0
    assert not ceiling_reached(result)


def test_metric_conclusion_requires_equal_denominators():
    result = run_tally()
    result.metrics["B1"]["trap_refused"] = Count(("T01",), 2)
    with pytest.raises(ValueError, match="分母"):
        metric_conclusion(result, "trap_refused")


def test_label_agreement_lists_differences():
    assert label_agreement({"a": "正", "b": "誤"}, {"a": "正", "b": "正"}) == (1, 2, ["b"])
    with pytest.raises(ValueError, match="呼び名"):
        label_agreement({"a": "正"}, {"b": "正"})
