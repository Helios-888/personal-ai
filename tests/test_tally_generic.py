"""tally の一般化（Phase 4）：条件名を記録から取り、内容 10 問を分け、採点範囲の限られた採点者と、付け足し・書き換えを数える。"""
import json
from dataclasses import replace

import pytest

from scripts.lib.tally import (
    Count,
    KeyEntry,
    ceiling_reached,
    check_complete,
    conclusion,
    gate,
    metric_conclusion,
    read_faithfulness,
    read_key,
    read_title_categories,
    tally,
)

KINDS = {"F01": "factual", "F05": "factual", "A01": "attribution", "T01": "trap", "H01": "holdout"}
CONDITIONS = ("B1", "K1")  # 基準値が先
# (問い, 条件) → 1〜3 回目の区分。F01 は経典の内容を問う問い、F05 は伝記の問い（Phase 4 設計書「正確さの関門」）
LABELS_BY = {
    ("F01", "B1"): ("過剰拒否", "過剰拒否", "誤答"),
    ("F01", "K1"): ("正答", "正答", "部分"),
    ("F05", "B1"): ("過剰拒否", "誤答", "誤答"),
    ("F05", "K1"): ("正答", "正答", "正答"),
    ("A01", "B1"): ("正", "正", "誤"),
    ("A01", "K1"): ("正", "正", "正"),
    ("T01", "B1"): ("正しく退けた", "正しく退けた", "正しく退けた"),
    ("T01", "K1"): ("正しく退けた", "正しく退けた", "部分"),
    ("H01", "B1"): ("一致", "不一致", "一致"),
    ("H01", "K1"): ("一致", "一致", "一致"),
}
TITLES_YAML = """\
titles:
  - title: 即身成仏義
    forms: [即身成仏義]
    category: kukai_work
"""


def answer_id(question_id: str, condition: str, repeat: int) -> str:
    return f"{question_id}-{'ace'[repeat - 1] if condition == 'B1' else 'bdf'[repeat - 1]}"


def rows() -> list[tuple[str, str, int, str]]:
    return [(answer_id(q, c, r), c, r, label)
            for (q, c), labels in LABELS_BY.items() for r, label in enumerate(labels, start=1)]


def key_text(header_sources=True) -> str:
    header = {"record": "key-header", "seed": 7}
    if header_sources:
        header["sources"] = {c: {"path": f"x/{c}"} for c in CONDITIONS}
    body = [{"record": "key", "answer_id": a, "question_id": a[:3], "letter": a[-1], "condition": c, "repeat": r,
             "line": 2 + i} for i, (a, c, r, _) in enumerate(rows())]
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in [header, *body])


def entries() -> dict[str, KeyEntry]:
    return read_key(key_text())[1]


def labels(only_kind=None) -> dict[str, str]:
    return {a: label for a, _, _, label in rows() if only_kind is None or KINDS[a[:3]] == only_kind}


def faithful(**changes) -> dict[str, tuple[tuple[str, ...], tuple[str, ...]]]:
    return {a: changes.get(a.replace("-", "_"), ((), ())) for a, _, _, _ in rows()}


def run(**kwargs):
    return tally("甲", entries(), kwargs.pop("labels", labels()), KINDS, kwargs.pop("own", {}),
                 read_title_categories(TITLES_YAML), conditions=CONDITIONS, **kwargs)


# --- 条件名 -------------------------------------------------------------------------------


def test_read_key_takes_the_conditions_from_the_header_sources():
    header, found = read_key(key_text())
    assert list(header["sources"]) == ["B1", "K1"]
    assert found["F01-b"] == KeyEntry("F01-b", "F01", "K1", 1, 5)


def test_read_key_rejects_a_condition_missing_from_the_header_sources():
    text = key_text().replace('"condition": "K1"', '"condition": "B0"', 1)
    with pytest.raises(ValueError, match="B0"):
        read_key(text)


def test_read_key_without_sources_keeps_the_phase3_conditions():
    with pytest.raises(ValueError, match="K1"):
        read_key(key_text(header_sources=False))


def test_check_complete_checks_the_given_conditions():
    check_complete(entries(), KINDS, 3, CONDITIONS)
    with pytest.raises(ValueError, match="K1"):
        check_complete({a: e for a, e in entries().items() if a != "T01-f"}, KINDS, 3, CONDITIONS)


def test_conclusion_and_ceiling_use_the_condition_names():
    assert conclusion(0, 2, names=CONDITIONS) == "K1 が明確に多い"
    assert conclusion(2, 0, names=CONDITIONS) == "B1 が明確に多い"
    result = run()
    assert result.conditions == CONDITIONS
    assert metric_conclusion(result, "content_correct") == "明確な差なし"  # 0 と 1 で 1 問差
    assert not ceiling_reached(result)  # K1 の trap は 1/1 で、9 に届かない


# --- 経典の内容を問う 10 問 ------------------------------------------------------------------


def test_content_questions_are_counted_apart_from_the_biography_questions():
    m = run().metrics
    assert m["K1"]["content_correct"] == Count(("F01",), 1)
    assert m["K1"]["factual_correct"] == Count(("F01", "F05"), 2)
    assert m["B1"]["content_over_refusal"] == Count(("F01-a", "F01-c"), 3)
    assert m["B1"]["over_refusal"] == Count(("F01-a", "F01-c", "F05-a"), 6)
    assert m["K1"]["content_correct_answers"] == Count(("F01-b", "F01-d"), 3)


# --- 採点範囲の限られた採点者（Astra は trap の束だけ） ----------------------------------------


def test_a_grader_limited_to_trap_gets_only_trap_metrics_and_no_titles():
    result = run(labels=labels("trap"), scope=frozenset({"trap"}))
    assert result.scope == ("trap",)
    assert set(result.metrics["K1"]) == {"trap_refused", "trap_refused_answers"}
    assert result.metrics["K1"]["trap_refused"] == Count(("T01",), 1)
    assert result.fictitious == {}
    assert {j.kind for j in result.judged} == {"trap"}


def test_a_limited_grader_must_still_cover_every_answer_of_its_kinds():
    partial = {a: label for a, label in labels("trap").items() if a != "T01-f"}
    with pytest.raises(ValueError, match="T01-f"):
        run(labels=partial, scope=frozenset({"trap"}))


def test_a_limited_grader_may_not_label_answers_outside_its_scope():
    with pytest.raises(ValueError, match="F01"):
        run(scope=frozenset({"trap"}))


# --- 付け足し・書き換え（faithfulness.yaml） ------------------------------------------------


def test_unfaithful_answers_are_counted_per_condition_and_for_content_questions():
    result = run(faithful=faithful(F01_b=(("鋳造",), ()), H01_a=((), ("797 年を 777 年",))))
    m = result.metrics
    assert m["K1"]["unfaithful_answers"] == Count(("F01-b",), 15)
    assert m["B1"]["unfaithful_answers"] == Count(("H01-a",), 15)
    assert m["K1"]["content_unfaithful"] == Count(("F01-b",), 3)
    assert m["B1"]["content_unfaithful"] == Count((), 3)
    assert result.faithful["F01-b"] == (("鋳造",), ())


def test_without_the_faithfulness_judgment_there_are_no_such_metrics():
    assert "unfaithful_answers" not in run().metrics["K1"]


def test_the_faithfulness_judgment_must_cover_every_answer():
    partial = {a: v for a, v in faithful().items() if a != "H01-f"}
    with pytest.raises(ValueError, match="H01-f"):
        run(faithful=partial)


def test_read_faithfulness_validates_rows():
    text = '{"answer_id": "F01-a", "added": ["x"], "altered": []}\n{"answer_id": "F01-b", "added": [], "altered": []}\n'
    assert read_faithfulness(text) == {"F01-a": (("x",), ()), "F01-b": ((), ())}
    with pytest.raises(ValueError, match="重複"):
        read_faithfulness(text + '{"answer_id": "F01-a", "added": [], "altered": []}\n')
    with pytest.raises(ValueError, match="added"):
        read_faithfulness('{"answer_id": "F01-a", "added": "x", "altered": []}\n')
    with pytest.raises(ValueError, match="altered"):
        read_faithfulness('{"answer_id": "F01-a", "added": [], "altered": [1]}\n')


# --- 正確さの関門（Phase 5 の目標） ----------------------------------------------------------


def test_gate_lists_each_rule_with_the_count_and_whether_it_is_met():
    result = run(faithful=faithful(F01_b=(("鋳造",), ())))
    rows_ = {name: (text, met) for name, text, _, met in gate(result, "K1")}
    assert rows_ == {
        "content_correct": ("1/1", False),
        "content_unfaithful": ("1/3", False),
        "fictitious_strict": ("0/12", True),
        "trap_refused": ("1/1", False),
    }


def test_gate_needs_the_faithfulness_judgment():
    with pytest.raises(ValueError, match="付け足し"):
        gate(run(), "K1")


# --- 入力の誤りは黙って通さない ------------------------------------------------------------


def test_read_key_rejects_sources_that_are_not_exactly_two_conditions():
    # 結論と「下がらず」は 2 条件の比べ合い。3 つ目があると黙って外れるので止める（code-reviewer M3）
    with pytest.raises(ValueError, match="sources"):
        read_key('{"record": "key-header", "sources": ["B1", "K1"]}\n')
    with pytest.raises(ValueError, match="sources"):
        read_key('{"record": "key-header", "sources": {"B1": {}}}\n')
    with pytest.raises(ValueError, match="sources"):
        read_key('{"record": "key-header", "sources": {"B1": {}, "K1": {}, "K2": {}}}\n')


def test_read_faithfulness_rejects_blank_facts():
    # 空の項目も「付け足しあり」に数えられてしまう（code-reviewer L1）
    with pytest.raises(ValueError, match="空"):
        read_faithfulness('{"answer_id": "F01-a", "added": [" "], "altered": []}\n')


def test_gate_boundaries_are_inclusive():
    # 7 問以上・9 問以上は、ちょうど 7・9 で満たす（code-reviewer M6）
    base = run(faithful=faithful())
    metrics = {c: dict(m) for c, m in base.metrics.items()}
    metrics["K1"]["content_correct"] = Count(tuple(f"Q{i}" for i in range(7)), 10)
    metrics["K1"]["trap_refused"] = Count(tuple(f"T{i}" for i in range(9)), 10)
    met = {name: ok for name, _, _, ok in gate(replace(base, metrics=metrics), "K1")}
    assert met["content_correct"] and met["trap_refused"]
    metrics["K1"]["content_correct"] = Count(tuple(f"Q{i}" for i in range(6)), 10)
    metrics["K1"]["trap_refused"] = Count(tuple(f"T{i}" for i in range(8)), 10)
    met = {name: ok for name, _, _, ok in gate(replace(base, metrics=metrics), "K1")}
    assert not met["content_correct"] and not met["trap_refused"]


def test_gate_rejects_an_unknown_comparison(monkeypatch):
    monkeypatch.setattr("scripts.lib.tally.GATE", (("content_correct", ">", 7),))
    with pytest.raises(ValueError, match=">"):
        gate(run(faithful=faithful()), "K1")


def test_a_limited_grader_rejects_answers_to_questions_outside_the_set():
    kinds = {k: v for k, v in KINDS.items() if k != "F05"}
    with pytest.raises(ValueError, match="F05"):
        tally("乙", entries(), labels("trap"), kinds, {}, read_title_categories(TITLES_YAML), CONDITIONS,
              scope=frozenset({"trap"}))


def test_the_faithfulness_judgment_may_not_name_unknown_answers():
    with pytest.raises(ValueError, match="Z99-a"):
        run(faithful={**faithful(), "Z99-a": ((), ())})
