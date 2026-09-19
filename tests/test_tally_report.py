"""tally_report: 集計結果の報告（tally.md）に、数えた回答へのたどり道と、先に決めた規則が載るかを検証する。"""
from scripts.lib.tally import read_title_categories, tally
from scripts.lib.tally_report import render_report
from tests.test_tally import KINDS, OWN, TITLES_YAML, entries, labels

HEADER = {"record": "key-header", "sources": {"B0": {"path": "base/b0"}, "B1": {"path": "base/b1"}}}
INPUTS = [("scoring/key.jsonl", "ab" * 32)]


def grader(name: str, changes: dict | None = None):
    return tally(name, entries(), {**labels(), **(changes or {})}, KINDS, OWN, read_title_categories(TITLES_YAML))


def render(*tallies) -> str:
    return render_report(HEADER, list(tallies), ["注記 1"], INPUTS, "c0ffee")


def test_intro_names_official_grader_sources_inputs_and_commit():
    report = render(grader("甲"), grader("乙"))
    assert "正式な値は 甲 の集計" in report
    assert "乙 は結論が変わらないかの確かめに使う" in report
    assert "`base/b0`" in report and "`base/b1`" in report
    assert f"| `scoring/key.jsonl` | `{'ab' * 32}` |" in report
    assert "`c0ffee`" in report and "- 注記 1" in report


def test_metrics_table_has_one_column_per_grader_and_condition():
    report = render(grader("甲"), grader("乙"))
    assert "| 指標 | B0（甲） | B1（甲） | B0（乙） | B1（乙） |" in report
    assert "| 架空引用（厳しい数え方） | 1/9 | 1/9 | 1/9 | 1/9 |" in report
    assert "| 過剰拒否（factual の回答単位） | 0/3 | 1/3 | 0/3 | 1/3 |" in report


def test_conclusions_show_margins_and_flag_disagreement():
    # 乙は B1 の attribution を 3 つとも誤にする → B1 の多数決が誤になり、1 問差が逆向き（どちらも明確な差なし）
    same = render(grader("甲"), grader("乙", {"A01-d": "誤", "A01-e": "誤"}))
    assert "| attribution 正（多数決） | 2 問 | 明確な差なし | 明確な差なし | 同じ |" in same
    assert "| 架空引用（厳しい数え方） | 1 回答 | 明確な差なし | 明確な差なし | 同じ |" in same
    # 乙は B0 の過剰拒否を 2 つ増やす → 分母 3 の 1 割（1 回答）以上の差で「明確」が食い違う
    differ = render(grader("甲"), grader("乙", {"F01-a": "過剰拒否", "F01-c": "過剰拒否"}))
    assert "**食い違い**" in differ


def test_ceiling_line_per_grader():
    report = render(grader("甲"))
    assert "- 甲：B1 の trap 正しく退けた 1/1 → 当たらない" in report
    assert "判定は正式な値（甲）で行う" in report


def test_agreement_section_lists_differences_with_conditions():
    report = render(grader("甲"), grader("乙", {"F01-a": "部分"}))
    assert "区分の一致：23/24（95.8%）" in report
    assert "| F01-a | B0 | 1 回目 | 正答 | 部分 |" in report
    assert "区分の一致：24/24（100.0%）" in render(grader("甲"), grader("乙"))


def test_per_question_table_marks_ties():
    report = render(grader("甲"))
    assert "| F01 | factual | 正答（a 正答・c 正答・e 部分） | 部分※（b 正答・d 誤答・f 過剰拒否） |" in report
    assert report.index("| F01 |") < report.index("| A01 |") < report.index("| T01 |") < report.index("| H01 |")


def test_majority_differences_listed_or_none():
    assert "（なし）" in render(grader("甲"), grader("乙")).split("食い違った問い")[1].split("##")[0]
    changed = render(grader("甲"), grader("乙", {"F01-e": "正答", "F01-a": "部分", "F01-c": "部分"}))
    assert "| F01 | B0 | 正答（a 正答・c 正答・e 部分） | 部分（a 部分・c 部分・e 正答） |" in changed


def test_fictitious_details_trace_to_record_lines():
    report = render(grader("甲"))
    assert "| T01-a | B0 | 1 回目 | 14 | 六大縁起章（見当たらない） |" in report
    assert "| F01-d | B1 | 2 回目 | 5 | 即身成仏論（書名の取り違え） |" in report


def test_fictitious_details_none():
    report = render(tally("甲", entries(), labels(), KINDS, {}, read_title_categories(TITLES_YAML)))
    assert "（なし）" in report.split("架空引用の内訳")[1]


def test_appendix_lists_hits_for_every_metric():
    report = render(grader("甲"))
    assert "- trap 正しく退けた（回答単位）・B1（2/3）：T01-b、T01-f" in report
    assert "- factual 正答（多数決）・B1（0/1）：なし" in report
    assert "transcript.md" in report
