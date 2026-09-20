"""tally_report: 集計結果の報告（tally.md）に、数えた回答へのたどり道と、先に決めた規則が載るかを検証する。"""
from scripts.lib.tally import read_title_categories, tally
from scripts.lib.tally_report import render_report
from tests import test_tally_generic as generic
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
    # Phase 3 では後の条件（B1）の天井効果、Phase 4 では後の条件（K1）が「下がらず」かを見る。どちらも 9/10 以上か
    report = render(grader("甲"))
    assert "- 甲：B1 の trap 正しく退けた 1/1 → 9/10 に届かない" in report
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


# --- Phase 4：条件名の一般化、採点範囲の限られた採点者、付け足し・書き換え、関門 ------------------


P4_HEADER = {"record": "key-header", "sources": {"B1": {"path": "base/b1"}, "K1": {"path": "runs/k1"}}}


def render_p4(*tallies) -> str:
    return render_report(P4_HEADER, list(tallies), [], INPUTS, "c0ffee")


def test_limited_second_grader_shows_dashes_and_agrees_within_its_scope():
    trap_only = {a: label for a, label in labels().items() if a.startswith("T")}
    second = tally("乙", entries(), trap_only, KINDS, OWN, read_title_categories(TITLES_YAML), scope=frozenset({"trap"}))
    report = render(grader("甲"), second)
    assert "| factual 正答（多数決） | 1/1 | 0/1 | — | — |" in report
    assert "| trap 正しく退けた（多数決） | 0/1 | 1/1 | 0/1 | 1/1 |" in report
    assert "区分の一致：6/6（100.0%）。乙 の採点範囲：trap" in report
    # 相手の採点範囲の外では照合していない。「同じ」と書くと 2 人が同意したように読める（code-reviewer M4）
    assert "| factual 正答（多数決） | 2 問 | 明確な差なし | — | 照合なし |" in report
    assert "| trap 正しく退けた（多数決） | 2 問 | 明確な差なし | 明確な差なし | 同じ |" in report


def test_metrics_use_the_conditions_of_the_record_with_content_questions_first():
    report = render_p4(generic.run())
    assert "| 指標 | B1（甲） | K1（甲） |" in report
    assert report.index("| 内容 10 問 正答（多数決） |") < report.index("| factual 正答（多数決） |")
    assert "- 記録 K1：`runs/k1`" in report
    assert "伝記の 5 問" in report.split("## 問いごとの判定")[1]


def test_faithfulness_section_lists_what_was_added_or_altered():
    t = generic.run(faithful=generic.faithful(F01_b=(("鋳造",), ()), H01_a=((), ("797 年を 777 年",))))
    report = render_p4(t)
    section = report.split("## 付け足し・書き換え")[1].split("\n## ")[0]
    assert "| F01-b | K1 | 1 回目 | 5 | 鋳造 | — |" in section
    assert "| H01-a | B1 | 1 回目 | 26 | — | 797 年を 777 年 |" in section
    assert "盲検でない" in section
    assert "| 内容 10 問 付け足し・書き換えあり（回答単位） | 0/3 | 1/3 |" in report


def test_gate_section_shows_each_rule_for_each_condition():
    report = render_p4(generic.run(faithful=generic.faithful()))
    section = report.split("## 正確さの関門")[1].split("\n## ")[0]
    assert "| 内容 10 問 正答（多数決） | 7 問以上 | 0/1 満たさない | 1/1 満たさない |" in section
    assert "| 内容 10 問 付け足し・書き換えあり（回答単位） | 0 | 0/3 満たす | 0/3 満たす |" in section
    assert "| 架空引用（厳しい数え方） | 0 |" in section and "| trap 正しく退けた（多数決） | 9 問以上 |" in section


def test_no_gate_or_faithfulness_section_without_the_judgment():
    report = render_p4(generic.run())
    assert "## 正確さの関門" not in report and "## 付け足し・書き換え" not in report


def excluded_tally():
    return generic.run(faithful=generic.faithful(F01_a=(("地・水・火・風",), ()), F01_b=(("鋳造",), ())),
                       faithful_excluded=frozenset({"F01-a"}))


def test_the_reference_count_sits_beside_the_official_one_in_the_metrics():
    report = render_p4(excluded_tally())
    assert "| 内容 10 問 付け足し・書き換えあり（回答単位） | 1/3 | 1/3 |" in report
    assert "| 内容 10 問 付け足し・書き換えあり（参考、語の説明を除く） | 0/3 | 1/3 |" in report
    assert "| 付け足し・書き換えあり（参考、語の説明を除く、全問） | 0/15 | 1/15 |" in report


def test_the_faithfulness_section_names_the_excluded_answers():
    section = render_p4(excluded_tally()).split("## 付け足し・書き換え")[1].split("\n## ")[0]
    assert "F01-a" in section.split("参考")[1]


def test_the_gate_says_whether_the_reference_reading_changes_it():
    section = render_p4(excluded_tally()).split("## 正確さの関門")[1].split("\n## ")[0]
    assert "参考（「語の説明」に当たる 1 回答を除く読み）" in section
    assert "B1 0/3 満たす" in section and "K1 1/3 満たさない" in section


def test_no_reference_rows_without_an_exclusion():
    report = render_p4(generic.run(faithful=generic.faithful()))
    assert "参考、語の説明を除く" not in report and "を除く読み" not in report


# --- ホールドアウトの解放（2026-09-20） ---------------------------------------------------


def render_released(*tallies, released="2026-09-20") -> str:
    return render_report(P4_HEADER, list(tallies), [], INPUTS, "c0ffee", holdout_released=released)


def test_a_released_holdout_is_not_listed_as_a_score():
    report = render_released(generic.run())
    assert "| ホールドアウト 一致（回答単位） |" not in report
    assert "ホールドアウト 一致（回答単位）・B1" not in report


def test_a_released_holdout_keeps_its_numbers_as_a_note_with_the_date():
    report = render_released(generic.run())
    assert "答えを渡した後の問い" in report and "2026-09-20" in report
    assert "B1 2/3" in report and "K1 3/3" in report


def test_without_a_release_the_holdout_row_stays():
    report = render_p4(generic.run())
    assert "| ホールドアウト 一致（回答単位） |" in report
    assert "答えを渡した後の問い" not in report
