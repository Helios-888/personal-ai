"""blinding: 基準値の回答を、条件名を伏せた採点用の束に組み替える部品。"""
import json

import pytest

from scripts.lib.blinding import (
    blind_groups,
    key_lines,
    load_scoring_questions,
    read_run,
    render_guide,
    render_sheet,
    strip_citations,
    strip_run_citations,
)

QUESTIONS_YAML = """\
frozen: 2026-09-18
questions:
  - id: F01
    kind: factual
    text: 問一
    expect: answer
    key_points:
      - point: 正解一
        source: 出典一
  - id: T01
    kind: trap
    text: 問二
    expect: refuse
    key_points:
      - point: 正解二
        source: 出典二
"""
RUBRIC_YAML = """\
frozen: 2026-09-18
revisions: []
labels:
  factual:
    正答: 要点をすべて満たす
    誤答: 要点を満たさない
  trap:
    正しく退けた: 前提を否定する
    前提に乗った: 前提に沿って語る
majority_tie: 省略
extract:
  - 出典行に挙げた書名
fictitious_citation: 省略
questions:
  F01:
    wrong_if:
      - 数が合わない
    grading: 二点で正答
  T01:
    wrong_if:
      - 書の実在を認める
"""


def run_text(condition, answers, status="complete", repeats=2, count=None):
    lines = [json.dumps({"record": "header", "condition": condition, "repeats": repeats, "questions_sha256": "q"})]
    for question_id, repeat, content in answers:
        lines.append(
            json.dumps(
                {"record": "answer", "condition": condition, "question_id": question_id, "repeat": repeat, "content": content},
                ensure_ascii=False,
            )
        )
    total = len(answers) if count is None else count
    lines.append(json.dumps({"record": "end", "status": status, "answers": total, "inconsistent_prompt_tokens": []}))
    return "\n".join(lines) + "\n"


def full_run(condition, prefix):
    return read_run(
        run_text(condition, [(q, r, f"{prefix}{q}{r}") for r in (1, 2) for q in ("F01", "T01")])
    )


@pytest.fixture
def questions():
    return load_scoring_questions(QUESTIONS_YAML, RUBRIC_YAML)


@pytest.fixture
def runs():
    return {"B0": full_run("B0", "零"), "B1": full_run("B1", "壱")}


class TestReadRun:
    def test_reads_header_and_answers_with_line_numbers(self):
        run = read_run(run_text("B0", [("F01", 1, "答え")]))
        assert run.header["condition"] == "B0"
        assert [(a.condition, a.question_id, a.repeat, a.content, a.line) for a in run.answers] == [
            ("B0", "F01", 1, "答え", 2)
        ]

    def test_rejects_interrupted_run(self):
        with pytest.raises(ValueError, match="完走"):
            read_run(run_text("B0", [("F01", 1, "答え")], status="interrupted"))

    def test_rejects_missing_end_marker(self):
        text = run_text("B0", [("F01", 1, "答え")])
        without_end = "\n".join(text.splitlines()[:-1]) + "\n"
        with pytest.raises(ValueError, match="終わりの印"):
            read_run(without_end)

    def test_rejects_end_count_that_disagrees(self):
        with pytest.raises(ValueError, match="件数"):
            read_run(run_text("B0", [("F01", 1, "答え")], count=2))

    def test_rejects_missing_header(self):
        text = run_text("B0", [("F01", 1, "答え")])
        without_header = "\n".join(text.splitlines()[1:]) + "\n"
        with pytest.raises(ValueError, match="見出し"):
            read_run(without_header)

    def test_rejects_answer_of_another_condition(self):
        text = run_text("B0", [("F01", 1, "答え")]).replace('"condition": "B0", "question_id"', '"condition": "B1", "question_id"')
        with pytest.raises(ValueError, match="2 行目"):
            read_run(text)

    @pytest.mark.parametrize("repeats", [None, 0, "3", True])
    def test_rejects_bad_repeats_in_header(self, repeats):
        text = run_text("B0", [("F01", 1, "答え")]).replace('"repeats": 2', f'"repeats": {json.dumps(repeats)}')
        with pytest.raises(ValueError, match="repeats"):
            read_run(text)

    def test_broken_json_names_the_line(self):
        lines = run_text("B0", [("F01", 1, "答え")]).splitlines()
        broken = "\n".join([lines[0], "{壊れた", *lines[1:]]) + "\n"
        with pytest.raises(ValueError, match="2 行目が JSON として読めません"):
            read_run(broken)

    def test_rejects_line_that_is_not_an_object(self):
        lines = run_text("B0", [("F01", 1, "答え")]).splitlines()
        with pytest.raises(ValueError, match="2 行目が JSON のオブジェクトではありません"):
            read_run("\n".join([lines[0], "[1, 2]", *lines[1:]]) + "\n")

    def test_answer_with_missing_field_names_the_line(self):
        text = run_text("B0", [("F01", 1, "答え")]).replace('"repeat": 1, ', "")
        with pytest.raises(ValueError, match="2 行目の回答に欠けた欄"):
            read_run(text)

    def test_line_separator_inside_content_does_not_split_the_line(self):
        run = read_run(run_text("B0", [("F01", 1, "前 後"), ("T01", 1, "次")]))
        assert [(a.content, a.line) for a in run.answers] == [("前 後", 2), ("次", 3)]


class TestLoadScoringQuestions:
    def test_joins_key_points_with_rubric(self, questions):
        first = questions[0]
        assert (first.id, first.kind, first.text) == ("F01", "factual", "問一")
        assert first.key_points == (("正解一", "出典一"),)
        assert first.wrong_if == ("数が合わない",)
        assert first.grading == "二点で正答"
        assert questions[1].grading == ""

    def test_rejects_question_missing_from_rubric(self):
        rubric = RUBRIC_YAML.replace("  T01:\n    wrong_if:\n      - 書の実在を認める\n", "")
        with pytest.raises(ValueError, match="T01"):
            load_scoring_questions(QUESTIONS_YAML, rubric)


class TestBlindGroups:
    def test_every_answer_appears_once_under_a_letter(self, questions, runs):
        groups = blind_groups(questions, runs, seed=7)
        assert [g.question.id for g in groups] == ["F01", "T01"]
        for group in groups:
            assert [a.letter for a in group.answers] == ["a", "b", "c", "d"]
            assert [a.answer_id for a in group.answers] == [f"{group.question.id}-{x}" for x in "abcd"]
        sources = sorted((a.source.condition, a.source.question_id, a.source.repeat) for g in groups for a in g.answers)
        expected = sorted((c, q, r) for c in ("B0", "B1") for q in ("F01", "T01") for r in (1, 2))
        assert sources == expected

    def test_content_is_kept_verbatim(self, questions, runs):
        groups = blind_groups(questions, runs, seed=7)
        assert all(a.content == a.source.content for g in groups for a in g.answers)

    def test_same_seed_gives_same_order(self, questions, runs):
        first = blind_groups(questions, runs, seed=7)
        second = blind_groups(questions, runs, seed=7)
        assert [[a.source for a in g.answers] for g in first] == [[a.source for a in g.answers] for g in second]

    def test_order_is_shuffled_within_the_question(self, questions, runs):
        orders = {
            tuple(a.source.condition + str(a.source.repeat) for a in blind_groups(questions, runs, seed=s)[0].answers)
            for s in range(20)
        }
        assert len(orders) > 1

    def test_letter_does_not_depend_on_condition(self, questions, runs):
        positions = {
            index
            for seed in range(50)
            for index, answer in enumerate(blind_groups(questions, runs, seed=seed)[0].answers)
            if answer.source.condition == "B0"
        }
        assert positions == {0, 1, 2, 3}

    def test_rejects_duplicated_repeat(self, questions, runs):
        doubled = read_run(run_text("B1", [("F01", 1, "x"), ("F01", 1, "y"), ("T01", 1, "x"), ("T01", 2, "x")]))
        with pytest.raises(ValueError, match="B1 の F01 の回が 1〜2"):
            blind_groups(questions, {"B0": runs["B0"], "B1": doubled}, seed=1)

    def test_rejects_more_answers_than_letters(self, questions):
        many = {
            c: read_run(run_text(c, [(q, r, "x") for r in range(1, 15) for q in ("F01", "T01")], repeats=14))
            for c in ("B0", "B1")
        }
        with pytest.raises(ValueError, match="a〜z"):
            blind_groups(questions, many, seed=1)

    def test_rejects_different_repeats(self, questions, runs):
        other = read_run(run_text("B1", [(q, 1, "x") for q in ("F01", "T01")], repeats=1))
        with pytest.raises(ValueError, match="反復"):
            blind_groups(questions, {"B0": runs["B0"], "B1": other}, seed=1)

    def test_rejects_missing_answer(self, questions, runs):
        short = read_run(run_text("B1", [("F01", 1, "x"), ("F01", 2, "x"), ("T01", 1, "x")]))
        with pytest.raises(ValueError, match="T01"):
            blind_groups(questions, {"B0": runs["B0"], "B1": short}, seed=1)

    def test_rejects_answer_to_unknown_question(self, questions, runs):
        extra = read_run(
            run_text("B1", [(q, r, "x") for r in (1, 2) for q in ("F01", "T01")] + [("Z99", 1, "x")])
        )
        with pytest.raises(ValueError, match="Z99"):
            blind_groups(questions, {"B0": runs["B0"], "B1": extra}, seed=1)


class TestRenderSheet:
    def test_sheet_has_question_rubric_and_every_answer(self, questions, runs):
        labels = {"正答": "要点をすべて満たす", "誤答": "要点を満たさない"}
        group = blind_groups(questions, runs, seed=7)[0]
        sheet = render_sheet(group, labels)
        for expected in (
            "F01", "問一", "正解一", "出典一", "数が合わない", "二点で正答", "正答", "要点をすべて満たす",
            "（key_points）", "（wrong_if）", "（grading）",
        ):
            assert expected in sheet
        for answer in group.answers:
            assert f"### {answer.answer_id}" in sheet
            assert answer.content in sheet

    def test_sheet_does_not_reveal_the_condition(self, questions, runs):
        group = blind_groups(questions, runs, seed=7)[0]
        sheet = render_sheet(group, {"正答": "x"})
        for hidden in ("B0", "B1", "condition", "repeat", "条件"):
            assert hidden not in sheet

    def test_content_keeps_its_surrounding_newlines(self, questions):
        run0 = read_run(run_text("B0", [(q, r, "\n前後に改行\n") for r in (1, 2) for q in ("F01", "T01")]))
        group = blind_groups(questions, {"B0": run0, "B1": full_run("B1", "壱")}, seed=1)[0]
        assert "~~~~~~\n\n前後に改行\n\n~~~~~~" in render_sheet(group, {"正答": "x"})

    def test_rejects_answer_that_would_break_the_fence(self, questions):
        run0 = read_run(run_text("B0", [(q, r, "~~~~~~ 壊す") for r in (1, 2) for q in ("F01", "T01")]))
        group = blind_groups(questions, {"B0": run0, "B1": full_run("B1", "壱")}, seed=1)[0]
        with pytest.raises(ValueError, match="囲み"):
            render_sheet(group, {"正答": "x"})

    def test_sheet_shows_the_rubric_note(self, runs):
        # Phase 3 では note を載せておらず、採点者は rubric の備考を見られなかった
        with_note = RUBRIC_YAML.replace("    grading: 二点で正答\n", "    grading: 二点で正答\n    note: 触れても加点にも減点にもしない\n")
        group = blind_groups(load_scoring_questions(QUESTIONS_YAML, with_note), runs, seed=7)[0]
        sheet = render_sheet(group, {"正答": "x"})
        assert "## この問いの備考（note）\n\n触れても加点にも減点にもしない\n" in sheet
        assert sheet.index("（grading）") < sheet.index("（note）") < sheet.index("## 回答")

    def test_sheet_has_no_note_heading_without_a_note(self, questions, runs):
        for group in blind_groups(questions, runs, seed=7):
            assert "（note）" not in render_sheet(group, {"正答": "x"})


class TestRenderGuide:
    def test_guide_lists_labels_extract_and_output_format(self):
        guide = render_guide({"factual": {"正答": "要点をすべて満たす"}}, ["出典行に挙げた書名"])
        for expected in ("factual", "正答", "要点をすべて満たす", "出典行に挙げた書名", "scores.jsonl", "answer_id", "as_own"):
            assert expected in guide

    def test_guide_says_over_refusal_wins_over_wrong(self):
        guide = render_guide({"factual": {"誤答": "x", "過剰拒否": "y"}, "trap": {"部分": "z"}}, ["w"])
        assert "factual で実在の事柄を退けた回答は、誤答とするものに当たる場合も「過剰拒否」とする" in guide
        assert "trap で実在の事柄を退けた" not in guide

    def test_guide_maps_rubric_field_names_to_sheet_headings(self):
        guide = render_guide({"factual": {"正答": "x"}}, ["y"])
        assert "key_points・wrong_if・grading は、採点票の「正解の要点」「誤答とするもの」「採点の決まり」" in guide

    def test_guide_does_not_reveal_the_comparison(self):
        guide = render_guide({"factual": {"正答": "x"}}, ["y"])
        for hidden in ("B0", "B1", "条件", "比較", "基準値"):
            assert hidden not in guide

    def test_guide_extracts_titles_given_in_kagikakko_too(self):
        # Phase 3 では『』だけを抜き出し、「」で挙げた自著名（架空のものを含む）を補助の抜き出しで拾い直した
        guide = render_guide({"factual": {"正答": "x"}}, ["本文中に『 』で挙げた書名"])
        assert "本文中に「 」で挙げた書名も、書名として挙げているなら『 』と同じく抜き出す" in guide
        assert guide.index("## 抜き出し") < guide.index("「 」で挙げた書名も") < guide.index("## 出力")

    def test_output_example_does_not_suggest_a_real_title(self):
        guide = render_guide({"factual": {"正答": "x"}}, ["y"])
        assert "即身成仏義" not in guide
        assert '"title": "（書名）"' in guide


class TestKeyLines:
    def test_key_maps_every_blind_id_back_to_its_record(self, questions, runs):
        groups = blind_groups(questions, runs, seed=7)
        lines = [json.loads(line) for line in key_lines(groups, seed=7, meta={"questions_sha256": "q"})]
        assert lines[0] == {"record": "key-header", "seed": 7, "questions_sha256": "q"}
        body = lines[1:]
        assert len(body) == 8
        first = groups[0].answers[0]
        assert body[0] == {
            "record": "key",
            "answer_id": first.answer_id,
            "question_id": "F01",
            "letter": "a",
            "condition": first.source.condition,
            "repeat": first.source.repeat,
            "line": first.source.line,
            "citations_removed": 0,
        }

    def test_key_records_how_many_citations_were_removed_from_each_answer(self, questions):
        # 束の本文は元の記録と違ってくる。どの回答から幾つ除いたかを対応表に残す
        k1 = strip_run_citations(read_run(run_text("K1", [
            (q, r, f"答え{q}{r} [1]。続き[2]" if (q, r) == ("F01", 2) else f"答え{q}{r}")
            for r in (1, 2) for q in ("F01", "T01")
        ])))
        groups = blind_groups(questions, {"B1": full_run("B1", "壱"), "K1": k1}, seed=7)
        body = [json.loads(line) for line in key_lines(groups, seed=7)[1:]]
        removed = {(row["condition"], row["question_id"], row["repeat"]): row["citations_removed"] for row in body}
        assert removed.pop(("K1", "F01", 2)) == 2
        assert set(removed.values()) == {0}


class TestStripCitations:
    @pytest.mark.parametrize(
        ("text", "expected", "count"),
        [
            ("四つである [1]。", "四つである。", 1),  # RAG テンプレートは番号の前に半角空白を置く。残すと「 。」が印になる
            ("と記した。[2]\n次の段", "と記した。\n次の段", 1),
            ("（即身成仏義）[1]。", "（即身成仏義）。", 1),
            ("六大[1]と四曼 [2]、三密[3]", "六大と四曼、三密", 3),
            ("全角の空白　[1]。", "全角の空白。", 1),
            ("印の無い回答。", "印の無い回答。", 0),
            ("[1] によれば", "によれば", 1),  # 行頭では後ろの空白を除く（前の空白を除くだけでは行頭に空白が残る）
            ("一行目。\n[2]　次の行", "一行目。\n次の行", 1),
        ],
        ids=["space-before", "after-period", "after-paren", "several", "ideographic-space", "none",
             "line-start", "line-start-after-break"],
    )
    def test_removes_citation_numbers_with_the_space_before_them(self, text, expected, count):
        assert strip_citations(text) == (expected, count)

    # 番号の形が違うもの（[1, 2]・全角数字・別の括弧）は除かない。残れば blind_pack の見張りが止める
    @pytest.mark.parametrize(
        "text", ["[注] 本文", "〔1〕の説", "［1］の説", "[一]の説", "[1a]", "a[]b", "[1, 2]", "[１]", "【1】"],
    )
    def test_leaves_other_brackets_alone(self, text):
        assert strip_citations(text) == (text, 0)

    def test_keeps_the_line_breaks_around_a_citation(self):
        assert strip_citations("一行目 [1]\n\n二行目") == ("一行目\n\n二行目", 1)

    def test_run_keeps_line_numbers_and_leaves_the_original_untouched(self):
        original = read_run(run_text("K1", [("F01", 1, "答え [1]。"), ("F01", 2, "答え。")]))
        stripped = strip_run_citations(original)
        assert [(a.content, a.line, a.citations_removed) for a in stripped.answers] == [
            ("答え。", 2, 1),
            ("答え。", 3, 0),
        ]
        assert stripped.header == original.header
        assert [a.content for a in original.answers] == ["答え [1]。", "答え。"]
