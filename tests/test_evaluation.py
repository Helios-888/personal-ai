"""evaluation: 凍結した問いの読み込み、問う順番、記録（JSONL・Markdown）の組み立て。純粋関数。"""
import json
from dataclasses import fields, replace

import pytest

from scripts.lib.evaluation import (
    AnswerRecord,
    EvalQuestion,
    RunHeader,
    answer_json,
    count_truncated,
    end_json,
    header_json,
    inconsistent_prompt_tokens,
    load_eval_questions,
    plan_calls,
    render_eval_transcript,
    select_questions,
)
from scripts.lib.llm_client import ChatResult

FROZEN = """\
frozen: 2026-09-18
questions:
  - id: F01
    kind: factual
    text: 六大とは何ですか。
    expect: answer
    key_points:
      - point: 地・水・火・風・空・識
        source: SAT
  - id: H01
    kind: holdout
    episode: ep-kukai-0001
    text: |
      次の場面で、あなたならどうしますか。

      船は福州に流れ着いた。
    expect: direction
    key_points:
      - point: 自ら筆を執る
        source: 性霊集
"""

HEADER = RunHeader(
    condition="B1",
    route="openwebui",
    url="http://localhost:3000/api/chat/completions",
    model="kukai-ai",
    date="2026-09-19",
    questions_sha256="2e2c" + "0" * 60,
    system_prompt_sha256="ab" * 32,
    system_prompt_sent=False,
    system_tokens=2037,
    tokens_exact=True,
    temperature=0.7,
    max_tokens=800,
    repeats=3,
    question_count=2,
    agent="agents/kukai/agent.yaml",
    question_ids=("F01", "H01"),
    git_commit="c0ffee",
    git_dirty=(),
)


def result(content="答", *, finish="stop", reasoning="", prompt_tokens=2100):
    return ChatResult(
        content=content,
        reasoning=reasoning,
        finish_reason=finish,
        completion_tokens=5,
        elapsed_seconds=1.25,
        prompt_tokens=prompt_tokens,
    )


def test_load_eval_questions_reads_id_kind_and_text_in_file_order():
    assert load_eval_questions(FROZEN) == [
        EvalQuestion("F01", "factual", "六大とは何ですか。"),
        EvalQuestion("H01", "holdout", "次の場面で、あなたならどうしますか。\n\n船は福州に流れ着いた。"),
    ]


def test_eval_question_has_no_field_that_could_carry_the_answer():
    # 正解の要点（key_points）がモデルへ渡る経路を、型の段階で作らない
    assert {field.name for field in fields(EvalQuestion)} == {"id", "kind", "text"}


@pytest.mark.parametrize(
    "body, needle",
    [
        ("questions: []\n", "questions"),
        ("questions:\n  - id: F01\n    kind: factual\n", "1 番目"),
        ("questions:\n  - kind: factual\n    text: q\n", "1 番目"),
        ("questions:\n  - id: F01\n    text: q\n", "1 番目"),
        (
            "questions:\n  - id: F01\n    kind: factual\n    text: a\n  - id: F01\n    kind: factual\n    text: b\n",
            "F01",
        ),
    ],
    ids=["empty", "no-text", "no-id", "no-kind", "duplicate-id"],
)
def test_load_eval_questions_rejects_malformed_entries(body, needle):
    with pytest.raises(ValueError) as excinfo:
        load_eval_questions(body)

    assert needle in str(excinfo.value)


def test_select_questions_keeps_file_order_and_rejects_unknown_ids():
    questions = load_eval_questions(FROZEN)

    assert select_questions(questions, None) == questions
    assert select_questions(questions, []) == questions
    assert [q.id for q in select_questions(questions, ["H01", "F01"])] == ["F01", "H01"]
    with pytest.raises(ValueError) as excinfo:
        select_questions(questions, ["F01", "F99"])
    assert "F99" in str(excinfo.value)


def test_plan_calls_asks_every_question_once_per_round():
    # 途中で止まっても、各問いの 1 回目が先に揃うように巡回する
    questions = load_eval_questions(FROZEN)

    assert [(r, q.id) for r, q in plan_calls(questions, 3)] == [
        (1, "F01"),
        (1, "H01"),
        (2, "F01"),
        (2, "H01"),
        (3, "F01"),
        (3, "H01"),
    ]


def test_header_json_is_one_line_tagged_as_the_header():
    line = header_json(HEADER)

    assert "\n" not in line
    data = json.loads(line)
    assert data["record"] == "header"
    assert data["condition"] == "B1"
    assert data["questions_sha256"] == HEADER.questions_sha256
    assert data["system_prompt_sha256"] == HEADER.system_prompt_sha256
    assert data["system_prompt_sent"] is False
    assert data["question_ids"] == ["F01", "H01"]
    assert (data["git_commit"], data["git_dirty"]) == ("c0ffee", [])


def test_answer_json_keeps_the_answer_verbatim_with_its_question_and_round():
    question = EvalQuestion("H01", "holdout", "状況\n\n続き")

    line = answer_json("B1", AnswerRecord(2, question, result("一行目\n二行目\n")))

    assert "\n" not in line
    assert json.loads(line) == {
        "record": "answer",
        "condition": "B1",
        "question_id": "H01",
        "kind": "holdout",
        "repeat": 2,
        "content": "一行目\n二行目\n",
        "reasoning": "",
        "finish_reason": "stop",
        "prompt_tokens": 2100,
        "completion_tokens": 5,
        "elapsed_seconds": 1.25,
    }


def test_end_json_marks_whether_the_run_completed():
    # 途中で止まった記録を、機械が読む側でも完走した記録と区別できるように
    assert json.loads(end_json("complete", 6, ["F01"])) == {
        "record": "end",
        "status": "complete",
        "answers": 6,
        "inconsistent_prompt_tokens": ["F01"],
    }
    assert json.loads(end_json("interrupted", 0, []))["status"] == "interrupted"
    with pytest.raises(ValueError):
        end_json("done", 6, [])


def test_inconsistent_prompt_tokens_names_questions_whose_input_size_changed_between_rounds():
    # 同じ問いなら入力は毎回同じ大きさになる。違えば途中で定義か経路が変わった
    f01 = EvalQuestion("F01", "factual", "q1")
    f02 = EvalQuestion("F02", "factual", "q2")
    records = [
        AnswerRecord(1, f01, result(prompt_tokens=2100)),
        AnswerRecord(1, f02, result(prompt_tokens=2200)),
        AnswerRecord(2, f01, result(prompt_tokens=2100)),
        AnswerRecord(2, f02, result(prompt_tokens=4300)),
    ]

    assert inconsistent_prompt_tokens(records) == ["F02"]


def test_inconsistent_prompt_tokens_ignores_answers_without_usage():
    question = EvalQuestion("F01", "factual", "q")
    records = [AnswerRecord(1, question, result(prompt_tokens=2100)), AnswerRecord(2, question, result(prompt_tokens=0))]

    assert inconsistent_prompt_tokens(records) == []


def test_count_truncated_counts_answers_cut_by_the_output_limit():
    question = EvalQuestion("F01", "factual", "q")
    records = [AnswerRecord(1, question, result()), AnswerRecord(2, question, result(finish="length"))]

    assert count_truncated(records) == 1


def test_render_eval_transcript_groups_the_rounds_under_each_question():
    f01 = EvalQuestion("F01", "factual", "六大とは")
    h01 = EvalQuestion("H01", "holdout", "状況一\n\n状況二")
    records = [
        AnswerRecord(1, f01, result("答一")),
        AnswerRecord(1, h01, result("答二")),
        AnswerRecord(2, f01, result("答三", finish="length")),
    ]

    text = render_eval_transcript(HEADER, records)

    assert text.index("## F01 [factual]") < text.index("答一") < text.index("答三") < text.index("## H01 [holdout]")
    assert text.index("## H01 [holdout]") < text.index("答二")
    assert "> 状況一\n>\n> 状況二" in text
    assert "### 2 回目（1.2 秒、2 字、入力 2100 トークン、出力 5 トークン、finish=length）" in text


def test_render_eval_transcript_header_records_the_comparison_ground():
    text = render_eval_transcript(HEADER, [])

    for needle in (
        "# 評価の記録 B1（2026-09-19）",
        "openwebui（http://localhost:3000/api/chat/completions）",
        "kukai-ai",
        HEADER.questions_sha256,
        HEADER.system_prompt_sha256,
        "2037 トークン",
        "temperature 0.7",
        "max_tokens 800",
        "2 問 × 3 回",
        "agents/kukai/agent.yaml",
        "要求には含めない",
        "Git：c0ffee（定義とコードに未コミットの変更なし）",
    ):
        assert needle in text, needle


def test_render_eval_transcript_lists_uncommitted_changes():
    text = render_eval_transcript(replace(HEADER, git_dirty=(" M agents/kukai/system.md", "?? scripts/x.py")), [])

    assert "Git：c0ffee。未コミットの変更：M agents/kukai/system.md、?? scripts/x.py" in text


def test_render_eval_transcript_says_when_the_system_prompt_was_sent_and_counted_roughly():
    text = render_eval_transcript(replace(HEADER, system_prompt_sent=True, tokens_exact=False), [])

    assert "2037 トークン（概算）" in text
    assert "要求の system として送った" in text


def test_render_eval_transcript_shows_the_note_the_truncation_count_and_reasoning():
    question = EvalQuestion("F01", "factual", "q")
    records = [AnswerRecord(1, question, result(finish="length", reasoning="考"))]

    text = render_eval_transcript(replace(HEADER, note="途中で中断（1/6 回答まで）"), records)

    assert "途中で中断（1/6 回答まで）" in text
    assert "出力枠切れ（finish=length）1/1" in text
    assert "思考 1 字" in text


def test_render_eval_transcript_marks_an_empty_answer():
    question = EvalQuestion("F01", "factual", "q")

    text = render_eval_transcript(HEADER, [AnswerRecord(1, question, result("  "))])

    assert "（本文なし）" in text
