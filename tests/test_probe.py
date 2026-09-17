"""probe: 質問ファイルの読み込みと、確認記録（Markdown）の組み立て。純粋関数。"""
from pathlib import Path

import pytest

from scripts.lib.llm_client import ChatResult
from scripts.lib.probe import ProbeRecord, Question, TranscriptHeader, load_questions, render_transcript


def write_questions(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


def test_load_questions_reads_kind_and_text_in_file_order(tmp_path):
    path = write_questions(
        tmp_path / "q.yaml",
        "questions:\n  - kind: 語調\n    text: 弟子入りを願う若者\n  - kind: 判断\n    text: 最澄の借覧\n",
    )

    assert load_questions(path) == [Question("語調", "弟子入りを願う若者"), Question("判断", "最澄の借覧")]


def test_load_questions_names_the_entry_that_lacks_text(tmp_path):
    path = write_questions(tmp_path / "q.yaml", "questions:\n  - kind: 語調\n    text: ok\n  - kind: 判断\n")

    with pytest.raises(ValueError) as excinfo:
        load_questions(path)

    assert "2" in str(excinfo.value)
    assert "text" in str(excinfo.value)


def test_load_questions_rejects_a_file_without_any_question(tmp_path):
    path = write_questions(tmp_path / "q.yaml", "questions: []\n")

    with pytest.raises(ValueError):
        load_questions(path)


HEADER = TranscriptHeader(
    title="語り口 下書き 2 の確認",
    date="2026-09-18",
    conditions="RAG なし、思考 OFF",
    temperature=0.7,
    max_tokens=800,
    system_tokens=2103,
    tokens_exact=True,
)


def result(content, *, reasoning="", finish="stop", tokens=12, seconds=6.23):
    return ChatResult(
        content=content,
        reasoning=reasoning,
        finish_reason=finish,
        completion_tokens=tokens,
        elapsed_seconds=seconds,
    )


def test_render_transcript_puts_conditions_in_the_title_line_and_token_count_below():
    md = render_transcript(HEADER, [])

    assert md.startswith("# 語り口 下書き 2 の確認（2026-09-18、RAG なし、思考 OFF、temperature 0.7、max_tokens 800）\n")
    assert "\nシステムプロンプト 2103 トークン。\n" in md


def test_render_transcript_marks_the_token_count_when_it_is_only_an_estimate():
    header = TranscriptHeader(**{**HEADER.__dict__, "tokens_exact": False})

    assert "システムプロンプト 2103 トークン（概算）。" in render_transcript(header, [])


def test_render_transcript_writes_one_section_per_question_with_timing_and_answer():
    records = [ProbeRecord(Question("語調", "弟子入りを願う若者"), result("若者よ、よく来た。\n"))]

    md = render_transcript(HEADER, records)

    assert "## [語調] 弟子入りを願う若者\n\n（6.2 秒、9 字、12 トークン、finish=stop）\n\n若者よ、よく来た。\n" in md


def test_render_transcript_shows_reasoning_length_and_a_marker_when_the_answer_is_empty():
    records = [ProbeRecord(Question("判断", "最澄"), result("", reasoning="x" * 3606, finish="length", tokens=1200))]

    md = render_transcript(HEADER, records)

    assert "（6.2 秒、0 字、1200 トークン、思考 3606 字、finish=length）" in md
    assert "（本文なし）" in md


def test_render_transcript_separates_sections_with_a_blank_line_and_ends_with_a_newline():
    records = [
        ProbeRecord(Question("a", "q1"), result("A")),
        ProbeRecord(Question("b", "q2"), result("B")),
    ]

    md = render_transcript(HEADER, records)

    assert "\nA\n\n## [b] q2\n" in md
    assert md.endswith("B\n")


# --- レビュー指摘への回帰テスト（2026-09-18） ---


def test_load_questions_collapses_line_breaks_inside_a_question_so_the_heading_stays_one_line(tmp_path):
    path = write_questions(tmp_path / "q.yaml", "questions:\n  - kind: 語調\n    text: |\n      一行目\n      二行目\n")

    assert load_questions(path) == [Question("語調", "一行目 二行目")]


def test_load_questions_rejects_a_text_that_is_not_a_string(tmp_path):
    path = write_questions(tmp_path / "q.yaml", "questions:\n  - kind: 語調\n    text: {a: 1}\n")

    with pytest.raises(ValueError) as excinfo:
        load_questions(path)

    assert "text" in str(excinfo.value)


def test_load_questions_gives_a_readable_default_kind(tmp_path):
    path = write_questions(tmp_path / "q.yaml", "questions:\n  - text: 問い\n")

    assert load_questions(path) == [Question("その他", "問い")]


def test_render_transcript_names_model_url_and_agent_definition_below_the_token_line():
    header = TranscriptHeader(
        **{**HEADER.__dict__, "model": "kukai", "base_url": "http://localhost:8080", "agent": "agents/kukai/agent.yaml"}
    )

    md = render_transcript(header, [])

    assert "システムプロンプト 2103 トークン。\nモデル kukai（http://localhost:8080）、定義 agents/kukai/agent.yaml。\n" in md


def test_render_transcript_shows_the_interruption_note_when_given():
    header = TranscriptHeader(**{**HEADER.__dict__, "note": "途中で中断（1/4 問まで）"})

    md = render_transcript(header, [ProbeRecord(Question("a", "q1"), result("A"))])

    assert "\n途中で中断（1/4 問まで）\n" in md
    assert md.index("途中で中断") < md.index("## [a] q1")
