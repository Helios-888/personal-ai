"""prompt_builder: parts を順に連結してシステムプロンプトを作る純粋関数のテスト。"""
import pytest

from scripts.lib.prompt_builder import build_system_prompt, is_over_budget, load_parts


def test_build_joins_parts_in_given_order_separated_by_blank_line():
    assert build_system_prompt(["A", "B", "C"]) == "A\n\nB\n\nC\n"


def test_build_strips_trailing_whitespace_of_each_part_and_ends_with_one_newline():
    assert build_system_prompt(["A\n\n\n", "  B  \n"]) == "A\n\n  B\n"


def test_build_with_no_parts_returns_empty_string():
    assert build_system_prompt([]) == ""


def test_load_parts_reads_files_relative_to_root_in_the_given_order(tmp_path):
    (tmp_path / "a.md").write_text("one", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.md").write_text("two", encoding="utf-8")

    assert load_parts(tmp_path, ["sub/b.md", "a.md"]) == ["two", "one"]


def test_load_parts_names_the_missing_file_in_the_error(tmp_path):
    with pytest.raises(FileNotFoundError) as excinfo:
        load_parts(tmp_path, ["missing.md"])
    assert "missing.md" in str(excinfo.value)


def test_is_over_budget_is_true_only_when_count_exceeds_budget():
    assert is_over_budget(5001, 5000) is True
    assert is_over_budget(5000, 5000) is False
