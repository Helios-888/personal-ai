""".env の読み込み（外部ライブラリを増やさないための最小実装）。"""
from scripts.lib.env import load_dotenv


def test_load_dotenv_parses_key_values_and_ignores_comments_and_blank_lines(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("# comment\nA=1\nB = two words \n\nC=\n", encoding="utf-8")

    assert load_dotenv(env_file) == {"A": "1", "B": "two words", "C": ""}


def test_load_dotenv_returns_empty_dict_when_file_is_missing(tmp_path):
    assert load_dotenv(tmp_path / ".env") == {}


def test_load_dotenv_strips_matching_quotes(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text('KEY="sk-abc"\nOTHER=\'x\'\n', encoding="utf-8")

    assert load_dotenv(env_file) == {"KEY": "sk-abc", "OTHER": "x"}
