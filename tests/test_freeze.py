"""freeze: 凍結したファイルの指紋（SHA-256）の記録と照合。

評価セットは Knowledge 投入前に凍結し、以後変えない（引き継ぎ書 §7）。
変わったことを人の注意ではなく機械で検出するための部品。
"""
import hashlib

import pytest

from scripts.lib.freeze import (
    FrozenFileError,
    file_sha256,
    read_frozen_text,
    read_record,
    record_path,
    verify_frozen,
    write_record,
)

CONTENT = "frozen: 2026-09-18\nquestions:\n  - id: F01\n    text: 六大とは何ですか。\n"
CHANGED = "変更されています"


@pytest.fixture
def frozen_file(tmp_path):
    path = tmp_path / "questions.yaml"
    path.write_bytes(CONTENT.encode("utf-8"))
    write_record(path)
    return path


def test_record_path_sits_next_to_the_file(tmp_path):
    assert record_path(tmp_path / "questions.yaml") == tmp_path / "questions.sha256"


def test_record_uses_the_sha256sum_format(frozen_file):
    expected = hashlib.sha256(CONTENT.encode("utf-8")).hexdigest()
    assert record_path(frozen_file).read_text(encoding="utf-8") == f"{expected}  questions.yaml\n"


def test_unchanged_file_passes_and_returns_its_digest(frozen_file):
    assert verify_frozen(frozen_file) == file_sha256(frozen_file)


def test_one_character_change_is_detected(frozen_file):
    frozen_file.write_bytes(CONTENT.replace("六大", "四曼").encode("utf-8"))
    with pytest.raises(FrozenFileError, match=f"questions.yaml は凍結後に{CHANGED}"):
        verify_frozen(frozen_file)


def test_line_ending_conversion_is_detected(frozen_file):
    # Windows で CRLF に変換されただけでも「変わった」と扱う（バイト列で照合するため）
    frozen_file.write_bytes(CONTENT.replace("\n", "\r\n").encode("utf-8"))
    with pytest.raises(FrozenFileError, match=CHANGED):
        verify_frozen(frozen_file)


def test_record_is_written_only_once(frozen_file):
    with pytest.raises(FileExistsError):
        write_record(frozen_file)


def test_recording_a_missing_file_leaves_no_empty_record(tmp_path):
    # 空の記録が残ると、以後は「書式が不正」としか出ず本当の原因（パスの誤り）が見えなくなる
    path = tmp_path / "questions.yaml"
    with pytest.raises(FileNotFoundError):
        write_record(path)
    assert not record_path(path).exists()


def test_missing_record_is_an_error(tmp_path):
    path = tmp_path / "questions.yaml"
    path.write_text(CONTENT, encoding="utf-8")
    with pytest.raises(FrozenFileError, match="記録がありません"):
        verify_frozen(path)


@pytest.mark.parametrize(
    "record",
    [
        "abc  questions.yaml\n",  # 桁数が足りない
        f"{'0' * 64} questions.yaml\n",  # 区切りの空白が 1 つ
        f"{'0' * 64}  questions.yaml\n{'0' * 64}  other.yaml\n",  # 2 行ある
        "",
    ],
)
def test_malformed_record_is_an_error(tmp_path, record):
    path = tmp_path / "questions.yaml"
    path.write_text(CONTENT, encoding="utf-8")
    record_path(path).write_text(record, encoding="utf-8")
    with pytest.raises(FrozenFileError, match="書式"):
        verify_frozen(path)


def test_record_in_another_encoding_is_reported_as_malformed(tmp_path):
    # Windows PowerShell 5.1 の `>` でリダイレクトすると UTF-16 で書かれる
    path = tmp_path / "questions.yaml"
    path.write_text(CONTENT, encoding="utf-8")
    record_path(path).write_bytes(f"{file_sha256(path)}  questions.yaml\n".encode("utf-16"))
    with pytest.raises(FrozenFileError, match="UTF-8"):
        verify_frozen(path)


def test_record_for_another_file_is_an_error(tmp_path):
    path = tmp_path / "questions.yaml"
    path.write_text(CONTENT, encoding="utf-8")
    record_path(path).write_text(f"{file_sha256(path)}  evaluations/kukai/questions.yaml\n", encoding="utf-8")
    with pytest.raises(FrozenFileError, match="別のファイル.*同じフォルダ"):
        verify_frozen(path)


def test_read_record_returns_digest_and_name(frozen_file):
    digest, name = read_record(record_path(frozen_file))
    assert (digest, name) == (file_sha256(frozen_file), "questions.yaml")


def test_read_frozen_text_returns_the_verified_content(frozen_file):
    assert read_frozen_text(frozen_file) == CONTENT


def test_read_frozen_text_refuses_a_changed_file(frozen_file):
    # 照合と読み込みを同じバイト列で行う。照合だけ通して別の中身を使う、という隙を作らない
    frozen_file.write_bytes(CONTENT.replace("六大", "四曼").encode("utf-8"))
    with pytest.raises(FrozenFileError, match=CHANGED):
        read_frozen_text(frozen_file)
