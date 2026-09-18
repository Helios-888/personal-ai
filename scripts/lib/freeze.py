"""凍結したファイル（評価セット）の指紋（SHA-256）の記録と照合。

記録は対象の隣に `<名前>.sha256` として、sha256sum と同じ書式（"<16 進 64 桁>  <ファイル名>"）で置く。
llm01 では `cd evaluations/kukai && sha256sum -c questions.sha256` でも同じ照合ができる。
照合はバイト列で行うため、改行コードの変換（LF → CRLF）も変更として検出する。

凍結したファイルを使う側（評価の実行など）は read_frozen_text で読む。照合と読み込みを同じバイト列で行い、
照合した後に読み直す間に中身が変わる隙を作らないため。
"""
import hashlib
import re
from pathlib import Path

RECORD_SUFFIX = ".sha256"
_RECORD_PATTERN = re.compile(r"([0-9a-f]{64})  (\S+)\n?")
_PROCEDURE = "凍結後の扱いは docs/specs/2026-09-18-phase3-design.md「凍結の仕組み」を参照"


class FrozenFileError(ValueError):
    """凍結したファイルが記録と食い違う、または記録が無い・読めない。"""


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record_path(path: Path) -> Path:
    return Path(path).with_suffix(RECORD_SUFFIX)


def write_record(path: Path) -> Path:
    """凍結の記録を書く。既にあれば FileExistsError（凍結の記録は黙って書き換えない）。"""
    path = Path(path)
    line = f"{file_sha256(path)}  {path.name}\n"  # 先に計算する。対象が無ければここで失敗し、空の記録を残さない
    record = record_path(path)
    with open(record, "x", encoding="utf-8", newline="\n") as file:
        file.write(line)
    return record


def read_record(record: Path) -> tuple[str, str]:
    """記録から（指紋, ファイル名）を読む。"""
    record = Path(record)
    if not record.is_file():
        raise FrozenFileError(f"凍結の記録がありません: {record.name}")
    try:
        text = record.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise FrozenFileError(f"凍結の記録の書式が不正です（UTF-8 で読めません）: {record.name}") from error
    match = _RECORD_PATTERN.fullmatch(text)
    if not match:
        raise FrozenFileError(f"凍結の記録の書式が不正です（sha256sum の 1 行の形式）: {record.name}")
    return match.group(1), match.group(2)


def verify_frozen(path: Path) -> str:
    """記録どおりなら指紋を返す。食い違えば FrozenFileError。"""
    path = Path(path)
    digest = file_sha256(path)
    _check_against_record(path, digest)
    return digest


def read_frozen_text(path: Path) -> str:
    """記録どおりであることを確かめた本文を返す。照合と読み込みは同じバイト列で行う。"""
    path = Path(path)
    data = path.read_bytes()
    _check_against_record(path, hashlib.sha256(data).hexdigest())
    return data.decode("utf-8")


def _check_against_record(path: Path, actual: str) -> None:
    expected, name = read_record(record_path(path))
    if name != path.name:
        raise FrozenFileError(
            f"{record_path(path).name} は別のファイル（{name}）の記録です。"
            f"記録は対象と同じフォルダで、ファイル名だけを書く形で作る（{_PROCEDURE}）"
        )
    if actual != expected:
        raise FrozenFileError(
            f"{path.name} は凍結後に変更されています（記録 {expected[:12]}…、現在 {actual[:12]}…）。{_PROCEDURE}"
        )
