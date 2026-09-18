"""コマンド（probe_voice・run_eval）が共有する引数の検査と表示。"""
import argparse
import re
from pathlib import Path

LABEL_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")  # ファイル名に埋め込むので区切り文字を許さない


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("1 以上の整数を指定してください")
    return number


def display(path: Path, root: Path) -> str:
    """表示用にリポジトリ相対へ。外のパスはそのまま返す（表示のために落とさない）。"""
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)
