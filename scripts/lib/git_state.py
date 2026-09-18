"""記録の見出しに残す Git の状態（どの版の定義とコードで取ったか）。

指紋（SHA-256）は中身が同じことを示すが、どの版から来たかは示さない。コミットを残せば履歴から辿れる。
"""
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class GitState:
    commit: str
    dirty: tuple[str, ...]  # 未コミットの変更（git status --porcelain の行）。空なら記録どおりに再現できる


def git_state(root: Path, paths: list[str], runner: Callable = subprocess.run) -> GitState:
    """HEAD のコミットと、paths のうち未コミットの変更（追跡外のファイルを含む）を返す。"""
    commit = _git(runner, root, "rev-parse", "HEAD").strip()
    status = _git(runner, root, "-c", "core.quotepath=false", "status", "--porcelain", "--", *paths)
    return GitState(commit=commit, dirty=tuple(line for line in status.splitlines() if line.strip()))


def _git(runner: Callable, root: Path, *args: str) -> str:
    completed = runner(["git", "-C", str(root), *args], capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise ValueError(f"git {' '.join(args)} に失敗しました: {completed.stderr.strip()}")
    return completed.stdout
