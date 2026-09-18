"""git_state: 記録の見出しに残す Git の状態（どの版の定義とコードで取ったか）。"""
import shutil
import subprocess
from pathlib import Path

import pytest

from scripts.lib.git_state import GitState, git_state


class FakeRunner:
    def __init__(self, outputs, returncode=0, stderr=""):
        self.outputs = list(outputs)
        self.returncode = returncode
        self.stderr = stderr
        self.calls = []

    def __call__(self, args, **kwargs):
        self.calls.append(args)
        return subprocess.CompletedProcess(args, self.returncode, stdout=self.outputs.pop(0), stderr=self.stderr)


def test_git_state_reads_head_and_the_uncommitted_changes_of_the_given_paths():
    runner = FakeRunner(["abc123\n", " M agents/kukai/system.md\n?? scripts/new.py\n"])

    state = git_state(Path("/repo"), ["agents/kukai", "scripts"], runner=runner)

    assert state == GitState(commit="abc123", dirty=(" M agents/kukai/system.md", "?? scripts/new.py"))
    assert runner.calls[0][-2:] == ["rev-parse", "HEAD"]
    assert runner.calls[1][-3:] == ["--", "agents/kukai", "scripts"]
    assert all(call[:3] == ["git", "-C", "/repo"] for call in runner.calls)


def test_git_state_reports_a_clean_tree_as_no_dirty_paths():
    state = git_state(Path("/repo"), ["scripts"], runner=FakeRunner(["abc123\n", ""]))

    assert state.dirty == ()


def test_git_state_fails_with_the_git_message_when_git_fails():
    runner = FakeRunner([""], returncode=128, stderr="fatal: not a git repository")

    with pytest.raises(ValueError) as excinfo:
        git_state(Path("/repo"), ["scripts"], runner=runner)

    assert "not a git repository" in str(excinfo.value)


@pytest.mark.skipif(shutil.which("git") is None, reason="git が無い環境")
def test_git_state_against_a_real_repository(tmp_path):
    def git(*args):
        subprocess.run(
            ["git", "-C", str(tmp_path), "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
            check=True,
            capture_output=True,
        )

    git("init", "-q")
    (tmp_path / "a.md").write_text("one\n", encoding="utf-8")
    (tmp_path / "b.md").write_text("one\n", encoding="utf-8")
    git("add", ".")
    git("commit", "-q", "-m", "init")

    clean = git_state(tmp_path, ["a.md"])
    (tmp_path / "a.md").write_text("two\n", encoding="utf-8")
    (tmp_path / "b.md").write_text("two\n", encoding="utf-8")

    assert len(clean.commit) == 40
    assert clean.dirty == ()
    assert git_state(tmp_path, ["a.md"]).dirty == (" M a.md",)  # 指定外の b.md は数えない
