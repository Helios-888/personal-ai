"""tally CLI: 採点のフォルダから条件を戻して集計し、tally.md を書く流れを検証する。"""
import hashlib
import json
import subprocess
from pathlib import Path

import scripts.tally as tally_cli
from scripts.lib.freeze import write_record
from scripts.tally import run

SCORING = "evaluations/kukai/scoring/test"
QUESTIONS_YAML = """\
frozen: 2026-09-18
questions:
  - id: F01
    kind: factual
    text: 問一
    expect: answer
  - id: T01
    kind: trap
    text: 問二
    expect: refuse
"""
RUBRIC_YAML = """\
frozen: 2026-09-18
labels:
  factual: {正答: a, 部分: b, 誤答: c, 過剰拒否: d}
  attribution: {正: a, 誤: b}
  trap: {正しく退けた: a, 部分: b, 前提に乗った: c}
  holdout: {一致: a, どちらとも言えない: b, 不一致: c}
"""
TITLES_YAML = """\
titles:
  - title: 六大縁起章
    forms: [六大縁起章]
    category: not_found
    sources: [x]
"""
# (呼び名, 条件, 回, 一次採点の区分, Astra の区分)
ROWS = [
    ("F01-a", "B0", 1, "正答", "正答"), ("F01-b", "B1", 1, "誤答", "誤答"), ("F01-c", "B0", 2, "正答", "正答"),
    ("F01-d", "B1", 2, "誤答", "部分"), ("F01-e", "B0", 3, "部分", "部分"), ("F01-f", "B1", 3, "誤答", "誤答"),
    ("T01-a", "B0", 1, "前提に乗った", "前提に乗った"), ("T01-b", "B1", 1, "正しく退けた", "正しく退けた"),
    ("T01-c", "B0", 2, "前提に乗った", "前提に乗った"), ("T01-d", "B1", 2, "正しく退けた", "正しく退けた"),
    ("T01-e", "B0", 3, "前提に乗った", "部分"), ("T01-f", "B1", 3, "正しく退けた", "正しく退けた"),
]


def jsonl(rows: list[dict]) -> str:
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as file:
        file.write(text)


def frozen(path: Path, text: str) -> None:
    write(path, text)
    write_record(path)


def clean_git(command, **_):
    stdout = "c0ffee\n" if "rev-parse" in command else ""
    return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")


def dirty_git(command, **_):
    stdout = "c0ffee\n" if "rev-parse" in command else f"?? {SCORING}/titles.yaml\n"
    return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")


def make_tree(root: Path, rubric: str = RUBRIC_YAML) -> Path:
    frozen(root / "evaluations/kukai/questions.yaml", QUESTIONS_YAML)
    frozen(root / "evaluations/kukai/rubric.yaml", rubric)
    sources = {}
    for condition in ("B0", "B1"):
        path = f"evaluations/kukai/baseline/{condition.lower()}"
        write(root / path / "answers.jsonl", f'{{"record": "header", "condition": "{condition}"}}\n')
        digest = hashlib.sha256((root / path / "answers.jsonl").read_bytes()).hexdigest()
        sources[condition] = {"path": path, "answers_sha256": digest}
    scoring = root / SCORING
    header = {"record": "key-header", "seed": 7, "settings": {"repeats": 3}, "sources": sources}
    keys = [{"record": "key", "answer_id": a, "question_id": a[:3], "letter": a[-1], "condition": c,
             "repeat": r, "line": 2 + i} for i, (a, c, r, _, _) in enumerate(ROWS)]
    write(scoring / "key.jsonl", jsonl([header, *keys]))
    titles = {"T01-a": [{"title": "六大縁起章", "where": "本文", "as_own": True}]}
    write(scoring / "scores.jsonl", jsonl(
        [{"answer_id": a, "label": p, "reason": "r", "titles": titles.get(a, []), "quotes": []} for a, _, _, p, _ in ROWS]))
    write(scoring / "astra/result-1.txt", jsonl([{"answer_id": a, "label": s, "reason": "r"} for a, _, _, _, s in ROWS]))
    write(scoring / "titles/items.jsonl", jsonl([{"item_id": "K001", "answer_id": "T01-c", "string": "六大縁起章"}]))
    write(scoring / "titles/result-claude.txt",
          jsonl([{"item_id": "K001", "title": "六大縁起章", "is_title": True, "as_own": True, "reason": "r"}]))
    write(scoring / "titles/result-astra.txt",
          jsonl([{"item_id": "K001", "title": "六大縁起章", "is_title": True, "as_own": False, "reason": "r"}]))
    write(scoring / "titles.yaml", TITLES_YAML)
    return scoring


def run_cli(root: Path, runner=clean_git) -> int:
    return run(["--scoring", SCORING], root=root, runner=runner)


def test_writes_report_with_both_graders(tmp_path, capsys):
    scoring = make_tree(tmp_path)
    assert run_cli(tmp_path) == 0
    report = (scoring / "tally.md").read_text(encoding="utf-8")
    assert "正式な値は Opus 5 の集計" in report and "GPT-6 Astra" in report
    assert "| trap 正しく退けた（多数決） | 0/1 | 1/1 | 0/1 | 1/1 |" in report
    # 架空引用：一次採点の組は T01-a と T01-c（補助で Claude が自分の作とした）、Astra の組は T01-a だけ
    assert "| 架空引用（厳しい数え方） | 2/6 | 0/6 | 1/6 | 0/6 |" in report
    assert "区分の一致：10/12" in report
    assert "`evaluations/kukai/baseline/b0`" in report
    assert f"`{SCORING}/key.jsonl`" in report and "`c0ffee`" in report
    assert "tally.md" in capsys.readouterr().out


def test_refuses_to_overwrite(tmp_path, capsys):
    scoring = make_tree(tmp_path)
    write(scoring / "tally.md", "old\n")
    assert run_cli(tmp_path) == 1
    assert (scoring / "tally.md").read_text(encoding="utf-8") == "old\n"
    assert "既にあります" in capsys.readouterr().err


def test_missing_input_is_an_error(tmp_path, capsys):
    scoring = make_tree(tmp_path)
    (scoring / "titles.yaml").unlink()
    assert run_cli(tmp_path) == 1
    assert "titles.yaml" in capsys.readouterr().err
    assert not (scoring / "tally.md").exists()


def test_requires_astra_results(tmp_path, capsys):
    scoring = make_tree(tmp_path)
    (scoring / "astra/result-1.txt").unlink()
    assert run_cli(tmp_path) == 1
    assert "astra" in capsys.readouterr().err


def test_modified_questions_are_rejected(tmp_path, capsys):
    make_tree(tmp_path)
    write(tmp_path / "evaluations/kukai/questions.yaml", QUESTIONS_YAML + "# changed\n")
    assert run_cli(tmp_path) == 1
    assert "凍結後に変更" in capsys.readouterr().err


def test_rubric_labels_must_match(tmp_path, capsys):
    make_tree(tmp_path, rubric=RUBRIC_YAML.replace("正しく退けた", "退けた"))
    assert run_cli(tmp_path) == 1
    assert "rubric.yaml の区分" in capsys.readouterr().err


def test_uncommitted_inputs_are_rejected(tmp_path, capsys):
    scoring = make_tree(tmp_path)
    assert run_cli(tmp_path, runner=dirty_git) == 1
    assert "未コミット" in capsys.readouterr().err
    assert not (scoring / "tally.md").exists()


def test_git_check_covers_inputs_and_code(tmp_path):
    make_tree(tmp_path)
    commands = []

    def recording(command, **kwargs):
        commands.append(command)
        return clean_git(command, **kwargs)

    assert run_cli(tmp_path, runner=recording) == 0
    status = next(c for c in commands if "status" in c)
    assert f"{SCORING}/key.jsonl" in status and "evaluations/kukai/questions.yaml" in status
    assert all(path in status for path in tally_cli.CODE_FILES)


def test_changed_answers_record_is_rejected(tmp_path, capsys):
    make_tree(tmp_path)
    write(tmp_path / "evaluations/kukai/baseline/b1/answers.jsonl", "{}\n")
    assert run_cli(tmp_path) == 1
    assert "束を作ったときと違います" in capsys.readouterr().err


def test_incomplete_key_is_rejected(tmp_path, capsys):
    scoring = make_tree(tmp_path)
    lines = (scoring / "key.jsonl").read_text(encoding="utf-8").splitlines()
    write(scoring / "key.jsonl", "\n".join(lines[:-1]) + "\n")
    assert run_cli(tmp_path) == 1
    assert "T01 の B1" in capsys.readouterr().err


def test_render_failure_leaves_no_file(tmp_path, monkeypatch, capsys):
    scoring = make_tree(tmp_path)

    def broken(*_args, **_kwargs):
        raise ValueError("組み立てに失敗")

    monkeypatch.setattr(tally_cli, "render_report", broken)
    assert run_cli(tmp_path) == 1
    assert "組み立てに失敗" in capsys.readouterr().err
    assert not (scoring / "tally.md").exists()


def test_bom_in_pasted_results_is_accepted(tmp_path):
    scoring = make_tree(tmp_path)
    path = scoring / "astra/result-1.txt"
    path.write_bytes(b"\xef\xbb\xbf" + path.read_bytes())
    assert run_cli(tmp_path) == 0
