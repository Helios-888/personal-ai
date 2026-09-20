"""tally CLI: 採点のフォルダから条件を戻して集計し、tally.md を書く流れを検証する。"""
import hashlib
import json
import subprocess
from pathlib import Path

import pytest

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


def run_cli(root: Path, runner=clean_git, *extra: str) -> int:
    return run(["--scoring", SCORING, *extra], root=root, runner=runner)


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


# --- Phase 4：B1 と K1、Astra は trap の束だけ、「」の書名は一次採点が抜き出す、付け足し・書き換えの判定 ---


P4_CONDITIONS = {"B0": "B1", "B1": "K1"}  # ROWS の条件を Phase 4 の条件に読み替える
P4_TITLES_YAML = """\
titles:
  - title: 即身成仏義
    forms: [即身成仏義]
    category: kukai_work
    sources: [x]
"""


def make_phase4_tree(root: Path, drop_faithfulness: str = "", faithfulness: bool = True, astra_all: bool = False,
                     baseline_first: bool = True, earlier_in_runs: bool = False) -> Path:
    frozen(root / "evaluations/kukai/questions.yaml", QUESTIONS_YAML)
    frozen(root / "evaluations/kukai/rubric.yaml", RUBRIC_YAML)
    sources = {}
    earlier = "evaluations/kukai/runs/b1" if earlier_in_runs else "evaluations/kukai/baseline/b1"
    records = [("B1", earlier), ("K1", "evaluations/kukai/runs/k1")]
    for condition, path in records if baseline_first else records[::-1]:
        write(root / path / "answers.jsonl", json.dumps({"record": "header", "condition": condition}) + "\n")
        sources[condition] = {"path": path, "answers_sha256": hashlib.sha256((root / path / "answers.jsonl").read_bytes()).hexdigest()}
    rows = [(a, P4_CONDITIONS[c], r, p, s) for a, c, r, p, s in ROWS]
    scoring = root / SCORING
    header = {"record": "key-header", "seed": 7, "settings": {"repeats": 3}, "sources": sources}
    keys = [{"record": "key", "answer_id": a, "question_id": a[:3], "letter": a[-1], "condition": c,
             "repeat": r, "line": 2 + i} for i, (a, c, r, _, _) in enumerate(rows)]
    write(scoring / "key.jsonl", jsonl([header, *keys]))
    titles = {"F01-b": [{"title": "即身成仏義", "where": "出典行", "as_own": True}]}
    write(scoring / "scores.jsonl", jsonl(
        [{"answer_id": a, "label": p, "reason": "r", "titles": titles.get(a, []), "quotes": []} for a, _, _, p, _ in rows]))
    write(scoring / "astra/result-trap.txt",
          jsonl([{"answer_id": a, "label": s, "reason": "r"} for a, _, _, _, s in rows if astra_all or a.startswith("T")]))
    write(scoring / "titles.yaml", P4_TITLES_YAML)
    if faithfulness:
        write(scoring / "faithfulness.jsonl", jsonl(
            [{"answer_id": a, "added": ["x"] if a == "F01-b" else [], "altered": []} for a, *_ in rows if a != drop_faithfulness]))
    return scoring


@pytest.fixture
def content_f01(monkeypatch):
    # テストの評価セットは F01 と T01 だけなので、内容 10 問の照合を F01 だけにする
    monkeypatch.setattr(tally_cli, "CONTENT_QUESTIONS", frozenset({"F01"}))


def test_phase4_layout_takes_conditions_from_the_key_and_adds_the_gate(tmp_path, capsys, content_f01):
    scoring = make_phase4_tree(tmp_path)
    assert run_cli(tmp_path) == 0
    report = (scoring / "tally.md").read_text(encoding="utf-8")
    assert "| 指標 | B1（Opus 5） | K1（Opus 5） | B1（GPT-6 Astra） | K1（GPT-6 Astra） |" in report
    assert "| factual 正答（多数決） | 1/1 | 0/1 | — | — |" in report
    assert "区分の一致：5/6（83.3%）。GPT-6 Astra の採点範囲：trap" in report
    assert "| 内容 10 問 付け足し・書き換えあり（回答単位） | 0/3 | 1/3 |" in report
    assert "## 正確さの関門" in report
    assert "`evaluations/kukai/runs/k1`" in report and f"`{SCORING}/faithfulness.jsonl`" in report
    assert "補助抜き出し" not in report.split("## 指標")[0]


def test_phase4_faithfulness_must_cover_every_answer(tmp_path, capsys, content_f01):
    scoring = make_phase4_tree(tmp_path, drop_faithfulness="T01-f")
    assert run_cli(tmp_path) == 1
    assert "T01-f" in capsys.readouterr().err
    assert not (scoring / "tally.md").exists()


def test_phase4_stops_without_the_faithfulness_judgment_unless_told(tmp_path, capsys, content_f01):
    # 黙って関門の無い tally.md を書かない（code-reviewer H1）。検索箇所を記録できなかったときだけ明示して通す
    scoring = make_phase4_tree(tmp_path, faithfulness=False)
    assert run_cli(tmp_path) == 1
    assert "faithfulness.jsonl" in capsys.readouterr().err
    assert not (scoring / "tally.md").exists()
    assert run(["--scoring", SCORING, "--no-faithfulness"], root=tmp_path, runner=clean_git) == 0
    report = (scoring / "tally.md").read_text(encoding="utf-8")
    assert "--no-faithfulness" in report and "## 正確さの関門" not in report


def test_second_grader_scope_must_match_the_declared_one(tmp_path, capsys, content_f01):
    # 結果ファイルの欠けを「範囲が狭い」と取り違えない（code-reviewer H2）。Phase 4 の既定は trap の束だけ
    scoring = make_phase4_tree(tmp_path, astra_all=True)
    assert run_cli(tmp_path) == 1
    assert "採点範囲" in capsys.readouterr().err
    assert run(["--scoring", SCORING, "--second-scope", "all"], root=tmp_path, runner=clean_git) == 0
    assert "区分の一致：10/12" in (scoring / "tally.md").read_text(encoding="utf-8")


def test_phase3_second_grader_missing_a_whole_kind_is_rejected(tmp_path, capsys):
    scoring = make_tree(tmp_path)
    write(scoring / "astra/result-1.txt", jsonl([{"answer_id": a, "label": s, "reason": "r"} for a, _, _, _, s in ROWS
                                                 if a.startswith("T")]))
    assert run_cli(tmp_path) == 1
    assert "採点範囲" in capsys.readouterr().err
    assert not (scoring / "tally.md").exists()


def test_phase4_needs_all_ten_content_questions(tmp_path, capsys):
    # 評価セットに内容 10 問がそろわないと、関門の「7 問以上」が小さい分母で判定される（code-reviewer M2）
    make_phase4_tree(tmp_path)
    assert run_cli(tmp_path) == 1
    assert "内容を問う 10 問" in capsys.readouterr().err


def test_the_baseline_record_must_come_first(tmp_path, capsys, content_f01):
    # 「下がらず」は後の条件で判定する。K1 が先だと B1 で判定してしまう（code-reviewer M3）
    make_phase4_tree(tmp_path, baseline_first=False)
    assert run_cli(tmp_path) == 1
    assert "基準値" in capsys.readouterr().err


def test_the_earlier_run_outside_the_baseline_is_declared(tmp_path, capsys, content_f01):
    # Phase 5：設定を変えた前後を比べるので、2 つの記録がどちらも runs/ にある。先に取ったほうを宣言して通す
    scoring = make_phase4_tree(tmp_path, earlier_in_runs=True)
    assert run_cli(tmp_path, clean_git, "--prior", "B1") == 0
    assert "--prior" in (scoring / "tally.md").read_text(encoding="utf-8")


def test_a_record_outside_the_baseline_stops_without_the_declaration(tmp_path, capsys, content_f01):
    make_phase4_tree(tmp_path, earlier_in_runs=True)
    assert run_cli(tmp_path) == 1
    assert "--prior B1" in capsys.readouterr().err  # 宣言の仕方を示す


def test_the_declaration_must_name_the_first_condition(tmp_path, capsys, content_f01):
    make_phase4_tree(tmp_path, earlier_in_runs=True)
    assert run_cli(tmp_path, clean_git, "--prior", "K1") == 1
    assert "1 つ目の条件" in capsys.readouterr().err


def test_the_declaration_is_refused_when_the_record_is_already_the_baseline(tmp_path, capsys, content_f01):
    make_phase4_tree(tmp_path)
    assert run_cli(tmp_path, clean_git, "--prior", "B1") == 1
    assert "すでに基準値" in capsys.readouterr().err


# --- 参考値のために除く回答（faithfulness/excluded.yaml） --------------------------------------

EXCLUDED_YAML = '''reason: 「語の説明」の読みが判定役のあいだで揃わなかった回答
answers: [F01-b]
'''


def test_phase4_reads_the_excluded_answers_and_adds_the_reference_counts(tmp_path, capsys, content_f01):
    scoring = make_phase4_tree(tmp_path)
    write(scoring / 'faithfulness/excluded.yaml', EXCLUDED_YAML)
    assert run_cli(tmp_path) == 0
    report = (scoring / 'tally.md').read_text(encoding='utf-8')
    assert '| 内容 10 問 付け足し・書き換えあり（回答単位） | 0/3 | 1/3 |' in report
    assert '| 内容 10 問 付け足し・書き換えあり（参考、語の説明を除く） | 0/3 | 0/3 |' in report
    assert 'F01-b を付け足し無しとして数え直す' in report
    assert '「語の説明」の読みが判定役のあいだで揃わなかった回答' in report.split('## 指標')[0]
    assert f'`{SCORING}/faithfulness/excluded.yaml`' in report


def test_an_unknown_excluded_answer_stops_the_tally(tmp_path, capsys, content_f01):
    scoring = make_phase4_tree(tmp_path)
    write(scoring / 'faithfulness/excluded.yaml', 'reason: r\nanswers: [Z99-a]\n')
    assert run_cli(tmp_path) == 1
    assert 'Z99-a' in capsys.readouterr().err
    assert not (scoring / 'tally.md').exists()


def test_the_excluded_file_needs_a_reason_and_answers(tmp_path, capsys, content_f01):
    scoring = make_phase4_tree(tmp_path)
    write(scoring / 'faithfulness/excluded.yaml', 'answers: [F01-b]\n')
    assert run_cli(tmp_path) == 1
    assert 'reason' in capsys.readouterr().err
    write(scoring / 'faithfulness/excluded.yaml', 'reason: r\nanswers: []\n')
    assert run_cli(tmp_path) == 1
    assert 'answers' in capsys.readouterr().err


def test_the_exclusion_may_not_be_combined_with_no_faithfulness(tmp_path, capsys, content_f01):
    scoring = make_phase4_tree(tmp_path, faithfulness=False)
    write(scoring / 'faithfulness/excluded.yaml', EXCLUDED_YAML)
    assert run(['--scoring', SCORING, '--no-faithfulness'], root=tmp_path, runner=clean_git) == 1
    assert 'excluded.yaml' in capsys.readouterr().err
    assert not (scoring / 'tally.md').exists()


# --- ホールドアウトの解放（holdout-released.yaml） ---------------------------------------

RELEASED_YAML = 'date: 2026-09-20\nreason: 利用者の判断で伏せを解いた（Phase 5 設計書「方針」1）\n'


def test_a_release_declaration_is_recorded_and_fingerprinted(tmp_path, capsys, content_f01):
    scoring = make_phase4_tree(tmp_path)
    write(scoring / 'holdout-released.yaml', RELEASED_YAML)
    assert run_cli(tmp_path) == 0
    report = (scoring / 'tally.md').read_text(encoding='utf-8')
    intro = report.split('## 指標')[0]
    assert '2026-09-20' in intro and '利用者の判断で伏せを解いた' in intro
    assert f'`{SCORING}/holdout-released.yaml`' in report


def test_the_release_declaration_needs_a_date_and_a_reason(tmp_path, capsys, content_f01):
    scoring = make_phase4_tree(tmp_path)
    write(scoring / 'holdout-released.yaml', 'reason: r\n')
    assert run_cli(tmp_path) == 1
    assert 'date' in capsys.readouterr().err
    write(scoring / 'holdout-released.yaml', 'date: 2026-09-20\n')
    assert run_cli(tmp_path) == 1
    assert 'reason' in capsys.readouterr().err
    assert not (scoring / 'tally.md').exists()
