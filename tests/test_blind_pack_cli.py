"""blind_pack CLI: 基準値の 2 記録から、条件名を伏せた採点用の束と対応表を作る流れを検証する。"""
import json
from pathlib import Path

import pytest

import scripts.blind_pack as blind_pack
from scripts.blind_pack import check_question_ids, run
from scripts.lib.freeze import file_sha256, write_record

NL = "\n"
QUESTIONS_YAML = """\
frozen: 2026-09-18
questions:
  - id: F01
    kind: factual
    text: 問一
    expect: answer
    key_points:
      - point: 正解一
        source: 出典一
  - id: T01
    kind: trap
    text: 問二
    expect: refuse
    key_points:
      - point: 正解二
        source: 出典二
"""
RUBRIC_YAML = """\
frozen: 2026-09-18
revisions: []
labels:
  factual:
    正答: 要点をすべて満たす
  trap:
    正しく退けた: 前提を否定する
majority_tie: 省略
extract:
  - 出典行に挙げた書名
fictitious_citation: 省略
questions:
  F01:
    wrong_if:
      - 数が合わない
  T01:
    wrong_if:
      - 書の実在を認める
"""
B0_DIR = "evaluations/kukai/baseline/2026-09-18-B0-x"
B1_DIR = "evaluations/kukai/baseline/2026-09-19-B1-x"
OUT_DIR = "evaluations/kukai/scoring/test"


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline=NL) as file:
        file.write(text)
    return path


def answers_jsonl(condition, questions_sha256, content=lambda q, r: f"{q}の答え{r}", max_tokens=2000):
    header = {
        "record": "header",
        "condition": condition,
        "route": "openwebui",
        "temperature": 0.7,
        "max_tokens": max_tokens,
        "repeats": 2,
        "questions_sha256": questions_sha256,
    }
    answers = [
        json.dumps(
            {"record": "answer", "condition": condition, "question_id": q, "repeat": r, "content": content(q, r)},
            ensure_ascii=False,
        )
        for r in (1, 2)
        for q in ("F01", "T01")
    ]
    end = json.dumps({"record": "end", "status": "complete", "answers": 4, "inconsistent_prompt_tokens": []})
    return NL.join([json.dumps(header), *answers, end]) + NL


def rewrite_rubric(root: Path, text: str) -> None:
    rubric = write(root / "evaluations/kukai/rubric.yaml", text)
    rubric.with_suffix(".sha256").unlink()
    write_record(rubric)


@pytest.fixture
def root(tmp_path):
    questions = write(tmp_path / "evaluations/kukai/questions.yaml", QUESTIONS_YAML)
    write_record(questions)
    write_record(write(tmp_path / "evaluations/kukai/rubric.yaml", RUBRIC_YAML))
    digest = file_sha256(questions)
    write(tmp_path / B0_DIR / "answers.jsonl", answers_jsonl("B0", digest))
    write(tmp_path / B1_DIR / "answers.jsonl", answers_jsonl("B1", digest))
    return tmp_path


def invoke(root, *extra, seed=42):
    return run(["--b0", B0_DIR, "--b1", B1_DIR, "--out", OUT_DIR, *extra], root=root, seed_source=lambda: seed)


def pack_texts(root, out=OUT_DIR):
    return {path.name: path.read_text(encoding="utf-8") for path in sorted((root / out / "pack").iterdir())}


def read_key(root):
    return [json.loads(line) for line in (root / OUT_DIR / "key.jsonl").read_text(encoding="utf-8").splitlines()]


def test_writes_guide_one_sheet_per_question_and_the_key(root):
    assert invoke(root) == 0
    assert list(pack_texts(root)) == ["00-guide.md", "F01.md", "T01.md"]
    header, body = read_key(root)[0], read_key(root)[1:]
    assert header["seed"] == 42
    assert header["settings"] == {
        "questions_sha256": file_sha256(root / "evaluations/kukai/questions.yaml"),
        "route": "openwebui",
        "temperature": 0.7,
        "max_tokens": 2000,
        "repeats": 2,
    }
    assert header["python"]
    assert header["rubric_sha256"] == file_sha256(root / "evaluations/kukai/rubric.yaml")
    assert header["sources"]["B0"]["path"] == B0_DIR
    assert header["sources"]["B1"]["answers_sha256"] == file_sha256(root / B1_DIR / "answers.jsonl")
    assert len(body) == 8
    assert {(row["condition"], row["question_id"], row["repeat"]) for row in body} == {
        (c, q, r) for c in ("B0", "B1") for q in ("F01", "T01") for r in (1, 2)
    }


def test_key_leads_back_to_the_record_line_and_the_sheet(root):
    assert invoke(root) == 0
    key = read_key(root)
    sources = key[0]["sources"]
    sheets = pack_texts(root)
    for row in key[1:]:
        record = (root / sources[row["condition"]]["path"] / "answers.jsonl").read_text(encoding="utf-8")
        original = json.loads(record.split(NL)[row["line"] - 1])
        assert (original["condition"], original["question_id"], original["repeat"]) == (
            row["condition"],
            row["question_id"],
            row["repeat"],
        )
        section = sheets[f"{row['question_id']}.md"].split(f"### {row['answer_id']}{NL}")[1].split(f"{NL}### ")[0]
        assert f"~~~~~~{NL}{original['content']}{NL}~~~~~~" in section


def test_pack_does_not_reveal_conditions_or_records(root):
    assert invoke(root) == 0
    for name, text in pack_texts(root).items():
        for hidden in ("B0", "B1", "baseline", "2026-09-18-B0", "key.jsonl"):
            assert hidden not in text, f"{name} に {hidden} が含まれる"


def test_files_are_written_with_lf(root):
    assert invoke(root) == 0
    for path in [*(root / OUT_DIR / "pack").iterdir(), root / OUT_DIR / "key.jsonl"]:
        assert b"\r" not in path.read_bytes()


def test_same_seed_gives_same_pack(root):
    assert invoke(root) == 0
    assert run(["--b0", B0_DIR, "--b1", B1_DIR, "--out", OUT_DIR + "-2"], root=root, seed_source=lambda: 42) == 0
    assert pack_texts(root) == pack_texts(root, OUT_DIR + "-2")


def test_refuses_existing_output(root, capsys):
    (root / OUT_DIR).mkdir(parents=True)
    assert invoke(root) == 1
    assert "既に" in capsys.readouterr().err
    assert list((root / OUT_DIR).iterdir()) == []


def test_failure_while_writing_removes_the_half_made_output(root, monkeypatch, capsys):
    calls = []

    def failing_write(path, text):
        calls.append(path)
        if len(calls) == 2:
            raise OSError("容量不足")
        with open(path, "x", encoding="utf-8", newline=NL) as file:
            file.write(text)

    monkeypatch.setattr(blind_pack, "write_new", failing_write)
    assert invoke(root) == 1
    err = capsys.readouterr().err
    assert "作りかけ" in err and "容量不足" in err
    assert not (root / OUT_DIR).exists()


def test_refuses_record_of_another_question_set(root, capsys):
    write(root / B1_DIR / "answers.jsonl", answers_jsonl("B1", "0" * 64))
    assert invoke(root) == 1
    assert "questions" in capsys.readouterr().err
    assert not (root / OUT_DIR).exists()


def test_refuses_records_taken_with_different_settings(root, capsys):
    digest = file_sha256(root / "evaluations/kukai/questions.yaml")
    write(root / B1_DIR / "answers.jsonl", answers_jsonl("B1", digest, max_tokens=800))
    assert invoke(root) == 1
    assert "max_tokens" in capsys.readouterr().err
    assert not (root / OUT_DIR).exists()


def test_refuses_swapped_records(root, capsys):
    result = run(["--b0", B1_DIR, "--b1", B0_DIR, "--out", OUT_DIR], root=root, seed_source=lambda: 1)
    assert result == 1
    assert "B0" in capsys.readouterr().err
    assert not (root / OUT_DIR).exists()


def test_broken_record_names_the_file_and_line(root, capsys):
    path = root / B0_DIR / "answers.jsonl"
    lines = path.read_text(encoding="utf-8").split(NL)
    write(path, NL.join([lines[0], "{壊れた", *lines[1:]]))
    assert invoke(root) == 1
    assert f"{B0_DIR}/answers.jsonl: 2 行目" in capsys.readouterr().err


def test_refuses_changed_frozen_file(root, capsys):
    write(root / "evaluations/kukai/rubric.yaml", RUBRIC_YAML + "# 追記" + NL)
    assert invoke(root) == 1
    assert "rubric.yaml" in capsys.readouterr().err
    assert not (root / OUT_DIR).exists()


def test_refuses_kind_without_labels(root, capsys):
    rewrite_rubric(root, RUBRIC_YAML.replace("  trap:\n    正しく退けた: 前提を否定する\n", ""))
    assert invoke(root) == 1
    assert "trap" in capsys.readouterr().err
    assert not (root / OUT_DIR).exists()


def test_refuses_rubric_without_extract(root, capsys):
    rewrite_rubric(root, RUBRIC_YAML.replace("extract:\n  - 出典行に挙げた書名\n", ""))
    assert invoke(root) == 1
    assert "extract" in capsys.readouterr().err


def test_malformed_rubric_is_reported_not_raised(root, capsys):
    rewrite_rubric(root, "labels: [" + NL)
    assert invoke(root) == 1
    assert "error:" in capsys.readouterr().err


def test_refuses_answer_that_names_a_condition(root, capsys):
    digest = file_sha256(root / "evaluations/kukai/questions.yaml")
    write(root / B1_DIR / "answers.jsonl", answers_jsonl("B1", digest, content=lambda q, r: "B1 と申す"))
    assert invoke(root) == 1
    assert "伏せ" in capsys.readouterr().err
    assert not (root / OUT_DIR).exists()


def test_question_ids_must_be_safe_file_names():
    check_question_ids(["F01", "H02"])
    with pytest.raises(ValueError, match="../x"):
        check_question_ids(["F01", "../x"])
