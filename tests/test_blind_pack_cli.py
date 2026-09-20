"""blind_pack CLI: 条件ごとの記録から、条件名を伏せた採点用の束と対応表を作る流れを検証する。"""
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
K1_DIR = "evaluations/kukai/runs/2026-09-20-K1"
OUT_DIR = "evaluations/kukai/scoring/test"


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline=NL) as file:
        file.write(text)
    return path


def answers_jsonl(condition, questions_sha256, content=lambda q, r: f"{q}の答え{r}", max_tokens=2000, git_dirty=()):
    header = {
        "record": "header",
        "condition": condition,
        "route": "openwebui",
        "temperature": 0.7,
        "max_tokens": max_tokens,
        "repeats": 2,
        "questions_sha256": questions_sha256,
    }
    if git_dirty is not None:
        header["git_dirty"] = list(git_dirty)
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
    return run(["--run", f"B0={B0_DIR}", "--run", f"B1={B1_DIR}", "--out", OUT_DIR, *extra], root=root,
               seed_source=lambda: seed)


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
    argv = ["--run", f"B0={B0_DIR}", "--run", f"B1={B1_DIR}", "--out", OUT_DIR + "-2"]
    assert run(argv, root=root, seed_source=lambda: 42) == 0
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
    result = run(["--run", f"B0={B1_DIR}", "--run", f"B1={B0_DIR}", "--out", OUT_DIR], root=root, seed_source=lambda: 1)
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


# --- Phase 4：B1 と K1 の組など、条件は --run で渡す ---


def test_packs_any_pair_of_conditions(root):
    # Phase 4 は B1 の基準値に K1 を混ぜる（設計書「採点」）。記録の置き場は runs/
    write(root / K1_DIR / "answers.jsonl", answers_jsonl("K1", file_sha256(root / "evaluations/kukai/questions.yaml")))
    argv = ["--run", f"B1={B1_DIR}", "--run", f"K1={K1_DIR}", "--out", OUT_DIR]
    assert run(argv, root=root, seed_source=lambda: 3) == 0
    key = read_key(root)
    assert set(key[0]["sources"]) == {"B1", "K1"}
    assert {row["condition"] for row in key[1:]} == {"B1", "K1"}
    for name, text in pack_texts(root).items():
        for hidden in ("B1", "K1", "runs", "2026-09-20-K1"):
            assert hidden not in text, f"{name} に {hidden} が含まれる"


def test_needs_at_least_two_runs(root, capsys):
    assert run(["--run", f"B1={B1_DIR}", "--out", OUT_DIR], root=root, seed_source=lambda: 1) == 1
    assert "2 つ以上" in capsys.readouterr().err
    assert not (root / OUT_DIR).exists()


def test_refuses_the_same_condition_twice(root, capsys):
    argv = ["--run", f"B1={B1_DIR}", "--run", f"B1={B0_DIR}", "--out", OUT_DIR]
    assert run(argv, root=root, seed_source=lambda: 1) == 1
    assert "B1" in capsys.readouterr().err
    assert not (root / OUT_DIR).exists()


@pytest.mark.parametrize("bad", [B1_DIR, "=" + B1_DIR, "B1=", "../x=" + B1_DIR])
def test_refuses_a_malformed_run_argument(root, capsys, bad):
    assert run(["--run", bad, "--run", f"B0={B0_DIR}", "--out", OUT_DIR], root=root, seed_source=lambda: 1) == 1
    assert "--run" in capsys.readouterr().err
    assert not (root / OUT_DIR).exists()


def k1_record(root, content=lambda q, r: f"{q}の答え{r}", folder=K1_DIR):
    write(root / folder / "answers.jsonl",
          answers_jsonl("K1", file_sha256(root / "evaluations/kukai/questions.yaml"), content=content))
    return ["--run", f"B1={B1_DIR}", "--run", f"K1={folder}", "--out", OUT_DIR]


@pytest.mark.parametrize("git_dirty", [[" M agents/kukai/rules.md"], None], ids=["dirty", "missing"])
def test_refuses_a_record_taken_with_uncommitted_definitions(root, capsys, git_dirty):
    # run_eval の見張りをすり抜けた記録（runs/ の外で取ったものなど）も、採点の束には入れない
    digest = file_sha256(root / "evaluations/kukai/questions.yaml")
    write(root / B1_DIR / "answers.jsonl", answers_jsonl("B1", digest, git_dirty=git_dirty))
    assert invoke(root) == 1
    err = capsys.readouterr().err
    assert "未コミット" in err and "B1" in err
    assert not (root / OUT_DIR).exists()


@pytest.mark.parametrize(
    "marker",
    ["T2428_.77.0382c22 に曰く", '<source id="1">', "primary__即身成仏義.md による", "資料はprimary__即身成仏義",
     "[1] と T2428_.77.0382c22",
     "四つである [1, 2]。", "［1］", "【1】", "[１]", "[[1]2]", "である [1] 。"],
    ids=["sat-line-id", "source-tag", "file-prefix", "file-prefix-after-kana", "citation-and-line-id",
         "citation-list", "fullwidth-brackets", "lenticular-brackets", "fullwidth-digit", "nested", "space-left-before-period"],
)
def test_refuses_answers_that_carry_retrieval_markers(root, capsys, marker):
    # 検索の印は資料ありの条件にしか出ないので、残すと採点者に条件が分かる。引用の番号 [n] だけは除いて進む（下）。
    # 形の違う番号や、除いた跡の空白（「 。」）が残れば止まる（code-reviewer M1・L1、2026-09-19）
    argv = k1_record(root, content=lambda q, r: f"{q}の答え{r}。{marker}")
    assert run(argv, root=root, seed_source=lambda: 1) == 1
    err = capsys.readouterr().err
    assert "K1" in err and "印" in err
    assert not (root / OUT_DIR).exists()


def test_strips_citation_numbers_from_every_answer_and_keeps_the_record(root):
    # 利用者の判断（2026-09-19）：RAG テンプレートの引用の番号 [n] は、束を作るときに全回答から一様に除く。
    # K1 本番では 96 回答中 43 に計 93 個、うち 67 個は前に半角空白。元の記録（answers.jsonl）は変えない
    argv = k1_record(root, content=lambda q, r: f"{q}の答え{r} [1]。" if r == 1 else f"{q}の答え{r}。")
    before = file_sha256(root / K1_DIR / "answers.jsonl")
    assert run(argv, root=root, seed_source=lambda: 1) == 0
    texts = pack_texts(root)
    assert all("[1]" not in text and " 。" not in text for text in texts.values())
    assert f"~~~~~~{NL}F01の答え1。{NL}~~~~~~" in texts["F01.md"]
    key = read_key(root)
    assert key[0]["citations"]["removed"] == {"B1": 0, "K1": 2}
    assert key[0]["citations"]["pattern"]
    assert {(row["condition"], row["repeat"], row["citations_removed"]) for row in key[1:]} == {
        ("B1", 1, 0), ("B1", 2, 0), ("K1", 1, 1), ("K1", 2, 0),
    }
    assert file_sha256(root / K1_DIR / "answers.jsonl") == before == key[0]["sources"]["K1"]["answers_sha256"]


def test_hides_the_parent_folder_of_each_record(root, capsys):
    # K1 の記録は runs/ の下にある。その名が束に出れば条件が分かる
    argv = k1_record(root, content=lambda q, r: f"{q}の答え{r} runs")
    assert run(argv, root=root, seed_source=lambda: 1) == 1
    assert "runs" in capsys.readouterr().err
    assert not (root / OUT_DIR).exists()


def test_a_record_folder_at_the_repo_top_is_accepted(root):
    # 親フォルダの名が空になる。空の語を伏せる語に入れると、どの束も「漏れ」と判定される
    argv = k1_record(root, folder="2026-09-20-K1-top")
    assert run(argv, root=root, seed_source=lambda: 1) == 0


# --- Phase 5：同じ条件名で取った 2 つの記録を、束では別の名で呼ぶ（--recorded） ---

K2_DIR = "evaluations/kukai/runs/2026-09-20-K1-phase5"


def two_k1_records(root):
    """Phase 5 の本番は run_eval の条件 K1 で取ってある（K2 という条件名は run_eval に無い）。"""
    digest = file_sha256(root / "evaluations/kukai/questions.yaml")
    write(root / K1_DIR / "answers.jsonl", answers_jsonl("K1", digest))
    write(root / K2_DIR / "answers.jsonl", answers_jsonl("K1", digest))
    return ["--run", f"K1={K1_DIR}", "--run", f"K2={K2_DIR}", "--out", OUT_DIR]


def test_packs_two_records_of_the_same_recorded_condition(root):
    argv = [*two_k1_records(root), "--recorded", "K2=K1"]
    assert run(argv, root=root, seed_source=lambda: 5) == 0
    key = read_key(root)
    assert key[0]["sources"]["K2"] == {
        "path": K2_DIR,
        "recorded_condition": "K1",
        "answers_sha256": file_sha256(root / K2_DIR / "answers.jsonl"),
    }
    assert key[0]["sources"]["K1"] == {
        "path": K1_DIR,
        "answers_sha256": file_sha256(root / K1_DIR / "answers.jsonl"),
    }
    assert {row["condition"] for row in key[1:]} == {"K1", "K2"}
    assert len(key[1:]) == 8


def test_refuses_an_undeclared_rename(root, capsys):
    assert run(two_k1_records(root), root=root, seed_source=lambda: 1) == 1
    assert "--recorded K2=K1" in capsys.readouterr().err
    assert not (root / OUT_DIR).exists()


def test_refuses_a_declaration_that_does_not_match_the_record(root, capsys):
    argv = [*two_k1_records(root), "--recorded", "K2=B0"]
    assert run(argv, root=root, seed_source=lambda: 1) == 1
    err = capsys.readouterr().err
    assert "K2=B0" in err and "K1" in err
    assert not (root / OUT_DIR).exists()


def test_refuses_a_declaration_for_a_run_not_given(root, capsys):
    argv = [*two_k1_records(root), "--recorded", "K3=K1"]
    assert run(argv, root=root, seed_source=lambda: 1) == 1
    assert "K3" in capsys.readouterr().err
    assert not (root / OUT_DIR).exists()


def test_refuses_the_same_declaration_twice(root, capsys):
    argv = [*two_k1_records(root), "--recorded", "K2=K1", "--recorded", "K2=B0"]
    assert run(argv, root=root, seed_source=lambda: 1) == 1
    assert "重複" in capsys.readouterr().err
    assert not (root / OUT_DIR).exists()


@pytest.mark.parametrize("bad", ["K2", "=K1", "K2=", "K2=../x"])
def test_refuses_a_malformed_recorded_argument(root, capsys, bad):
    argv = [*two_k1_records(root), "--recorded", bad]
    assert run(argv, root=root, seed_source=lambda: 1) == 1
    assert "--recorded" in capsys.readouterr().err
    assert not (root / OUT_DIR).exists()


def test_hides_the_recorded_condition_from_the_pack(root, capsys):
    # 記録での条件名も、束に出れば採点者に条件が分かる
    digest = file_sha256(root / "evaluations/kukai/questions.yaml")
    write(root / K1_DIR / "answers.jsonl", answers_jsonl("K1", digest, content=lambda q, r: "K1 と申す"))
    argv = ["--run", f"B1={B1_DIR}", "--run", f"K2={K1_DIR}", "--recorded", "K2=K1", "--out", OUT_DIR]
    assert run(argv, root=root, seed_source=lambda: 1) == 1
    assert "伏せ" in capsys.readouterr().err
    assert not (root / OUT_DIR).exists()
