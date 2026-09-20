"""凍結した評価セット（evaluations/kukai/）が凍結時のままで、形が設計どおりであることを確かめる。

設計は docs/specs/2026-09-18-phase3-design.md「凍結の仕組み」。問いと正解（questions.yaml）は完全凍結、
採点基準（rubric.yaml）は 1 回だけ改訂できる。凍結後の誤りは errata.yaml に記録する。
付け足し・書き換えの判定（faithfulness.yaml、Phase 4）は rubric とは別の定義で、K1 を取る前に凍結する。
"""
import datetime as dt
import re
from collections import Counter
from pathlib import Path

import pytest
import yaml

from scripts.lib.episodes import load_episodes
from scripts.lib.freeze import verify_frozen

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "evaluations" / "kukai"
QUESTIONS = EVAL_DIR / "questions.yaml"
RUBRIC = EVAL_DIR / "rubric.yaml"
ERRATA = EVAL_DIR / "errata.yaml"
FAITHFULNESS = EVAL_DIR / "faithfulness.yaml"
AGENT = ROOT / "agents" / "kukai" / "agent.yaml"

# 凍結時の指紋。記録（*.sha256）とここの二か所で照合する。記録ごと書き換えても、ここを書き換えない限り落ち、
# 書き換えればテストの差分に必ず現れる。
QUESTIONS_DIGEST = "2e2ca6f789ced133b35b075243318acdeabedc26d590f7df41c0c831a3d317de"  # 2026-09-18 凍結
# rubric の指紋の履歴。改訂したら末尾に足し、rubric.yaml の revisions にも 1 件足す（改訂は 1 回まで）。
RUBRIC_DIGESTS = ("c125b0ccad027065a3bc952e04f12eabba2b8fb67f9cb01564fd61d8cb7ec4db",)  # 2026-09-18 凍結
MAX_RUBRIC_REVISIONS = 1
FAITHFULNESS_DIGEST = "271b2cad8ba85a86a1011655b4f8606227ce0430e70a77f4c79fd2ea8973b9b4"  # 2026-09-19 凍結（K1 を取る前）

EXPECTED_COUNTS = {"factual": 15, "attribution": 5, "trap": 10, "holdout": 2}
RUBRIC_KEYS = ("wrong_if", "grading", "note")
# 伏せた episode のうち、問いに入れてはならない節（判断そのものと、それを裏づける記述）
JUDGEMENT_HEADINGS = ("判断", "根拠となる記述")
MIN_SENTENCE_CHARS = 6  # これより短い文（「」だけ等）は照合しない


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _sections(text: str, headings: tuple[str, ...]) -> str:
    """Markdown の「## 見出し」単位で、指定した見出しの節の本文だけをつなげて返す。"""
    kept, current = [], None
    for line in text.splitlines():
        heading = re.match(r"^## (.+?)\s*$", line)
        if heading:
            current = heading.group(1)
        elif current in headings:
            kept.append(line)
    return "\n".join(kept)


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"[。\n]", text) if len(s.strip()) >= MIN_SENTENCE_CHARS]


@pytest.fixture(scope="module")
def questions() -> list[dict]:
    return _load(QUESTIONS)["questions"]


@pytest.fixture(scope="module")
def rubric() -> dict:
    return _load(RUBRIC)


@pytest.fixture(scope="module")
def holdouts(questions) -> list[dict]:
    found = [q for q in questions if q["kind"] == "holdout"]
    assert found, "ホールドアウトの問いが見つからない（以下の検査が空回りする）"
    return found


def test_questions_are_exactly_as_frozen():
    assert verify_frozen(QUESTIONS) == QUESTIONS_DIGEST


def test_rubric_is_the_latest_pinned_version():
    assert verify_frozen(RUBRIC) == RUBRIC_DIGESTS[-1]


def test_faithfulness_is_exactly_as_frozen():
    assert verify_frozen(FAITHFULNESS) == FAITHFULNESS_DIGEST


def test_faithfulness_defines_added_and_altered():
    judgements = _load(FAITHFULNESS)["judgements"]
    assert set(judgements) == {"added", "altered"}
    for name, judgement in judgements.items():
        assert judgement.get("question"), name


def test_rubric_revisions_match_the_pinned_history(rubric):
    revisions = rubric["revisions"]  # 欄ごと消されたら KeyError で落とす
    assert isinstance(revisions, list)
    assert len(revisions) == len(RUBRIC_DIGESTS) - 1
    assert len(revisions) <= MAX_RUBRIC_REVISIONS


@pytest.mark.parametrize("path", [QUESTIONS, RUBRIC, FAITHFULNESS], ids=lambda p: p.name)
def test_frozen_date_is_set(path):
    assert isinstance(_load(path)["frozen"], dt.date)


def test_question_counts_match_the_design(questions):
    assert dict(Counter(q["kind"] for q in questions)) == EXPECTED_COUNTS


def test_question_ids_are_unique(questions):
    ids = [q["id"] for q in questions]
    assert len(ids) == len(set(ids))


def test_every_key_point_has_a_source(questions):
    for question in questions:
        assert question["key_points"], question["id"]
        for key_point in question["key_points"]:
            assert key_point.get("point") and key_point.get("source"), question["id"]


def test_scoring_rules_live_only_in_the_rubric(questions):
    for question in questions:
        assert not set(question) & set(RUBRIC_KEYS), question["id"]


def test_rubric_covers_exactly_the_questions(questions, rubric):
    assert set(rubric["questions"]) == {q["id"] for q in questions}
    for question_id, rule in rubric["questions"].items():
        assert rule.get("wrong_if"), question_id


def test_errata_is_a_list_of_entries_for_existing_questions(questions):
    errata = _load(ERRATA)["errata"]  # 欄の名前を打ち間違えたら KeyError で落とす
    assert isinstance(errata, list)
    ids = {q["id"] for q in questions}
    for entry in errata:
        assert entry.get("id") in ids and entry.get("reason"), entry


def test_holdout_questions_point_to_the_episode_they_were_written_from(holdouts):
    # 2026-09-20、利用者の判断で伏せを解いた（Phase 5 設計書「方針」1）。
    # 「その episode が伏せられているか」の検査はここで役目を終えた。問いと episode の対応だけを守る。
    # 解放後、H01・H02 は成績に数えない（採点フォルダの holdout-released.yaml で宣言する）
    episodes = {e.id: e for e in load_episodes(ROOT)}
    for question in holdouts:
        assert question["episode"] in episodes, question["id"]


def test_holdout_questions_do_not_reveal_the_judgement(holdouts):
    # 問いに判断の文が入っていたら、ホールドアウト試験は答えを見せて問うことになる
    episodes = {e.id: e for e in load_episodes(ROOT)}
    for question in holdouts:
        episode = episodes[question["episode"]]
        body = episode.path.read_text(encoding="utf-8")
        sentences = _sentences(_sections(body, JUDGEMENT_HEADINGS))
        assert sentences, f"{episode.id} の判断の節が読めない（見出しが変わった？）"
        leaked = [s for s in (*sentences, episode.principle) if s in question["text"]]
        assert not leaked, (question["id"], leaked)


def test_evaluation_files_never_enter_the_always_on_layer():
    eval_root = (ROOT / "evaluations").resolve()
    parts = _load(AGENT)["system_prompt"]["parts"]
    leaked = [part for part in parts if (ROOT / part).resolve().is_relative_to(eval_root)]
    assert not leaked
