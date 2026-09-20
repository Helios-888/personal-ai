"""episodes: 判断様式の層の読み込みと、ホールドアウト漏洩の検出。

伏せた episode の判断や目印が常時層（agent.yaml の parts。principles.md に限らない）に載ると、
Phase 3 のホールドアウト試験は「覚えているか」を測るだけの試験になり無効化する。ここで機械的に弾く。
"""
from pathlib import Path

import pytest
import yaml

from scripts.lib.agent import load_agent
from scripts.lib.episodes import (
    Episode,
    folder_mismatches,
    holdout_leaks,
    load_episodes,
    unverified_principles,
)

BASE = {
    "id": "ep-kukai-0002",
    "title": "恵果からの伝法",
    "year": 805,
    "holdout": False,
    "principle": "授かった法は、身の安全より先に世へ渡す",
    "fact_reliability": "primary_text",
    "sources": ["『御請来目録』"],
    "verify": [],
}


def write_episode(root: Path, name: str, overrides: dict, *, holdout_dir: bool = False) -> Path:
    folder = root / "agents" / "kukai" / "episodes" / ("holdout" if holdout_dir else "")
    folder.mkdir(parents=True, exist_ok=True)
    data = {**BASE, **overrides}
    body = "---\n" + yaml.safe_dump(data, allow_unicode=True, sort_keys=False) + "---\n\n## 状況\n\n本文\n"
    path = folder / name
    path.write_text(body, encoding="utf-8")
    return path


def test_load_episodes_reads_files_in_both_the_main_and_the_holdout_folder(tmp_path):
    write_episode(tmp_path, "0002-a.md", {})
    write_episode(tmp_path, "0001-b.md", {"id": "ep-kukai-0001", "holdout": True, "holdout_markers": ["福州"]}, holdout_dir=True)

    episodes = load_episodes(tmp_path)

    assert [e.id for e in episodes] == ["ep-kukai-0001", "ep-kukai-0002"]  # 番号順
    assert [e.holdout for e in episodes] == [True, False]


def test_load_episodes_returns_the_declared_fields(tmp_path):
    write_episode(tmp_path, "0002-a.md", {"verify": ["灌頂の月次"], "confirmed": ["灌頂名"]})

    episode = load_episodes(tmp_path)[0]

    assert isinstance(episode, Episode)
    assert episode.principle == BASE["principle"]
    assert episode.verify == ["灌頂の月次"]
    assert episode.confirmed == ["灌頂名"]
    assert episode.path.name == "0002-a.md"


def test_load_episodes_names_the_file_that_lacks_a_required_key(tmp_path):
    path = write_episode(tmp_path, "0002-a.md", {})
    path.write_text(path.read_text(encoding="utf-8").replace("principle:", "principe:"), encoding="utf-8")

    with pytest.raises(ValueError) as excinfo:
        load_episodes(tmp_path)

    assert "0002-a.md" in str(excinfo.value)
    assert "principle" in str(excinfo.value)


def test_load_episodes_rejects_a_reliability_value_outside_the_schema(tmp_path):
    write_episode(tmp_path, "0002-a.md", {"fact_reliability": "だいたい正しい"})

    with pytest.raises(ValueError) as excinfo:
        load_episodes(tmp_path)

    assert "fact_reliability" in str(excinfo.value)


def test_load_episodes_requires_markers_on_a_holdout_episode(tmp_path):
    write_episode(tmp_path, "0001-b.md", {"id": "ep-kukai-0001", "holdout": True}, holdout_dir=True)

    with pytest.raises(ValueError) as excinfo:
        load_episodes(tmp_path)

    assert "holdout_markers" in str(excinfo.value)


def test_load_episodes_rejects_a_file_without_frontmatter(tmp_path):
    folder = tmp_path / "agents" / "kukai" / "episodes"
    folder.mkdir(parents=True)
    (folder / "0009-x.md").write_text("# 見出しだけ\n", encoding="utf-8")

    with pytest.raises(ValueError) as excinfo:
        load_episodes(tmp_path)

    assert "0009-x.md" in str(excinfo.value)


def test_folder_mismatches_flags_a_holdout_episode_kept_outside_the_holdout_folder(tmp_path):
    write_episode(tmp_path, "0001-b.md", {"id": "ep-kukai-0001", "holdout": True, "holdout_markers": ["福州"]})

    assert [Path(p).name for p in folder_mismatches(load_episodes(tmp_path))] == ["0001-b.md"]


def test_folder_mismatches_flags_a_used_episode_hidden_in_the_holdout_folder(tmp_path):
    write_episode(tmp_path, "0002-a.md", {}, holdout_dir=True)

    assert len(folder_mismatches(load_episodes(tmp_path))) == 1


def test_folder_mismatches_is_empty_when_every_episode_sits_where_its_flag_says(tmp_path):
    write_episode(tmp_path, "0002-a.md", {})
    write_episode(tmp_path, "0001-b.md", {"id": "ep-kukai-0001", "holdout": True, "holdout_markers": ["福州"]}, holdout_dir=True)

    assert folder_mismatches(load_episodes(tmp_path)) == []


def test_holdout_leaks_reports_the_marker_found_in_the_always_on_layer(tmp_path):
    write_episode(tmp_path, "0001-b.md", {"id": "ep-kukai-0001", "holdout": True, "holdout_markers": ["福州", "観察使"]}, holdout_dir=True)
    principles = "1. 立場が無くとも前に出る。（福州での嘆願）"

    leaks = holdout_leaks(load_episodes(tmp_path), principles)

    assert len(leaks) == 1
    assert "ep-kukai-0001" in leaks[0]
    assert "福州" in leaks[0]


def test_holdout_leaks_names_the_part_where_the_marker_was_found(tmp_path):
    write_episode(tmp_path, "0005-b.md", {"id": "ep-kukai-0005", "holdout": True, "holdout_markers": ["高野山"]}, holdout_dir=True)

    leaks = holdout_leaks(load_episodes(tmp_path), "高野山を開き、", where="agents/kukai/system.md")

    assert leaks == ["ep-kukai-0005（0005-b.md）の「高野山」が agents/kukai/system.md に現れています"]


def test_holdout_leaks_also_catches_the_principle_sentence_itself(tmp_path):
    write_episode(
        tmp_path,
        "0001-b.md",
        {"id": "ep-kukai-0001", "holdout": True, "holdout_markers": ["福州"], "principle": "立場が無くとも前に出る"},
        holdout_dir=True,
    )

    leaks = holdout_leaks(load_episodes(tmp_path), "3. 立場が無くとも前に出る。")

    assert len(leaks) == 1


def test_holdout_leaks_is_empty_when_principles_avoids_every_marker(tmp_path):
    write_episode(tmp_path, "0001-b.md", {"id": "ep-kukai-0001", "holdout": True, "holdout_markers": ["福州"]}, holdout_dir=True)
    write_episode(tmp_path, "0002-a.md", {})

    assert holdout_leaks(load_episodes(tmp_path), "1. 形式の伝達では本質は渡らない。") == []


def test_holdout_leaks_ignores_markers_of_episodes_that_are_not_held_out(tmp_path):
    write_episode(tmp_path, "0002-a.md", {"holdout_markers": ["恵果"]})

    assert holdout_leaks(load_episodes(tmp_path), "恵果から法を受けた") == []


def test_unverified_principles_lists_episodes_still_carrying_open_verify_items(tmp_path):
    write_episode(tmp_path, "0002-a.md", {"principle": "隠さず数え上げる", "verify": ["提出日"]})

    flagged = unverified_principles(load_episodes(tmp_path), "1. 隠さず数え上げる。")

    assert len(flagged) == 1
    assert "ep-kukai-0002" in flagged[0]


def test_unverified_principles_is_empty_once_verify_is_cleared(tmp_path):
    write_episode(tmp_path, "0002-a.md", {"principle": "隠さず数え上げる", "verify": []})

    assert unverified_principles(load_episodes(tmp_path), "1. 隠さず数え上げる。") == []


# --- 実際のリポジトリに対する検査（これが本番の防波堤） ---

REPO = Path(__file__).resolve().parents[1]
AGENT = "agents/kukai/agent.yaml"
PRINCIPLES = "agents/kukai/principles.md"


def test_repository_episodes_all_parse_and_sit_in_the_right_folder():
    episodes = load_episodes(REPO)

    assert len(episodes) >= 8
    assert folder_mismatches(episodes) == []
    # 2026-09-20、利用者の判断でホールドアウトを解放した（Phase 5 設計書「方針」1）。
    # 伏せたままなのは 0006 泰範だけで、これは根拠不足で使わない episode である（H01・H02 の出どころではない）
    assert sum(1 for e in episodes if e.holdout) == 1


def test_repository_holdout_judgements_have_not_leaked_into_the_always_on_layer():
    # principles.md だけでなく、常時層を成す parts をすべて見る。
    # 2026-09-18、system.md の自己紹介「高野山を開き」（H02 の目印）を principles.md だけの検査が見逃した
    parts = load_agent(REPO, AGENT)["system_prompt"]["parts"]
    assert PRINCIPLES in parts  # 検査の対象から原則が外れていれば、この検査は空回りする
    episodes = load_episodes(REPO)

    leaks = [
        leak
        for part in parts
        for leak in holdout_leaks(episodes, (REPO / part).read_text(encoding="utf-8"), where=part)
    ]

    assert leaks == []
