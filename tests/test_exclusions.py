"""exclusions: 伏せた場面に触れる原典の箇所を、削るか理由つきで残すかに登録させる見張り。"""
from pathlib import Path

import pytest

from scripts.lib.exclusions import apply_cuts, load_exclusions, problems

ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE = ROOT / "knowledge" / "kukai"
EXCLUSIONS = ROOT / "evaluations" / "kukai" / "knowledge-exclusions.yaml"
INGESTED = ("primary", "later-attribution")  # Knowledge に入れるフォルダ（設計書「段 2」）

YAML = """\
terms: [高野, 南山]
cut:
  - file: later/x.md
    first: T2431_.77.0409b02
    last: T2431_.77.0409b03
    holdout: H02
    reason: 山を請う判断
kept:
  - line: T2431_.77.0409b05
    terms: [高野]
    reason: 奥書
"""
LINES = [
    ("T2431_.77.0409b01", "序。"),
    ("T2431_.77.0409b02", "表請南山。"),
    ("T2431_.77.0409b03", "移入南山。"),
    ("T2431_.77.0409b04", "餘事。"),
    ("T2431_.77.0409b05", "書於高野山。"),
]


def as_file(lines):
    return "# x\n\n- 出典\n\n" + "\n".join(f"{i} {b}" if i else b for i, b in lines) + "\n"


def test_apply_cuts_replaces_the_range_with_a_marker_only():
    cut = load_exclusions(YAML).cuts[0]
    assert apply_cuts(LINES, [cut]) == [
        ("T2431_.77.0409b01", "序。"),
        ("", "〔略 T2431_.77.0409b02–T2431_.77.0409b03〕"),
        ("T2431_.77.0409b04", "餘事。"),
        ("T2431_.77.0409b05", "書於高野山。"),
    ]


def test_apply_cuts_refuses_a_range_that_is_not_in_the_text():
    cut = load_exclusions(YAML.replace("last: T2431_.77.0409b03", "last: T2431_.77.0409c03")).cuts[0]
    with pytest.raises(ValueError, match="0409c03"):
        apply_cuts(LINES, [cut])


def test_no_problems_when_cut_and_every_hit_is_registered():
    ex = load_exclusions(YAML)
    assert problems({"later/x.md": as_file(apply_cuts(LINES, ex.cuts))}, ex) == []


def test_the_uncut_copy_is_caught():
    # 設計書「削る前の写しでテストが落ちることを確かめてから使う」
    found = problems({"later/x.md": as_file(LINES)}, load_exclusions(YAML))
    assert any("T2431_.77.0409b02" in p and "削る" in p for p in found)
    assert any("南山" in p for p in found)


def test_an_unregistered_hit_is_caught_even_across_a_line_break():
    lines = [*LINES[:4], ("T2431_.77.0409b05", "書於高"), ("T2431_.77.0409b06", "野山。")]
    ex = load_exclusions(YAML.replace("T2431_.77.0409b05", "T2431_.77.0409b07"))
    found = problems({"later/x.md": as_file(apply_cuts(lines, ex.cuts))}, ex)
    assert any("T2431_.77.0409b05" in p and "高野" in p for p in found)


def test_a_registered_hit_that_no_longer_occurs_is_reported():
    ex = load_exclusions(YAML)
    lines = [line for line in apply_cuts(LINES, ex.cuts) if line[0] != "T2431_.77.0409b05"]
    assert any("出現しない" in p for p in problems({"later/x.md": as_file(lines)}, ex))


def test_every_entry_needs_a_reason():
    with pytest.raises(ValueError, match="reason"):
        load_exclusions(YAML.replace("    reason: 奥書\n", ""))


def test_the_ingested_knowledge_passes_the_holdout_guard():
    files = {str(p.relative_to(KNOWLEDGE)): p.read_text(encoding="utf-8")
             for folder in INGESTED for p in sorted((KNOWLEDGE / folder).glob("*.md"))}
    assert len(files) == 9, sorted(files)  # 空回りを防ぐ（設計書の資料 9 点）
    assert problems(files, load_exclusions(EXCLUSIONS.read_text(encoding="utf-8"))) == []
