"""Astra の JSON 行と語尾方式の候補を、build_titles_bundle.py が読む「番号<TAB>語」形式に揃える。"""
import json
import re
from pathlib import Path

TITLES = Path("evaluations/kukai/scoring/2026-09-19-baseline/titles")
PUNCT = re.compile(r"[、。，．！？!?\s・…]")
SUFFIXES = (
    "論", "義", "経", "章", "鈔", "抄", "疏", "録", "集", "記", "帰", "鑰", "式",
    "軌", "讃", "頌", "偈", "序", "訣", "伝", "表", "書", "釈", "文", "状", "図", "儀",
)


def write_tsv(path: Path, rows: list[tuple[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for number, string in rows:
            f.write(f"{number}\t{string}\n")


def main() -> None:
    src = [line.split("\t", 1) for line in (TITLES / "kagi-strings.txt").read_text(encoding="utf-8").splitlines()]
    astra = []
    for line in (TITLES / "astra-marks-result.txt").read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            astra.append((str(row["number"]), row["string"]))
    write_tsv(TITLES / "marks-astra.txt", astra)
    suffix = []
    for number, string in src:
        inner = re.sub(r"[『』「」〈〉《》\s]", "", string)
        if 3 <= len(inner) <= 20 and not PUNCT.search(string) and inner.endswith(SUFFIXES):
            suffix.append((number, string))
    write_tsv(TITLES / "marks-suffix.txt", suffix)
    union = {n for n, _ in astra} | {n for n, _ in suffix}
    union |= {line.split("\t", 1)[0] for line in (TITLES / "marks-claude.txt").read_text(encoding="utf-8").splitlines() if line.strip()}
    print(f"Astra {len(astra)} / 語尾方式 {len(suffix)} / 3 つの和集合 {len(union)}")


if __name__ == "__main__":
    main()
