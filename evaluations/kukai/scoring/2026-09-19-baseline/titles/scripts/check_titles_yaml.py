"""titles.yaml が自著として挙げられた書名をすべて覆っているかを検査する（読み取りのみ）。"""
import json
import re
import sys
from pathlib import Path

import yaml

B = Path("evaluations/kukai/scoring/2026-09-19-baseline")
CATEGORIES = {"kukai_work", "misnamed", "other_author", "later", "not_found", "not_title"}


def norm(title: str) -> str:
    return re.sub(r"[『』「」〈〉《》\s]|（[^）]*）", "", title)


def cited() -> set[str]:
    out = set()
    for line in (B / "scores.jsonl").read_text(encoding="utf-8").splitlines():
        out |= {norm(t["title"]) for t in json.loads(line)["titles"] if t["as_own"]}
    for name in ("claude", "astra"):
        for line in (B / "titles" / f"result-{name}.txt").read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if row["as_own"]:
                    out.add(norm(row["title"]))
    return out


def main() -> int:
    raw = (B / "titles.yaml").read_bytes()
    problems = []
    if b"\r" in raw:
        problems.append("CR が含まれる")
    entries = yaml.safe_load(raw.decode("utf-8"))["titles"]
    keys = [e["title"] for e in entries]
    if len(keys) != len(set(keys)):
        problems.append("title が重複")
    for e in entries:
        if e["category"] not in CATEGORIES:
            problems.append(f"{e['title']}: 区分が不正 {e['category']}")
        if e["category"] != "not_title" and not e.get("sources"):
            problems.append(f"{e['title']}: 出典が無い")
        if any(norm(f) != e["title"] for f in e["forms"]):
            problems.append(f"{e['title']}: forms を寄せると title と一致しない {e['forms']}")
        if e["category"] == "misnamed" and not e.get("refers_to"):
            problems.append(f"{e['title']}: misnamed なのに refers_to が無い")
        if e["category"] == "other_author" and not e.get("author"):
            problems.append(f"{e['title']}: other_author なのに author が無い")
    need = cited()
    missing, extra = sorted(need - set(keys)), sorted(set(keys) - need)
    if missing:
        problems.append(f"照合表に無い書名 {missing}")
    if extra:
        problems.append(f"どの回答も挙げていない書名 {extra}")
    counts = {}
    for e in entries:
        counts[e["category"]] = counts.get(e["category"], 0) + 1
    print(f"書名 {len(entries)} / 挙げられた書名 {len(need)}")
    print("区分ごとの数", counts)
    print("問題", problems if problems else "なし")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
