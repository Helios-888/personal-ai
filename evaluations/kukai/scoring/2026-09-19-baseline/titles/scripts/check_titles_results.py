"""書名の判定 2 人分（titles/result-*.txt）を機械検査し、区分の一致率を出す（読み取りのみ）。

区分：書名でない / 他者の作 / 自分の作。
"""
import json
import sys
from collections import Counter
from pathlib import Path

TITLES = Path("evaluations/kukai/scoring/2026-09-19-baseline/titles")
KEYS = {"item_id", "title", "is_title", "as_own", "reason"}


def category(row: dict) -> str:
    if not row["is_title"]:
        return "書名でない"
    return "自分の作" if row["as_own"] else "他者の作"


def load(path: Path, expected: set[str]) -> tuple[dict[str, dict], list[str]]:
    rows, problems = {}, []
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("```"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            problems.append(f"行 {number}: JSON として読めない ({exc})")
            continue
        if set(row) != KEYS:
            problems.append(f"行 {number}: キーが違う {sorted(row)}")
            continue
        if not isinstance(row["is_title"], bool) or not isinstance(row["as_own"], bool):
            problems.append(f"{row['item_id']}: is_title / as_own が真偽値でない")
        if row["as_own"] and not row["is_title"]:
            problems.append(f"{row['item_id']}: is_title が false なのに as_own が true")
        if row["item_id"] in rows:
            problems.append(f"{row['item_id']}: 重複")
        rows[row["item_id"]] = row
    missing, extra = sorted(expected - set(rows)), sorted(set(rows) - expected)
    if missing:
        problems.append(f"欠け {missing}")
    if extra:
        problems.append(f"余分 {extra}")
    return rows, problems


def main() -> int:
    items = {}
    for line in (TITLES / "items.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            items[item["item_id"]] = item
    graders = {}
    ok = True
    for path in sorted(TITLES.glob("result-*.txt")):
        rows, problems = load(path, set(items))
        print(f"{path.name}: {len(rows)} 件 {'OK' if not problems else 'NG'}")
        for p in problems:
            print(f"  - {p}")
        ok = ok and not problems
        graders[path.stem.removeprefix("result-")] = rows
    if len(graders) != 2 or not ok:
        return 1
    (a_name, a), (b_name, b) = sorted(graders.items())
    common = sorted(set(a) & set(b))
    agree = [i for i in common if category(a[i]) == category(b[i])]
    print()
    print(f"区分の一致：{len(agree)}/{len(common)} = {len(agree) / len(common):.1%}")
    print(f"混同表（{a_name} → {b_name}）")
    for (x, y), n in sorted(Counter((category(a[i]), category(b[i])) for i in common).items(), key=lambda t: -t[1]):
        print(f"  {x} → {y}: {n}{'' if x == y else '  ←不一致'}")
    print()
    print(f"不一致（項目 / 回答 / 語 / {a_name} / {b_name}）")
    for i in common:
        if category(a[i]) != category(b[i]):
            print(f"  {i} / {items[i]['answer_id']} / {items[i]['string']} / {category(a[i])} / {category(b[i])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
