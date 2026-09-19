"""Astra の印付け結果を検査し、既知 8 件の抜き打ち検査と Claude の印との比較をする（読み取りのみ）。"""
import json
from pathlib import Path

TITLES = Path("evaluations/kukai/scoring/2026-09-19-baseline/titles")
SENTINELS = {
    "いろは歌": ["29"], "即身成仏義": ["277"], "吽字義": ["299"], "密教具足之儀": ["413"],
    "当字解": ["433"], "御請来目録": ["440"], "御遺告": ["441", "442"], "文鏡秘府論": ["528"],
}


def tsv(path: Path) -> dict[str, str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return dict(line.split("\t", 1) for line in lines if line.strip())


def main() -> None:
    src = tsv(TITLES / "kagi-strings.txt")
    raw = (TITLES / "astra-marks-result.txt").read_bytes()
    print("改行", "CRLF" if b"\r\n" in raw else "LF", "/ BOM", "あり" if raw.startswith(b"\xef\xbb\xbf") else "なし")
    marks, problems = {}, []
    for i, line in enumerate(raw.decode("utf-8-sig").splitlines(), 1):
        text = line.strip()
        if not text:
            continue
        if text.startswith("```"):
            problems.append(f"行 {i}: 囲みの印 {text}")
            continue
        try:
            row = json.loads(text)
        except json.JSONDecodeError:
            problems.append(f"行 {i}: JSON でない {text[:60]}")
            continue
        number, string = str(row.get("number")), row.get("string")
        if set(row) != {"number", "string"}:
            problems.append(f"行 {i}: キー {sorted(row)}")
        if src.get(number) != string:
            problems.append(f"行 {i}: 番号 {number} の語が一覧と違う {string!r} / {src.get(number)!r}")
        if number in marks:
            problems.append(f"行 {i}: 重複 {number}")
        marks[number] = string
    order = [int(n) for n in marks]
    print("印", len(marks), "/", "昇順" if order == sorted(order) else "順序が乱れている")
    print("問題", problems if problems else "なし")
    for name, numbers in SENTINELS.items():
        hit = any(n in marks for n in numbers)
        print(f"  {name}: {'印あり' if hit else '印なし'}")
    claude = tsv(TITLES / "marks-claude.txt")
    a, c = set(marks), set(claude)
    print(f"共通 {len(a & c)} / Astra のみ {len(a - c)} / Claude のみ {len(c - a)} / 和集合 {len(a | c)}")
    print("Astra のみ:", " / ".join(src[n] for n in sorted(a - c, key=int)))
    print("Claude のみ:", " / ".join(src[n] for n in sorted(c - a, key=int)))


if __name__ == "__main__":
    main()
