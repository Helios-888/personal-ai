"""伏せた束の回答本文から「」の中身を重複なく集め、番号つき一覧を書く（呼び名・前後の文は付けない）。"""
import re
from pathlib import Path

PACK = Path("evaluations/kukai/scoring/2026-09-19-baseline/pack")
OUT = Path("evaluations/kukai/scoring/2026-09-19-baseline/titles/kagi-strings.txt")
FENCE = "~~~~~~"
KAGI = re.compile(r"「([^「」]{1,40})」")


def bodies(sheet: Path) -> list[str]:
    out, body, inside = [], [], False
    for line in sheet.read_text(encoding="utf-8").splitlines():
        if line == FENCE:
            if inside:
                out.append("\n".join(body))
                body = []
            inside = not inside
            continue
        if inside:
            body.append(line)
    return out


def main() -> None:
    found = set()
    for sheet in sorted(PACK.glob("[AFHT]*.md")):
        for body in bodies(sheet):
            found.update(s.replace("\n", " ") for s in KAGI.findall(body))
    strings = sorted(found)
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="\n") as f:
        for number, text in enumerate(strings, 1):
            f.write(f"{number}\t{text}\n")
    print(f"{len(strings)} 種を {OUT} に書いた")


if __name__ == "__main__":
    main()
