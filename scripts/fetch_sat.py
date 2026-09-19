"""SAT の空海著作 9 点を取得し、Knowledge に入れる Markdown にする（Phase 4 手順 4）。

設計は docs/specs/2026-09-19-phase4-design.md「段 2」。SAT の利用条件に従い、取得はこの 9 点だけ、非営利・再配布しない
（保存先は GitHub private）。1 ページごとに 1 秒あける。出力先に同名のファイルがあれば書かずに止まる。

使い方（llm01 のリポジトリ直下で）:
  .venv/bin/python scripts/fetch_sat.py --out-root knowledge/kukai
  → <out-root>/<区分>/<区分>__<書名>.md（区分は primary か later-attribution。区分を名の頭に付けるのは rules.md 規則 4）
  .venv/bin/python scripts/fetch_sat.py --out-root <dir> --only T2428   … 1 点だけ
"""
import argparse
import datetime as dt
import sys
import time
from pathlib import Path
from typing import Callable, Optional

if __package__ in (None, ""):  # スクリプトとして直接実行されたとき、リポジトリ直下を import 経路に加える
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import requests  # noqa: E402

from scripts.lib.exclusions import apply_cuts, load_exclusions  # noqa: E402
from scripts.lib.sat import merge_lines, next_useid, parse_lines, render_markdown  # noqa: E402

BASE = "https://21dzk.l.u-tokyo.ac.jp/SAT/ddb-sat2.php"
USER_AGENT = "Mozilla/5.0 (personal-ai; non-commercial research)"
WAIT_SECONDS = 1.0
MAX_PAGES = 30  # 十住心論でも 10 ページに満たない。続きのリンクが回り続けたときの止め
DEFAULT_EXCLUSIONS = "evaluations/kukai/knowledge-exclusions.yaml"
WORKS = (  # （番号, 書名, 区分）
    ("T2161", "御請来目録", "primary"),
    ("T2203A", "般若心経秘鍵", "primary"),
    ("T2425", "秘密曼荼羅十住心論", "primary"),
    ("T2426", "秘蔵宝鑰", "primary"),
    ("T2427", "弁顕密二教論", "primary"),
    ("T2428", "即身成仏義", "primary"),
    ("T2429", "声字実相義", "primary"),
    ("T2430", "吽字義", "primary"),
    ("T2431", "御遺告", "later-attribution"),
)
# 大正蔵の同じ番号に、後世の別本が続けて収められているもの：（本人の本文の最後の行, 範囲の注記）
LAST_LINES = {
    "T2428": ("T2428_.77.0384a19", "空海の本文（0381b16–0384a19）だけ。0384a22「眞言宗卽身成佛義問答一卷」以降の異本 6 篇は後世のものなので入れない"),
}
NOTES = {
    "primary": "空海本人の著作",
    "later-attribution": "後世に空海の名で伝えられた文書。空海の言葉として引用しない",
}


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SAT の空海著作 9 点を取得し、Knowledge に入れる Markdown にする")
    parser.add_argument("--out-root", required=True, help="区分ごとのフォルダを置く場所（リポジトリ直下から）")
    parser.add_argument("--only", default=None, help="取得する番号をカンマ区切りで（例 T2428）")
    parser.add_argument("--exclusions", default=DEFAULT_EXCLUSIONS,
                        help="伏せた場面に触れる箇所の一覧（削る範囲を取り込みのときに置き換える）")
    return parser.parse_args(argv)


def fetch_work(number: str, get: Callable[[str], str], wait: Callable[[float], None]) -> tuple[str, list]:
    """1 著作の全行。続きのリンクが無くなるまでたどる。"""
    useid = number[1:] + "_"
    url = f"{BASE}?mode=detail&useid={useid}"
    first_url, lines, seen = url, [], set()
    for _ in range(MAX_PAGES):
        page = get(url)
        lines = merge_lines(lines, parse_lines(page))
        following = next_useid(page)
        if following is None:
            break
        if following in seen:
            raise ValueError(f"{number} の続きのリンクが回っています（{following}）")
        seen.add(following)
        url = f"{BASE}?mode=detail&nonum=&kaeri=&mode2=2&useid={following}"
        wait(WAIT_SECONDS)
    else:
        raise ValueError(f"{number} が {MAX_PAGES} ページを超えました")
    stray = [line_id for line_id, _ in lines if not line_id.startswith(number)]
    if not lines or stray:
        raise ValueError(f"{number} の行が取れないか、別の番号の行が混じっています: {stray[:3]}")
    return first_url, lines


def http_get(url: str) -> str:
    response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=60)
    response.raise_for_status()
    response.encoding = "utf-8"
    return response.text


def run(argv: list[str], root: Path, get: Callable[[str], str] = http_get,
        wait: Callable[[float], None] = time.sleep, today: Optional[str] = None) -> int:
    args = parse_args(argv)
    only = set(args.only.split(",")) if args.only else None
    works = [w for w in WORKS if only is None or w[0] in only]
    if not works:
        print(f"error: 取得する著作がありません: {args.only}", file=sys.stderr)
        return 2
    fetched = today or dt.date.today().isoformat()
    cuts = load_exclusions((root / args.exclusions).read_text(encoding="utf-8")).cuts
    for number, title, reliability in works:
        name = f"{reliability}/{reliability}__{title}.md"  # 一覧の file 欄と同じ書き方
        out = root / args.out_root / name
        if out.exists():
            print(f"error: {out} は既にあります（上書きしない）", file=sys.stderr)
            return 1
        url, lines = fetch_work(number, get, wait)
        last, scope = LAST_LINES.get(number, (None, ""))
        if last:
            ids = [line_id for line_id, _ in lines]
            if last not in ids:
                raise ValueError(f"{number} に本文の最後の行 {last} がありません")
            lines = lines[: ids.index(last) + 1]
        lines = apply_cuts(lines, [cut for cut in cuts if cut.file == name])
        text = render_markdown(title=title, number=number, reliability=reliability, reliability_note=NOTES[reliability],
                               url=url, fetched=fetched, lines=lines, scope_note=scope)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "x", encoding="utf-8", newline="\n") as file:
            file.write(text)
        print(f"{number} {title}: {len(lines)} 行（{lines[0][0]}–{lines[-1][0]}）→ {out.relative_to(root)}")
        wait(WAIT_SECONDS)
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    return run(sys.argv[1:] if argv is None else argv, root=Path(__file__).resolve().parents[1])


if __name__ == "__main__":
    sys.exit(main())
