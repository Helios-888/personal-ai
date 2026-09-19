"""伏せた場面（ホールドアウト）に触れる原典の箇所の見張り（Phase 4 設計書「段 2」漏れの見張り）。純粋関数。

目印の文字列照合（episodes.holdout_leaks）は現代語の常時層向けで、漢文・旧字体の原典には効かない。原典には、
検索語の出現を一つずつ「削る（cut）」か「理由つきで残す（kept）」かに登録させ、登録に無い出現があれば問題として返す。
語は行をまたいで現れうるので、本文をつないでから探し、見つかった位置の行 ID で照合する。
"""
import re
from dataclasses import dataclass

import yaml

from scripts.lib.sat import LINE_ID

_LINE = re.compile(r"^(" + LINE_ID + r") (.*)$")


@dataclass(frozen=True)
class Cut:
    file: str  # knowledge/kukai からの相対パス
    first: str
    last: str

    @property
    def marker(self) -> str:
        return f"〔略 {self.first}–{self.last}〕"

    def covers(self, line_id: str) -> bool:
        return self.first <= line_id <= self.last  # 同じ著作の行 ID は文字列の順が本文の順


@dataclass(frozen=True)
class Exclusions:
    terms: tuple[str, ...]
    cuts: tuple[Cut, ...]
    kept: frozenset  # （行 ID, 語）


def load_exclusions(text: str) -> Exclusions:
    data = yaml.safe_load(text)
    for section in ("cut", "kept"):
        for entry in data.get(section) or []:
            if not str(entry.get("reason") or "").strip():
                raise ValueError(f"{section} の項目に reason がありません: {entry}")
    cuts = tuple(Cut(e["file"], e["first"], e["last"]) for e in data.get("cut") or [])
    kept = frozenset((e["line"], term) for e in data.get("kept") or [] for term in e["terms"])
    return Exclusions(tuple(data["terms"]), cuts, kept)


def apply_cuts(lines: list[tuple[str, str]], cuts: list[Cut]) -> list[tuple[str, str]]:
    """削る範囲の行を、印の 1 行（行 ID なし）に置き換える。範囲の端の行が本文に無ければ止める。"""
    ids = {line_id for line_id, _ in lines}
    for cut in cuts:
        missing = [end for end in (cut.first, cut.last) if end not in ids]
        if missing:
            raise ValueError(f"削る範囲の端の行が本文にありません: {', '.join(missing)}")
    result: list[tuple[str, str]] = []
    for line_id, body in lines:
        cut = next((c for c in cuts if c.covers(line_id)), None)
        if cut is None:
            result.append((line_id, body))
        elif line_id == cut.first:
            result.append(("", cut.marker))
    return result


def problems(files: dict[str, str], ex: Exclusions) -> list[str]:
    """（相対パス → 本文）の束を照合し、問題を文で返す。空なら見張りを通る。"""
    found: list[str] = []
    seen: set = set()
    for cut in ex.cuts:
        if cut.file not in files:
            found.append(f"削る範囲のあるファイルがありません: {cut.file}")
    for name, text in files.items():
        lines = [m.groups() for m in map(_LINE.match, text.splitlines()) if m]
        for cut in (c for c in ex.cuts if c.file == name):
            left = [line_id for line_id, _ in lines if cut.covers(line_id)]
            if left:
                found.append(f"{name}：削るはずの行が残っています（{left[0]} ほか {len(left)} 行）")
            if cut.marker not in text:
                found.append(f"{name}：削った印 {cut.marker} がありません")
        for line_id, term in _hits(lines, ex.terms):
            seen.add((line_id, term))
            if (line_id, term) not in ex.kept:
                found.append(f"{name} {line_id} に「{term}」があります（登録に無い）")
    stale = sorted(ex.kept - seen)
    if stale:
        found.append(f"登録にあるが出現しない語があります: {stale}")
    return found


def _hits(lines: list[tuple[str, str]], terms: tuple[str, ...]) -> list[tuple[str, str]]:
    joined, owner = "", []
    for line_id, body in lines:
        joined += body
        owner += [line_id] * len(body)
    return [(owner[m.start()], term) for term in terms for m in re.finditer(re.escape(term), joined)]
