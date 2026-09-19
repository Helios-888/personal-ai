"""SAT 大蔵経テキストデータベースの詳細ページ（ddb-sat2.php?mode=detail）から、行 ID つきの本文を取り出す。純粋関数。

設計は docs/specs/2026-09-19-phase4-design.md「段 2」。行 ID は引用の根拠に使うので本文に残す。
校異・注の番号（<button>）は除き、割注（小さい字の注）は括弧に入れて残す。
長い著作は 1 ページに全文が載らない。ページ末の「続き」（id="nextl"）をたどると、前の行も含めて表示が伸びていく。
"""
import html
import re
from typing import Optional

LINE_ID = r"T\d{4}[A-Z]?_?\.\d{2}\.\d{4}[abc]\d{2}"
_LINE_START = re.compile(r'<span style="color:black">(' + LINE_ID + r"): </span>")
_BUTTON = re.compile(r"<button\b[^>]*>.*?</button>", re.S)  # 校異・注の番号
_SMALL = re.compile(r'<span style="font-size:8pt">(.*?)</span>', re.S)  # 割注
_LINE_END = re.compile(r"<br\s*/?>|</table>|<div\b")
_TAG = re.compile(r"<[^>]+>")
_NEXT = re.compile(r'id="nextl"[^>]*useid=([^\'"&]+)')


def parse_lines(page: str) -> list[tuple[str, str]]:
    """（行 ID, 本文）の並び。ページの順のまま返す。"""
    text = _SMALL.sub(r"（\1）", _BUTTON.sub("", page))
    starts = list(_LINE_START.finditer(text))
    lines = []
    for index, match in enumerate(starts):
        end = starts[index + 1].start() if index + 1 < len(starts) else len(text)
        segment = text[match.end():end]
        cut = _LINE_END.search(segment)
        body = html.unescape(_TAG.sub("", segment[: cut.start()] if cut else segment))
        lines.append((match.group(1), body.replace("\xa0", " ").strip()))
    return lines


def next_useid(page: str) -> Optional[str]:
    """ページ末の「続き」が指す useid。無ければ最後のページ。"""
    found = _NEXT.search(page)
    return found.group(1) if found else None


def merge_lines(seen: list[tuple[str, str]], more: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """既に読んだ行に、新しい行だけを順に足す。同じ行 ID で本文が違えば止める（取り違えを黙って混ぜない）。"""
    known = dict(seen)
    merged = list(seen)
    for line_id, body in more:
        if line_id in known:
            if known[line_id] != body:
                raise ValueError(f"{line_id} の本文がページによって違います")
            continue
        known[line_id] = body
        merged.append((line_id, body))
    return merged


def render_markdown(*, title: str, number: str, reliability: str, reliability_note: str, url: str, fetched: str,
                    lines: list[tuple[str, str]], scope_note: str = "") -> str:
    """Knowledge に入れる 1 著作の Markdown。見出しの下に出典、その下に「行 ID 本文」を 1 行ずつ（空の行は除く）。"""
    head = [
        f"# {title}（{number}）",
        "",
        f"- 区分：{reliability}（{reliability_note}）",
        f"- 出典：SAT 大蔵経テキストデータベース {url}（{fetched} 取得）、{lines[0][0]}–{lines[-1][0]}",
        *([f"- 範囲：{scope_note}"] if scope_note else []),
        "- 行頭は大正新脩大藏經の行 ID（番号_.巻.頁 段 行）。引用の根拠に使う。",
        "",
    ]
    body_lines = (f"{line_id} {body}" if line_id else body for line_id, body in lines if body)  # 行 ID の無い行は削った印
    return "\n".join([*head, *body_lines]) + "\n"
