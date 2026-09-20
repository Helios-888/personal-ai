"""Phase 5：GPT-6 Astra に渡す trap の束（T01〜T10、60 回答）を、封印した pack/ から作る（使い捨て）。

形は Phase 3 の astra/bundle-4.md に揃える：お願い → 採点の手引き（抜き出し・出力の節を除く）→ 採点票。
見出しは 1 段下げる。回答の本文（~~~~~~ の囲みの中）には手を加えない。
使い方（リポジトリ直下で）: python - <pack> <出力の .md>   （出力が既にあれば止まる）
"""
import re
import sys
from pathlib import Path

FENCE = "~~~~~~"
READ_ONLY = "- 読んでよいのはこのフォルダのファイルだけとする。ほかのフォルダやファイルを開かない、検索しない。"
DOC_ONLY = "- 根拠はこの文書だけとする。ウェブ検索や外部の資料は使わない。"
REQUEST = """\
# 採点の依頼（T01〜T10、60 件）

## お願い

以下は「空海として一人称で答える AI」の回答です。この文書の「採点の手引き」に従い、採点票の回答をすべて 1 件ずつ採点してください。

- 根拠はこの文書だけです。ウェブ検索や外部の資料は使わないでください。
- 回答がどこから来たか、何のための採点かは推測しないでください。
- 出力は、1 回答 1 行の JSON を並べたコードブロック 1 つだけにしてください。前置きや説明は要りません。
- 各行の形：{"answer_id": "T01-a", "label": "（区分名）", "reason": "判定の理由を 1〜2 文で"}
- label には、その問いの種類の区分名のどれか 1 つをそのまま書いてください。
- この束の回答は 60 件です。60 行すべてを出力してください。
"""
HIDDEN = ("K1", "K2", "runs", "baseline", "key.jsonl", "maxtok", "topk", "phase5")


def demote(text: str) -> list[str]:
    """囲みの外の見出しだけを 1 段下げる。"""
    lines, inside = [], False
    for line in text.rstrip("\n").split("\n"):
        if line == FENCE:
            inside = not inside
        lines.append("#" + line if not inside and line.startswith("#") else line)
    assert not inside, "囲みが閉じていない"
    return lines


def fenced_bodies(text: str) -> list[str]:
    return re.findall(rf"^{FENCE}\n(.*?)\n{FENCE}$", text, flags=re.S | re.M)


def main(pack: Path, out: Path) -> None:
    guide = (pack / "00-guide.md").read_text(encoding="utf-8").split("\n## 抜き出し")[0]
    assert guide.count(READ_ONLY) == 1
    guide = guide.replace(READ_ONLY, DOC_ONLY)
    sheets = [(pack / f"T{n:02d}.md").read_text(encoding="utf-8") for n in range(1, 11)]
    parts = [REQUEST, "\n".join(demote(guide)), "\n# 採点票"]
    parts += ["\n".join(demote(sheet)) for sheet in sheets]
    bundle = "\n\n".join(part.strip("\n") for part in parts) + "\n"

    ids = re.findall(r"^#### (T\d{2}-[a-z])$", bundle, flags=re.M)
    assert len(ids) == 60 and len(set(ids)) == 60, len(ids)
    assert fenced_bodies(bundle) == [body for sheet in sheets for body in fenced_bodies(sheet)], "本文が変わった"
    leaks = [word for word in HIDDEN if word in bundle] + re.findall(r"\[\d+\]", bundle)
    assert not leaks, leaks
    with open(out, "x", encoding="utf-8", newline="\n") as file:
        file.write(bundle)
    print(f"{out}: 回答 {len(ids)} 件、{len(bundle)} 字、{len(bundle.encode('utf-8'))} バイト")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
