"""埋め込みモデルの選定（Phase 4 設計書 手順 3）。使い捨て。Open WebUI には触れない。

knowledge/kukai の 9 点を、Open WebUI v0.11.3 のいまの設定に近い形で区切り（見出しの行を除き、1,000 字・重なり 100 字の
RecursiveCharacterTextSplitter）、練習問題（devset.yaml）ごとに近い箇所を並べ、正解の行を含む箇所が何位に来るかを数える。
選び方は README.md（走らせる前に書いた）。リポジトリ直下から、埋め込み用の別の環境で実行する:

  HF_HOME=.cache-hf .venv-embed/bin/python evaluations/kukai/preflight/2026-09-19-embedding/compare.py
"""
import re
import sys
import time
from pathlib import Path

import yaml
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
KNOWLEDGE = ROOT / "knowledge" / "kukai"
LINE_ID = re.compile(r"T\d{4}[A-Z]?_?\.\d{2}\.\d{4}[abc]\d{2}")
MODELS = (  # （名前, モデル, 問いの前置き, 本文の前置き）。ruri は前置きありの版も参考に測る
    ("bge-m3", "BAAI/bge-m3", "", ""),
    ("ruri-v3-310m", "cl-nagoya/ruri-v3-310m", "", ""),
    ("ruri-v3-310m（前置きあり、参考）", "cl-nagoya/ruri-v3-310m", "検索クエリ: ", "検索文書: "),
)


def chunks() -> list[tuple[str, set[str]]]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    found = []
    for path in sorted(KNOWLEDGE.glob("*/*.md")):
        if path.parent.name not in ("primary", "later-attribution"):
            continue
        body = "\n".join(line for line in path.read_text(encoding="utf-8").splitlines() if not line.startswith("# "))
        for text in splitter.split_text(body):
            found.append((text, set(LINE_ID.findall(text))))
    return found


def main() -> int:
    questions = yaml.safe_load((HERE / "devset.yaml").read_text(encoding="utf-8"))["questions"]
    pieces = chunks()
    lines = [f"箇所の数：{len(pieces)}（1,000 字・重なり 100 字）", ""]
    lines += ["| モデル | 1 位に命中 | 上位 3 件に命中 | 上位 10 件に命中 | 各問の順位 | 所要 |", "|---|---|---|---|---|---|"]
    for name, model_id, query_prefix, doc_prefix in MODELS:
        started = time.monotonic()
        model = SentenceTransformer(model_id, device="cpu")
        docs = model.encode([doc_prefix + text for text, _ in pieces], normalize_embeddings=True, batch_size=8)
        ranks = []
        for q in questions:
            query = model.encode([query_prefix + q["text"]], normalize_embeddings=True)[0]
            order = (docs @ query).argsort()[::-1]
            rank = next(i + 1 for i, idx in enumerate(order) if set(q["target"]) & pieces[idx][1])
            ranks.append((q["id"], rank))
        hit = lambda k: sum(1 for _, r in ranks if r <= k)  # noqa: E731
        detail = "、".join(f"{qid} {r}" for qid, r in ranks)
        lines.append(f"| {name} | {hit(1)}/{len(ranks)} | {hit(3)}/{len(ranks)} | {hit(10)}/{len(ranks)} | {detail} | "
                     f"{time.monotonic() - started:.0f} 秒 |")
        print(lines[-1], flush=True)
    out = HERE / "results.md"
    if out.exists():
        raise SystemExit("results.md は既にあります（上書きしない）")
    out.write_text("# 埋め込みモデルの比較の結果\n\n" + "\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
