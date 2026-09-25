"""answers.jsonl の検索された箇所が、どのファイルのどの篇（「## ○」見出し）から来たかを数える（使い捨て）。

使い方（llm01 のリポジトリ直下で）:
  .venv/bin/python - <answers.jsonl> < retrieval_pieces.py
"""
import ast
import json
import sys
from pathlib import Path

import yaml

TARGETS = {
    "V01": ("primary__性霊集補闕抄.md", ("綜藝種智院式",)),
    "V02": ("primary__性霊集補闕抄.md", ("爲亡弟子智泉達嚫文",)),
    "V03": ("primary__性霊集.md", ("入山興", "山中有何樂")),
    "V04": ("primary__三教指帰.md", ("",)),  # 篇の見出しが無いので、ファイルのどこでもよい
    "V05": ("primary__性霊集補闕抄.md", ("高野山萬燈會願文",)),
}


def as_list(value):
    return ast.literal_eval(value) if isinstance(value, str) else value


def locate(text: str, chunk: str) -> int:
    lines = sorted((ln.strip() for ln in chunk.splitlines() if len(ln.strip()) >= 12), key=len, reverse=True)
    for line in lines[:5]:
        for start in (0, len(line) // 3):
            pos = text.find(line[start:start + 12])
            if pos >= 0:
                return pos
    return -1


def heading_at(text: str, pos: int) -> str:
    head = text.rfind("\n## ○", 0, pos + 1)
    if head < 0:
        return "(見出しの前)"
    end = text.find("\n", head + 1)
    return text[head + 5:end].strip()


def main(path: str) -> None:
    registered = yaml.safe_load(Path("agents/kukai/knowledge-registered.yaml").read_text(encoding="utf-8"))
    files = {f["file_id"]: f["path"] for k in registered["knowledge"] for f in k["files"]}
    texts = {}
    hits = {}
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        record = json.loads(raw)
        if "question_id" not in record:
            continue
        qid, rep = record["question_id"], record["repeat"]
        pieces = []
        for source in record.get("sources") or []:
            docs, metas, dists = (as_list(source.get(k)) for k in ("document", "metadata", "distances"))
            for doc, meta, dist in zip(docs, metas, dists):
                fpath = files.get(meta.get("file_id"), "?" + str(meta.get("file_id")))
                if fpath not in texts and not fpath.startswith("?"):
                    texts[fpath] = Path(fpath).read_text(encoding="utf-8")
                pos = locate(texts.get(fpath, ""), doc)
                name = Path(fpath).name
                heading = heading_at(texts[fpath], pos) if pos >= 0 else "(位置不明)"
                pieces.append((name, heading, round(float(dist), 3)))
        target_file, target_heads = TARGETS.get(qid, ("", ()))
        hit = any(n == target_file and any(h.startswith(t) for t in target_heads) for n, h, _ in pieces)
        hits.setdefault(qid, []).append(hit)
        print(f"## {qid} r{rep}  関係する篇が入った={hit}  箇所={len(pieces)}")
        for rank, (n, h, d) in enumerate(pieces, 1):
            print(f"  {rank:2d} {d:.3f} {n} | {h[:30]}")
    print()
    for qid, flags in hits.items():
        print(qid, "検索に入った回:", sum(flags), "/", len(flags))


if __name__ == "__main__":
    main(sys.argv[1])
