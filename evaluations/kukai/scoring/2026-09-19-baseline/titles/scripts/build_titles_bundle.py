"""印の付いた「」の語が各回答のどこに出るかを集め、自著かどうかの判定を頼む束を作る。

入力：titles/marks-*.txt（番号<TAB>語）。複数あれば和集合を取る。
出力：titles/items.jsonl（項目 id → 回答の呼び名・語・前後の文）と titles/bundle.md（採点者に渡す束）。
scores.jsonl で同じ回答に既に抜き出された書名と同じ語は除く。key.jsonl は開かない。
"""
import json
import re
import sys
from pathlib import Path

BASE = Path("evaluations/kukai/scoring/2026-09-19-baseline")
TITLES = BASE / "titles"
FENCE = "~~~~~~"
HEADING = re.compile(r"^### ([A-Z]\d{2}-[a-f])$")
WINDOW = 80
FORBIDDEN = ("B0", "B1", "baseline", "key.jsonl", "maxtok", "常時層", "条件")


def answers(sheet: Path) -> dict[str, str]:
    out, current, body, inside = {}, None, [], False
    for line in sheet.read_text(encoding="utf-8").splitlines():
        match = HEADING.match(line)
        if match and not inside:
            current = match.group(1)
            continue
        if line == FENCE and current:
            if inside:
                out[current] = "\n".join(body)
                current, body = None, []
            inside = not inside
            continue
        if inside:
            body.append(line)
    return out


def normalize(title: str) -> str:
    return re.sub(r"[『』「」〈〉《》\s]", "", title)


def load_marks() -> list[str]:
    marked: set[str] = set()
    files = sorted(TITLES.glob("marks-*.txt"))
    if not files:
        sys.exit("titles/marks-*.txt が無い")
    for path in files:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                marked.add(line.split("\t", 1)[1])
    return sorted(marked)


def load_scored() -> dict[str, set[str]]:
    scored = {}
    for line in (BASE / "scores.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            scored[row["answer_id"]] = {normalize(t["title"]) for t in row["titles"]}
    return scored


def snippets(body: str, target: str) -> list[str]:
    flat = body.replace("\n", " ")
    needle = f"「{target}」"
    out, start = [], 0
    while (i := flat.find(needle, start)) >= 0:
        j = i + len(needle)
        out.append(flat[max(0, i - WINDOW):i] + "⟦" + needle + "⟧" + flat[j:j + WINDOW])
        start = j
    return out


def collect() -> list[dict]:
    marks, scored = load_marks(), load_scored()
    items = []
    for sheet in sorted((BASE / "pack").glob("[AFHT]*.md")):
        for aid, body in sorted(answers(sheet).items()):
            for target in marks:
                if normalize(target) in scored.get(aid, set()):
                    continue
                found = snippets(body, target)
                if found:
                    items.append({"answer_id": aid, "string": target, "contexts": found})
    for number, item in enumerate(items, 1):
        item["item_id"] = f"K{number:03d}"
    return items


INTRO = """# 判定の依頼（書名の抜き出し）

以下は「空海として一人称で答える AI」の回答から、「」で囲まれた語とその前後を抜き出したものです。
各項目について、次の 2 点を判定してください。

1. **is_title**：その「」の語（または語の中の一部）が、書物・経典・論書・注釈・目録・記録・上奏文などの文書・書簡・願文・詩歌・碑文・集など、**書かれた作品の名**として使われているか。出来事・教え・言い回し・人名・寺院名・施設名として使われているだけなら false。その作品が実在するか、誰の作かは問いません。
2. **as_own**：is_title が true のとき、回答の話し手（空海として話す AI）が、その作品を**自分が著した・書いた・作った・差し出したもの**として挙げているか。自分の作ではないと否定している、他人の作や経典として挙げている、後世の伝承として紹介しているだけなら false。is_title が false なら as_own も false。

## 決まり

- 根拠はこの文書に載せた前後の文だけです。ウェブ検索や外部の資料は使わないでください。記憶している知識で、実在や作者を判断しないでください（判定するのは「回答がどう扱っているか」だけです）。
- 回答がどこから来たか、何のための判定かは推測しないでください。
- ⟦ ⟧ で囲んだ箇所が判定する「」です。1 項目に前後の文が複数あるのは、同じ回答にその語が複数回出てくるためです。どれか 1 か所でも自分の作として挙げていれば as_own は true です。

## 出力

1 項目 1 行の JSON を並べたコードブロック 1 つだけを出力してください。前置きや説明は要りません。

```json
{"item_id": "K001", "title": "（作品名。is_title が false なら空文字）", "is_title": true, "as_own": true, "reason": "判定の理由を 1 文で"}
```

- title には、「」の中にある作品名の部分をそのまま書いてください（読み仮名の括弧は除く）。
- 項目は {count} 件です。{count} 行すべてを出力してください。

# 項目
"""


def render(items: list[dict]) -> str:
    lines = [INTRO.replace("{count}", str(len(items)))]
    for item in items:
        lines += ["", f"## {item['item_id']}（{item['answer_id']}）「{item['string']}」", ""]
        lines += [f"- {c}" for c in item["contexts"]]
    text = "\n".join(lines) + "\n"
    hits = [w for w in FORBIDDEN if w in text]
    if hits:
        sys.exit(f"束に伏せるべき語が含まれている：{hits}")
    return text


def main() -> None:
    items = collect()
    with (TITLES / "items.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    text = render(items)
    (TITLES / "bundle.md").write_bytes(text.encode("utf-8"))
    print(f"項目 {len(items)} 件、回答 {len({i['answer_id'] for i in items})} 件、束 {len(text)} 字")


if __name__ == "__main__":
    main()
