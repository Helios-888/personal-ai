"""Phase 5：付け足し・書き換えの判定（faithfulness.yaml）の束を作る（使い捨て）。盲検でない回として行う。

回答ごとに、採点した本文（pack/ の採点票と同じ、引用の番号を除いたもの）と、その回答で検索された箇所（両条件にある。数は違う）を載せる。
すべての回答に、後の条件（K2）を取ったときの常時層（agent.yaml の parts を連結したもの、指紋を記録と照合）を与えた内容として見せる。
Phase 4 は 2 条件の常時層が僅かに違うとき、後の条件のものを全回答に見せた。Phase 5 も同じにする（K1 の回答には、
K1 の常時層に無く K2 の常時層にある事柄が付け足しに数えられない向きの誤差が残る。誤差の向きを notes.md に記す）。
条件名・記録の場所は束に書かない。対応表（key.jsonl）はこのスクリプトの中でだけ読み、画面に出さない。
使い方（リポジトリ直下で）: python - <scoring フォルダ> <出力フォルダ>   （出力フォルダが既にあれば止まる）
"""
import hashlib
import json
import re
import sys
from pathlib import Path

import yaml

sys.path.insert(0, ".")
from scripts.lib.agent import load_agent  # noqa: E402
from scripts.lib.prompt_builder import build_system_prompt, load_parts  # noqa: E402

GROUPS = {
    1: [f"F{n:02d}" for n in range(1, 9)],
    2: [f"F{n:02d}" for n in range(9, 16)],
    3: [f"A{n:02d}" for n in range(1, 6)] + ["H01", "H02"],
    4: [f"T{n:02d}" for n in range(1, 11)],
}
FENCE = "~~~~~~"
HIDDEN = ("K1", "K2", "runs", "baseline", "key.jsonl", "maxtok")  # 資料本文は与えた内容なので書き換えない（年譜の冒頭に設計書のパスがある）


def guide(faith: dict) -> str:
    j = faith["judgements"]
    return "\n".join([
        "# 付け足し・書き換えの判定の手引き",
        "",
        "空海として一人称で答える AI の回答を 1 件ずつ読み、AI に「与えた内容」と見比べて、次の 2 つを判定する。",
        "正解かどうかの採点は済んでいる。ここで見るのは、与えた内容のとおりに答えているかだけである。",
        "",
        "## 与えた内容",
        "",
        "- `01-always-on.md`：すべての回答に共通して、AI に常に与えていた内容（指示・語り口・目録・原則など）。",
        "- 各回答の下に「この回答で検索された箇所」がある。その回答のときに AI に渡した資料の本文である（箇所の数は回答によって違う）。"
        "この欄が「無し」の回答には、`01-always-on.md` だけを与えていた。",
        "- 問いの文。問いの文に書かれている事柄は、付け足しに数えない。",
        "",
        "## 判定",
        "",
        "### added（付け足し）",
        "",
        f"- 問い：{j['added']['question']}",
        f"- 事実とは：{j['added']['facts']}",
        f"- {j['added']['asserted']}",
        "- 数えないもの：",
        *[f"  - {item}" for item in j["added"]["not_counted"]],
        "",
        "### altered（書き換え）",
        "",
        f"- 問い：{j['altered']['question']}",
        "- 例：",
        *[f"  - {item}" for item in j["altered"]["examples"]],
        "",
        "## 決まり",
        "",
        "- 根拠は、与えた内容（`01-always-on.md` と、その回答で検索された箇所）と問いの文だけとする。",
        "- 事実が史実として正しいかどうかは判定に使わない。与えた内容に無ければ、正しい事実でも付け足しとする。記憶している知識で与えた内容を補わない。",
        "- 回答ごとに独立に判定する。ほかの回答と見比べて判定を変えない。",
        "- 読んでよいのはこのフォルダのファイルだけとする。ほかのフォルダやファイルを開かない、検索しない。",
        "",
        "## 出力",
        "",
        "1 回答 1 行の JSON で書く。足した事実・変えた事実は、回答の文言のまま書き出す。無ければ空の一覧 `[]` とする。",
        "",
        "```json",
        faith["output"],
        "```",
        "",
    ])


def fenced(text: str) -> list[str]:
    assert FENCE not in text
    return [FENCE, text, FENCE]


def passages(row: dict) -> list[str]:
    source = row["sources"][0]
    lines = []
    for n, (doc, meta) in enumerate(zip(source["document"], source["metadata"]), start=1):
        lines += ["", f"##### 箇所 {n}（資料：{meta['name']}）", "", *fenced(doc)]
    return lines


def main(scoring: Path, out: Path) -> None:
    faith = yaml.safe_load(Path("evaluations/kukai/faithfulness.yaml").read_text(encoding="utf-8"))
    key_rows = [json.loads(l) for l in (scoring / "key.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    header, entries = key_rows[0], {r["answer_id"]: r for r in key_rows[1:]}
    records = {
        condition: (Path(info["path"]) / "answers.jsonl").read_text(encoding="utf-8").split("\n")
        for condition, info in header["sources"].items()
    }
    tested = json.loads(records["K2"][0])  # 試している側の条件。常時層はこちらを全回答に見せる
    agent = load_agent(Path("."), tested["agent"])
    always_on = build_system_prompt(load_parts(Path("."), agent["system_prompt"]["parts"]))
    assert hashlib.sha256(always_on.encode("utf-8")).hexdigest() == tested["system_prompt_sha256"], "常時層の指紋が合わない"

    out.mkdir(parents=True)
    counts = {}
    for group, question_ids in GROUPS.items():
        folder = out / f"f{group}"
        folder.mkdir()
        files = {"00-guide.md": guide(faith), "01-always-on.md": always_on}
        for qid in question_ids:
            sheet = (scoring / "pack" / f"{qid}.md").read_text(encoding="utf-8")
            question = sheet.split("\n## 問い\n\n")[1].split("\n\n## ")[0]
            answers = re.findall(rf"^### ({qid}-[a-z])\n\n{FENCE}\n(.*?)\n{FENCE}$", sheet, flags=re.S | re.M)
            assert len(answers) == 6, (qid, len(answers))
            lines = [f"# {qid}", "", "## 問い", "", question, "", "## 回答"]
            for answer_id, content in answers:
                entry = entries[answer_id]
                row = json.loads(records[entry["condition"]][entry["line"] - 1])
                assert (row["question_id"], row["repeat"]) == (qid, entry["repeat"])
                lines += ["", f"### {answer_id}", "", *fenced(content), "", f"#### この回答で検索された箇所"]
                lines += passages(row) if row.get("sources") else ["", "無し"]
            files[f"{qid}.md"] = "\n".join(lines) + "\n"
            counts[qid] = len(answers)
        for name, text in files.items():
            leaks = [w for w in HIDDEN if w in text]
            assert not leaks, (name, leaks)
            with open(folder / name, "x", encoding="utf-8", newline="\n") as f:
                f.write(text)
    print("回答", sum(counts.values()), "問い", len(counts), "常時層", len(always_on), "字")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
