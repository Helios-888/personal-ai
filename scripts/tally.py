"""採点を条件へ戻して集計し、tally.md を書く（Phase 3 手順 6 の 5、Phase 4 手順 7）。

設計は docs/specs/2026-09-18-phase3-design.md「採点の段取り」5 と「指標」、docs/specs/2026-09-19-phase4-design.md「採点」。

使い方（llm01 のリポジトリ直下で）:
  .venv/bin/python scripts/tally.py --scoring evaluations/kukai/scoring/2026-09-19-phase4
  → <scoring>/tally.md（既にあれば何も書かずに止まる）

読むもの（<scoring> の中）:
  scores.jsonl              一次採点（Opus 5）の区分と書名の抜き出し
  astra/result-*.txt        GPT-6 Astra の区分。採点範囲は、その出力にある問いの種類（Phase 4 は trap の束だけ）
  titles.yaml               自分の作として挙げた書名の区分
  titles/items.jsonl        （Phase 3 だけ）「」の補助抜き出しの項目 → 回答。これがあれば次の 2 つも読む
  titles/result-claude.txt  （Phase 3 だけ）補助抜き出しの判定（一次採点と組にする）
  titles/result-astra.txt   （Phase 3 だけ）補助抜き出しの判定（Astra の区分と組にする）
  faithfulness.jsonl        （あれば）付け足し・書き換えの判定。一次採点の集計に付け、正確さの関門を判定する
  key.jsonl                 呼び名 → 条件・回・元の記録の行（ほかをすべて読んで確かめたあと、最後に開く）。
                            条件名は見出しの sources の順（基準値が先）
ほかに、凍結した questions.yaml・rubric.yaml（照合して読む）と、key.jsonl の見出しが指す記録の answers.jsonl
（指紋を照合する）を読む。入力と集計のコードに未コミットの変更があれば止まる（条件を戻す前の状態と、集計した版を
履歴に固定するため）。
"""
import argparse
import hashlib
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional

if __package__ in (None, ""):  # スクリプトとして直接実行されたとき、リポジトリ直下を import 経路に加える
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml  # noqa: E402

from scripts.lib.cli import display  # noqa: E402
from scripts.lib.freeze import read_frozen_text  # noqa: E402
from scripts.lib.git_state import git_state  # noqa: E402
from scripts.lib.tally import (  # noqa: E402
    CONTENT_QUESTIONS,
    LABELS,
    check_complete,
    key_conditions,
    merge_titles,
    own_titles_from_scores,
    own_titles_from_supplement,
    read_faithfulness,
    read_key,
    read_labels,
    read_title_categories,
    tally,
)
from scripts.lib.tally_report import render_report  # noqa: E402

DEFAULT_QUESTIONS = "evaluations/kukai/questions.yaml"
DEFAULT_RUBRIC = "evaluations/kukai/rubric.yaml"
OUT_FILE = "tally.md"
KEY_FILE = "key.jsonl"
ANSWERS_FILE = "answers.jsonl"
SUPPLEMENT_ITEMS = "titles/items.jsonl"  # Phase 3 だけ：「」の書名を補助抜き出しで拾い直した
FAITHFULNESS_FILE = "faithfulness.jsonl"
GRADERS = (  # （名前, 区分のファイル, 組にする補助抜き出しの判定）。1 人目が正式な値
    ("Opus 5", "scores.jsonl", "titles/result-claude.txt"),
    ("GPT-6 Astra", "astra/result-*.txt", "titles/result-astra.txt"),
)
CODE_FILES = ("scripts/tally.py", "scripts/lib/tally.py", "scripts/lib/tally_report.py")  # 集計の版も履歴からたどれるように
NOTES_SUPPLEMENT = [  # Phase 3
    "採点者 Opus 5：一次採点（`scores.jsonl`、この会話を見ていない Claude Opus 5 のサブエージェント 4 体）。"
    "書名は一次採点の抜き出しに、「」の補助抜き出しの Claude の判定を合わせる",
    "採点者 GPT-6 Astra：`astra/result-*.txt`（ChatGPT の画面で利用者が依頼）。"
    "書名は一次採点の抜き出しに、補助抜き出しの Astra の判定を合わせる",
]
NOTES = [  # Phase 4 から：抜き出しの手引きが「」の書名を含む
    "採点者 Opus 5：一次採点（`scores.jsonl`、この会話を見ていない Claude Opus 5 のサブエージェント 4 体）。"
    "書名は一次採点の抜き出し（「」で挙げた書名を含む）",
    "採点者 GPT-6 Astra：`astra/result-*.txt`（ChatGPT の画面で利用者が依頼）。採点範囲は宣言したもの（--second-scope）で、"
    "出力にある問いの種類と照合した。書名は一次採点の抜き出しを使う",
]
FAITHFULNESS_NOTE = ("付け足し・書き換え：`faithfulness.jsonl`（この会話を見ていない Claude Opus 5 のサブエージェントが 1 体ずつ判定。"
                     "盲検でない）")
NO_FAITHFULNESS_NOTES = {
    True: "付け足し・書き換えの判定：無し（Phase 3 には判定が無い）",
    False: "付け足し・書き換えの判定：無し（--no-faithfulness で集計した。正確さの関門は判定していない）",
}
SCOPES = {"all": None, "trap": frozenset({"trap"})}  # 2 人目の採点範囲。Phase 3 は全束、Phase 4 は trap の束だけ（設計書「採点」）
BASELINE_DIR = "baseline"  # 基準値の記録の置き場。1 つ目の条件はここから取る


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="採点を条件へ戻して集計し、tally.md を書く")
    parser.add_argument("--scoring", required=True, help="採点のフォルダ（リポジトリ直下から）")
    parser.add_argument("--questions", default=DEFAULT_QUESTIONS, help="凍結した評価セットのパス")
    parser.add_argument("--rubric", default=DEFAULT_RUBRIC, help="凍結した採点基準のパス")
    parser.add_argument("--second-scope", choices=sorted(SCOPES),
                        help="2 人目（Astra）の採点範囲。既定は Phase 3 の形（補助抜き出しがある）なら all、それ以外は trap")
    parser.add_argument("--no-faithfulness", action="store_true",
                        help="付け足し・書き換えの判定なしで集計する（検索箇所を記録できなかったときだけ。報告にその旨を書く）")
    return parser.parse_args(argv)


def run(argv: list[str], root: Path, runner: Callable = subprocess.run) -> int:
    args = parse_args(argv)
    try:
        return build(args, Path(root), runner)
    except (ValueError, KeyError, TypeError, OSError, yaml.YAMLError) as error:  # FrozenFileError も ValueError の仲間
        print(f"error: {error}", file=sys.stderr)
        return 1


class Inputs:
    """読んだファイルを覚えておき、報告に指紋を載せ、未コミットの変更を確かめる。"""

    def __init__(self, root: Path):
        self.root = root
        self.read: dict[str, str] = {}  # 相対パス → SHA-256

    def text(self, path: Path) -> str:
        if not path.is_file():
            raise ValueError(f"読むファイルがありません: {display(path, self.root)}")
        data = path.read_bytes()
        self.read[display(path, self.root)] = hashlib.sha256(data).hexdigest()
        return data.decode("utf-8-sig")

    def frozen(self, path: Path) -> str:
        text = read_frozen_text(path)  # 照合と読み込みを同じバイト列で行う。UTF-8 の往復でバイト列は変わらない
        self.read[display(path, self.root)] = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return text


def build(args: argparse.Namespace, root: Path, runner: Callable) -> int:
    scoring, inputs = root / args.scoring, Inputs(root)
    out = scoring / OUT_FILE
    if out.exists():
        raise ValueError(f"{display(out, root)} は既にあります（上書きしない）")
    kinds = {q["id"]: q["kind"] for q in yaml.safe_load(inputs.frozen(root / args.questions))["questions"]}
    check_rubric_labels(yaml.safe_load(inputs.frozen(root / args.rubric)))
    categories = read_title_categories(inputs.text(scoring / "titles.yaml"))
    supplement = (scoring / SUPPLEMENT_ITEMS).exists()
    items = inputs.text(scoring / SUPPLEMENT_ITEMS) if supplement else ""
    primary_titles = own_titles_from_scores(inputs.text(scoring / GRADERS[0][1]))
    graded = [(name, read_grader(scoring, pattern, inputs),
               merge_titles(primary_titles, own_titles_from_supplement(inputs.text(scoring / result), items))
               if supplement else primary_titles)
              for name, pattern, result in GRADERS]
    faithful = read_faithful(scoring, inputs, optional=supplement or args.no_faithfulness)
    if faithful is not None:
        check_content_questions(kinds)
    declared = SCOPES[args.second_scope or ("all" if supplement else "trap")]

    header, entries = read_key(inputs.text(scoring / KEY_FILE))  # ほかをすべて読んで確かめてから開く
    conditions = key_conditions(header)
    check_baseline_first(header, conditions)
    check_complete(entries, kinds, header["settings"]["repeats"], conditions)
    check_answers(header, conditions, root, inputs)
    commit = check_committed(root, [*inputs.read, *CODE_FILES], runner)
    tallies = [
        tally(name, entries, labels, kinds, own, categories, conditions,
              scope=None if i == 0 else checked_scope(name, declared, labels, entries, kinds),  # 一次採点はいつも全問
              faithful=faithful if i == 0 else None)
        for i, (name, labels, own) in enumerate(graded)
    ]

    notes = [*(NOTES_SUPPLEMENT if supplement else NOTES),
             FAITHFULNESS_NOTE if faithful is not None else NO_FAITHFULNESS_NOTES[supplement]]
    report = render_report(header, tallies, notes, sorted(inputs.read.items()), commit)  # 組み上げてから書く
    with open(out, "x", encoding="utf-8", newline="\n") as file:  # 同名があれば失敗する（上書きしない）
        file.write(report)
    print(f"tally: {display(out, root)}（回答 {len(entries)} 件、採点者 {len(tallies)} 人）")
    return 0


def read_grader(scoring: Path, pattern: str, inputs: Inputs) -> dict[str, str]:
    files = sorted(scoring.glob(pattern))
    if not files:
        raise ValueError(f"採点者の出力がありません: {display(scoring, inputs.root)}/{pattern}")
    return read_labels((f.name, inputs.text(f)) for f in files)


def check_rubric_labels(rubric: dict) -> None:
    """集計の区分（tally.LABELS）が、凍結した rubric.yaml の区分と同じか。"""
    found = {kind: tuple(labels) for kind, labels in (rubric.get("labels") or {}).items()}
    if found != LABELS:
        raise ValueError(f"rubric.yaml の区分が集計の区分と違います: {found}")


def read_faithful(scoring: Path, inputs: Inputs, optional: bool) -> Optional[dict]:
    """付け足し・書き換えの判定。Phase 4 の形では要る（無いと関門が黙って消える）。"""
    path = scoring / FAITHFULNESS_FILE
    if path.exists():
        return read_faithfulness(inputs.text(path))
    if optional:
        return None
    raise ValueError(f"{display(path, inputs.root)} がありません（正確さの関門に要る。"
                     "検索箇所を記録できず判定できなかったときだけ --no-faithfulness で集計する）")


def check_content_questions(kinds: dict[str, str]) -> None:
    """関門の「7 問以上」の分母になる内容 10 問が、評価セットに factual としてそろっているか。"""
    missing = sorted(q for q in CONTENT_QUESTIONS if kinds.get(q) != "factual")
    if missing:
        raise ValueError(f"経典の内容を問う 10 問のうち、評価セットに factual として無い問いがあります: {', '.join(missing)}")


def check_baseline_first(header: dict, conditions: tuple[str, ...]) -> None:
    """1 つ目の条件が基準値（baseline/ の記録）か。「下がらず」と天井は後の条件で判定するので、順が逆だと誤る。"""
    path = (header["sources"].get(conditions[0]) or {}).get("path", "") if header.get("sources") else ""
    if BASELINE_DIR not in Path(path).parts:
        raise ValueError(f"1 つ目の条件 {conditions[0]} の記録が基準値（{BASELINE_DIR}/）ではありません: {path}"
                         "（blind_pack の --run は基準値を先に渡す）")


def checked_scope(name: str, declared: Optional[frozenset], labels: dict[str, str], entries: dict,
                  kinds: dict[str, str]) -> Optional[frozenset]:
    """2 人目の出力にある問いの種類が、宣言した採点範囲と同じか。結果ファイルの欠けを「範囲が狭い」と取り違えない。"""
    found = frozenset(kinds[entries[a].question_id] for a in labels if a in entries and entries[a].question_id in kinds)
    expected = frozenset(kinds.values()) if declared is None else declared
    if found != expected:
        raise ValueError(f"{name} の採点範囲が宣言（{'・'.join(sorted(expected))}）と違います"
                         f"（出力にある種類：{'・'.join(sorted(found)) or 'なし'}）")
    return declared


def check_answers(header: dict, conditions: tuple[str, ...], root: Path, inputs: Inputs) -> None:
    """key.jsonl の見出しが指す記録の answers.jsonl が、束を作ったときと同じか（行番号でたどれるか）。"""
    for condition in conditions:
        source = header["sources"][condition]
        path = root / source["path"] / ANSWERS_FILE
        inputs.text(path)
        actual = inputs.read[display(path, root)]
        if actual != source["answers_sha256"]:
            raise ValueError(f"{display(path, root)} が束を作ったときと違います（{actual[:12]}…）")


def check_committed(root: Path, paths: list[str], runner: Callable) -> str:
    state = git_state(root, paths, runner)
    if state.dirty:
        raise ValueError(f"入力に未コミットの変更があります（先にコミットする）: {'; '.join(state.dirty)}")
    return state.commit


def main(argv: Optional[list[str]] = None) -> int:
    return run(sys.argv[1:] if argv is None else argv, root=Path(__file__).resolve().parents[1])


if __name__ == "__main__":
    sys.exit(main())
