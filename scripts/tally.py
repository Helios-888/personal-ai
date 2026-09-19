"""基準値の採点を条件へ戻して集計し、tally.md を書く（Phase 3 手順 6 の 5）。

設計は docs/specs/2026-09-18-phase3-design.md「採点の段取り」5 と「指標」。

使い方（llm01 のリポジトリ直下で）:
  .venv/bin/python scripts/tally.py --scoring evaluations/kukai/scoring/2026-09-19-baseline
  → <scoring>/tally.md（既にあれば何も書かずに止まる）

読むもの（<scoring> の中）:
  scores.jsonl              一次採点（Opus 5）の区分と書名の抜き出し
  astra/result-*.txt        GPT-6 Astra の区分
  titles/items.jsonl        「」の補助抜き出しの項目 → 回答
  titles/result-claude.txt  補助抜き出しの判定（一次採点と組にする）
  titles/result-astra.txt   補助抜き出しの判定（Astra の区分と組にする）
  titles.yaml               自分の作として挙げた書名の区分
  key.jsonl                 呼び名 → 条件・回・元の記録の行（ほかをすべて読んで確かめたあと、最後に開く）
ほかに、凍結した questions.yaml・rubric.yaml（照合して読む）と、key.jsonl の見出しが指す 2 記録の answers.jsonl
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
    CONDITIONS,
    LABELS,
    check_complete,
    merge_titles,
    own_titles_from_scores,
    own_titles_from_supplement,
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
GRADERS = (  # （名前, 区分のファイル, 組にする補助抜き出しの判定）。1 人目が正式な値
    ("Opus 5", "scores.jsonl", "titles/result-claude.txt"),
    ("GPT-6 Astra", "astra/result-*.txt", "titles/result-astra.txt"),
)
CODE_FILES = ("scripts/tally.py", "scripts/lib/tally.py", "scripts/lib/tally_report.py")  # 集計の版も履歴からたどれるように
NOTES = [
    "採点者 Opus 5：一次採点（`scores.jsonl`、この会話を見ていない Claude Opus 5 のサブエージェント 4 体）。"
    "書名は一次採点の抜き出しに、「」の補助抜き出しの Claude の判定を合わせる",
    "採点者 GPT-6 Astra：`astra/result-*.txt`（ChatGPT の画面で利用者が依頼）。"
    "書名は一次採点の抜き出しに、補助抜き出しの Astra の判定を合わせる",
]


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="基準値の採点を条件へ戻して集計し、tally.md を書く")
    parser.add_argument("--scoring", required=True, help="採点のフォルダ（リポジトリ直下から）")
    parser.add_argument("--questions", default=DEFAULT_QUESTIONS, help="凍結した評価セットのパス")
    parser.add_argument("--rubric", default=DEFAULT_RUBRIC, help="凍結した採点基準のパス")
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
    items = inputs.text(scoring / "titles/items.jsonl")
    primary_titles = own_titles_from_scores(inputs.text(scoring / GRADERS[0][1]))
    graded = [(name, read_grader(scoring, pattern, inputs),
               merge_titles(primary_titles, own_titles_from_supplement(inputs.text(scoring / supplement), items)))
              for name, pattern, supplement in GRADERS]

    header, entries = read_key(inputs.text(scoring / KEY_FILE))  # ほかをすべて読んで確かめてから開く
    check_complete(entries, kinds, header["settings"]["repeats"])
    check_answers(header, root, inputs)
    commit = check_committed(root, [*inputs.read, *CODE_FILES], runner)
    tallies = [tally(name, entries, labels, kinds, own, categories) for name, labels, own in graded]

    report = render_report(header, tallies, NOTES, sorted(inputs.read.items()), commit)  # 組み上げてから書く
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


def check_answers(header: dict, root: Path, inputs: Inputs) -> None:
    """key.jsonl の見出しが指す 2 記録の answers.jsonl が、束を作ったときと同じか（行番号でたどれるか）。"""
    for condition in CONDITIONS:
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
