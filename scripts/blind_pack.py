"""基準値の 2 記録（B0・B1）から、条件名を伏せた採点用の束と、条件へ戻す対応表を作る（Phase 3 手順 6 の 1）。

設計は docs/specs/2026-09-18-phase3-design.md「採点の段取り」。

使い方（llm01 のリポジトリ直下で）:
  .venv/bin/python scripts/blind_pack.py \\
      --b0 evaluations/kukai/baseline/2026-09-18-B0-maxtok2000 \\
      --b1 evaluations/kukai/baseline/2026-09-19-B1-maxtok2000 \\
      --out evaluations/kukai/scoring/2026-09-19-baseline
  → <out>/pack/（00-guide.md と問いごとの採点票）
    <out>/key.jsonl（呼び名から条件・回・元の記録の行へ戻す対応表。採点が終わるまで開かない）
  採点者には pack/ の写しをリポジトリの外に置いて渡す（リポジトリの中だと key.jsonl や履歴に届く）。
  出力先が既にあれば何も書かずに止まる。途中で書けなくなったら、作りかけの出力先を消して止まる。
  並べ替えの種は OS の乱数から作り、key.jsonl に記録する。
"""
import argparse
import hashlib
import secrets
import shutil
import sys
from pathlib import Path
from typing import Callable, Optional

if __package__ in (None, ""):  # スクリプトとして直接実行されたとき、リポジトリ直下を import 経路に加える
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml  # noqa: E402

from scripts.lib.blinding import (  # noqa: E402
    RunRecord,
    blind_groups,
    key_lines,
    load_scoring_questions,
    read_run,
    render_guide,
    render_sheet,
)
from scripts.lib.cli import LABEL_PATTERN, display  # noqa: E402
from scripts.lib.freeze import read_frozen_text, read_record, record_path  # noqa: E402

DEFAULT_QUESTIONS = "evaluations/kukai/questions.yaml"
DEFAULT_RUBRIC = "evaluations/kukai/rubric.yaml"
ANSWERS_FILE = "answers.jsonl"
PACK_DIR = "pack"
GUIDE_FILE = "00-guide.md"
KEY_FILE = "key.jsonl"
SAME_SETTINGS = ("questions_sha256", "route", "temperature", "max_tokens", "repeats")  # 比較の土俵。B0・B1 で揃う欄


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="基準値の 2 記録から、条件名を伏せた採点用の束と対応表を作る")
    parser.add_argument("--b0", required=True, help="B0 の記録フォルダ（リポジトリ直下から）")
    parser.add_argument("--b1", required=True, help="B1 の記録フォルダ（リポジトリ直下から）")
    parser.add_argument("--out", required=True, help="束と対応表を作るフォルダ（既にあれば止まる）")
    parser.add_argument("--questions", default=DEFAULT_QUESTIONS, help="凍結した評価セットのパス")
    parser.add_argument("--rubric", default=DEFAULT_RUBRIC, help="凍結した採点基準のパス")
    return parser.parse_args(argv)


def run(argv: list[str], root: Path, seed_source: Callable[[], int] = lambda: secrets.randbits(63)) -> int:
    args = parse_args(argv)
    try:
        return build_pack(args, Path(root), seed_source)
    except (ValueError, OSError, yaml.YAMLError) as error:  # FrozenFileError も ValueError の仲間
        print(f"error: {error}", file=sys.stderr)
        return 1


def build_pack(args: argparse.Namespace, root: Path, seed_source: Callable[[], int]) -> int:
    questions_path, rubric_path = root / args.questions, root / args.rubric
    questions_text = read_frozen_text(questions_path)
    rubric_text = read_frozen_text(rubric_path)
    questions_sha256 = read_record(record_path(questions_path))[0]
    records = {"B0": args.b0, "B1": args.b1}
    loaded = {condition: load_run(root, folder, condition, questions_sha256) for condition, folder in records.items()}
    runs = {condition: record for condition, (record, _) in loaded.items()}
    settings = same_settings(runs)

    seed = seed_source()
    questions = load_scoring_questions(questions_text, rubric_text)
    check_question_ids([q.id for q in questions])
    rubric = yaml.safe_load(rubric_text)
    labels = labels_for(rubric, [q.kind for q in questions])
    groups = blind_groups(questions, runs, seed)
    files = {
        GUIDE_FILE: render_guide(labels, rubric["extract"]),
        **{f"{g.question.id}.md": render_sheet(g, labels[g.question.kind]) for g in groups},
    }
    check_hidden(files, [*records, *(Path(folder).name for folder in records.values()), "baseline", KEY_FILE])

    meta = {
        "rubric_sha256": read_record(record_path(rubric_path))[0],
        "settings": settings,
        "sources": {
            condition: {"path": records[condition], "answers_sha256": digest}
            for condition, (_, digest) in loaded.items()
        },
        "python": sys.version.split()[0],  # random の並びは版によって変わりうる。再現の手がかりに残す
    }
    out = root / args.out
    try:
        out.mkdir(parents=True)
    except FileExistsError as error:
        raise ValueError(f"出力先が既にあります（上書きしない）: {display(out, root)}") from error
    write_outputs(out, files, key_lines(groups, seed, meta), root)
    answers = sum(len(g.answers) for g in groups)
    print(f"pack: {display(out / PACK_DIR, root)}（手引き 1 枚＋採点票 {len(groups)} 枚、回答 {answers} 件）")
    print(f"key:  {display(out / KEY_FILE, root)}（採点が終わるまで開かない）")
    return 0


def load_run(root: Path, folder: str, condition: str, questions_sha256: str) -> tuple[RunRecord, str]:
    """記録を 1 回だけ読み、その同じバイト列から中身と指紋を得る。"""
    data = (root / folder / ANSWERS_FILE).read_bytes()
    try:
        record = read_run(data.decode("utf-8"))
    except ValueError as error:
        raise ValueError(f"{folder}/{ANSWERS_FILE}: {error}") from error
    actual = record.header.get("condition")
    if actual != condition:
        raise ValueError(f"--{condition.lower()} に渡した記録（{folder}）の条件は {actual} です。{condition} の記録を渡してください")
    if record.header.get("questions_sha256") != questions_sha256:
        raise ValueError(f"{folder} は、いまの questions.yaml とは別の評価セットで取った記録です")
    return record, hashlib.sha256(data).hexdigest()


def same_settings(runs: dict[str, RunRecord]) -> dict:
    """B0・B1 で揃っているべき設定を確かめ、その値を返す（対応表の見出しに写す）。"""
    differing = [
        f"{name}（{', '.join(f'{c}={run.header.get(name)!r}' for c, run in runs.items())}）"
        for name in SAME_SETTINGS
        if len({repr(run.header.get(name)) for run in runs.values()}) != 1
    ]
    if differing:
        raise ValueError(f"記録どうしで設定が揃っていません: {'; '.join(differing)}")
    first = next(iter(runs.values()))
    return {name: first.header.get(name) for name in SAME_SETTINGS}


def check_question_ids(ids: list[str]) -> None:
    """問いの id はファイル名になるので、区切り文字を許さない。"""
    bad = [i for i in ids if not LABEL_PATTERN.fullmatch(i)]
    if bad:
        raise ValueError(f"ファイル名に使えない問いの id があります: {', '.join(bad)}")


def labels_for(rubric: dict, kinds: list[str]) -> dict[str, dict[str, str]]:
    """問いに現れる種類の区分だけを、現れた順に返す。"""
    labels = rubric.get("labels") or {}
    if not rubric.get("extract"):
        raise ValueError("rubric.yaml に抜き出し（extract）がありません")
    missing = sorted({kind for kind in kinds if kind not in labels})
    if missing:
        raise ValueError(f"rubric.yaml に区分が無い問いの種類があります: {', '.join(missing)}")
    return {kind: labels[kind] for kind in dict.fromkeys(kinds)}


def check_hidden(files: dict[str, str], hidden: list[str]) -> None:
    """採点者に渡すファイルに、条件名や記録の場所が紛れていれば、何も書かずに止める。"""
    leaks = [f"{name}: {word}" for name, text in files.items() for word in hidden if word in text]
    if leaks:
        raise ValueError(f"採点者に伏せる語が束に含まれています（{'; '.join(leaks)}）")


def write_outputs(out: Path, files: dict[str, str], key: list[str], root: Path) -> None:
    """束と対応表を書く。途中で失敗したら、この実行で作った出力先ごと消す（対応表の無い束を残さない）。"""
    try:
        (out / PACK_DIR).mkdir()
        for name, text in files.items():
            write_new(out / PACK_DIR / name, text)
        write_new(out / KEY_FILE, "\n".join(key) + "\n")
    except BaseException:
        shutil.rmtree(out, ignore_errors=True)
        print(f"作りかけの {display(out, root)} を消しました", file=sys.stderr)
        raise


def write_new(path: Path, text: str) -> None:
    with open(path, "x", encoding="utf-8", newline="\n") as file:  # 同名があれば失敗する（上書きしない）
        file.write(text)


def main(argv: Optional[list[str]] = None) -> int:
    return run(sys.argv[1:] if argv is None else argv, root=Path(__file__).resolve().parents[1])


if __name__ == "__main__":
    sys.exit(main())
