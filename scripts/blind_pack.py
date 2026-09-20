"""条件ごとの記録から、条件名を伏せた採点用の束と、条件へ戻す対応表を作る（Phase 3 手順 6 の 1、Phase 4 の採点）。

設計は docs/specs/2026-09-18-phase3-design.md「採点の段取り」と docs/specs/2026-09-19-phase4-design.md「採点」。

使い方（llm01 のリポジトリ直下で）。条件は「--run 条件=記録フォルダ」で 2 つ以上渡す:
  .venv/bin/python scripts/blind_pack.py \\
      --run B0=evaluations/kukai/baseline/2026-09-18-B0-maxtok2000 \\
      --run B1=evaluations/kukai/baseline/2026-09-19-B1-maxtok2000 \\
      --out evaluations/kukai/scoring/2026-09-19-baseline
  → <out>/pack/（00-guide.md と問いごとの採点票）
    <out>/key.jsonl（呼び名から条件・回・元の記録の行へ戻す対応表。採点が終わるまで開かない）
  採点者には pack/ の写しをリポジトリの外に置いて渡す（リポジトリの中だと key.jsonl や履歴に届く）。
  出力先が既にあれば何も書かずに止まる。途中で書けなくなったら、作りかけの出力先を消して止まる。
  並べ替えの種は OS の乱数から作り、key.jsonl に記録する。
  回答の本文は、引用の番号 [n] を前の空白ごと全回答から除いて載せ、除いた数を key.jsonl に残す（元の記録は変えない）。
  ほかの検索の印（<source>、SAT の行 ID、資料のファイル名）が残る回答があれば止まる。

  記録に付いた条件名（run_eval の --condition）と束での名前が違うときは、取り違えを防ぐため
  「--recorded 束での名前=記録での条件」で宣言する。run_eval の条件名は B0・B1・K1 しか無いので、
  同じ K1 で取った 2 つの記録を比べるときに要る（例：--run K2=evaluations/kukai/runs/… --recorded K2=K1）。
  宣言は key.jsonl の sources に recorded_condition として残り、記録での条件名も束から伏せる。
"""
import argparse
import hashlib
import re
import secrets
import shutil
import sys
from pathlib import Path
from typing import Callable, Optional

if __package__ in (None, ""):  # スクリプトとして直接実行されたとき、リポジトリ直下を import 経路に加える
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml  # noqa: E402

from scripts.lib.blinding import (  # noqa: E402
    CITATION,
    RunRecord,
    blind_groups,
    key_lines,
    load_scoring_questions,
    read_run,
    relabel_run,
    render_guide,
    render_sheet,
    strip_run_citations,
)
from scripts.lib.cli import LABEL_PATTERN, display  # noqa: E402
from scripts.lib.freeze import read_frozen_text, read_record, record_path  # noqa: E402

DEFAULT_QUESTIONS = "evaluations/kukai/questions.yaml"
DEFAULT_RUBRIC = "evaluations/kukai/rubric.yaml"
ANSWERS_FILE = "answers.jsonl"
PACK_DIR = "pack"
GUIDE_FILE = "00-guide.md"
KEY_FILE = "key.jsonl"
SAME_SETTINGS = ("questions_sha256", "route", "temperature", "max_tokens", "repeats")  # 比較の土俵。条件どうしで揃う欄
# 検索の印。資料ありの条件の回答にしか出ないので、束に残ると採点者に条件が分かる（Phase 4 設計書「採点」）。
# 引用の番号 [n] は除いてから確かめる（blinding.CITATION）。除けなかった形の番号と、除いた跡の空白はここで止める
RETRIEVAL_MARKERS = {
    "引用の番号の残り": re.compile(r"[\[［【]\s*[0-9０-９]+\s*(?:[,，、\-–]\s*[0-9０-９]+\s*)*[\]］】]"),
    "句読点の前の空白": re.compile(r"[^\S\n][。、]"),
    "<source> タグ": re.compile(r"<source"),
    "SAT の行 ID": re.compile(r"T\d{4}[A-Z]?_?\.\d{2}\.\d{4}[abc]\d{2}"),
    "資料のファイル名の区分": re.compile(r"(?<![a-z-])[a-z][a-z-]*__"),
}


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="条件ごとの記録から、条件名を伏せた採点用の束と対応表を作る")
    parser.add_argument("--run", action="append", required=True, metavar="条件=記録フォルダ",
                        help="条件名と記録フォルダ（リポジトリ直下から）。2 つ以上渡す（例 --run B1=... --run K1=...）")
    parser.add_argument("--recorded", action="append", metavar="条件=記録での条件",
                        help="記録に付いた条件名が束での名前と違うときに宣言する（例 --recorded K2=K1）")
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
    records = parse_runs(args.run)
    recorded = parse_recorded(args.recorded or [], records)
    loaded = {condition: load_run(root, folder, recorded.get(condition, condition), questions_sha256, condition)
              for condition, folder in records.items()}
    # 対応表の条件名は束での名前にそろえる（記録に付いた名ではない。--recorded で宣言したとき両者は違う）
    runs = {condition: relabel_run(strip_run_citations(record), condition)
            for condition, (record, _) in loaded.items()}
    removed = {condition: sum(a.citations_removed for a in run.answers) for condition, run in runs.items()}
    settings = same_settings(runs)
    check_committed_runs(runs)
    check_retrieval_markers(runs)

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
    folders = [Path(folder) for folder in records.values()]
    hidden = [*records, *recorded.values(), *(f.name for f in folders), *(f.parent.name for f in folders), KEY_FILE]
    check_hidden(files, [word for word in hidden if word])  # 最上位のフォルダは親の名が空になる

    meta = {
        "rubric_sha256": read_record(record_path(rubric_path))[0],
        "settings": settings,
        "sources": {
            condition: {
                "path": records[condition],
                **({"recorded_condition": recorded[condition]} if condition in recorded else {}),
                "answers_sha256": digest,
            }
            for condition, (_, digest) in loaded.items()
        },
        "citations": {"pattern": CITATION.pattern, "removed": removed},  # 束の本文と元の記録の違いはこれだけ
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
    print(f"除いた引用の番号: {'、'.join(f'{c} {n} 個' for c, n in removed.items())}")
    if recorded:
        print(f"記録での条件名: {'、'.join(f'{name} は {c}' for name, c in recorded.items())}")
    return 0


def parse_runs(values: list[str]) -> dict[str, str]:
    """「条件=記録フォルダ」の並びを、条件 → フォルダにする。条件名は束に伏せる語になるので、形を確かめる。"""
    records: dict[str, str] = {}
    for value in values:
        condition, separator, folder = value.partition("=")
        if not separator or not LABEL_PATTERN.fullmatch(condition) or not folder:
            raise ValueError(f"--run は「条件=記録フォルダ」の形で渡してください: {value}")
        if condition in records:
            raise ValueError(f"--run の条件 {condition} が重複しています")
        records[condition] = folder
    if len(records) < 2:
        raise ValueError("--run で記録を 2 つ以上渡してください（1 条件だけでは条件を伏せられない）")
    return records


def parse_recorded(values: list[str], records: dict[str, str]) -> dict[str, str]:
    """「束での名前=記録での条件」の並びを、束での名前 → 記録での条件にする。

    run_eval の条件名は B0・B1・K1 しか無い。同じ K1 で取った 2 つの記録（設定を変えた前後）を
    比べるときは、片方を束で別の名（K2）と呼ぶ。記録の書き換えはせず、ここで宣言する。
    """
    recorded: dict[str, str] = {}
    for value in values:
        name, separator, condition = value.partition("=")
        if not separator or not LABEL_PATTERN.fullmatch(name) or not LABEL_PATTERN.fullmatch(condition):
            raise ValueError(f"--recorded は「条件=記録での条件」の形で渡してください: {value}")
        if name in recorded:
            raise ValueError(f"--recorded の条件 {name} が重複しています")
        if name not in records:
            raise ValueError(f"--recorded の {name} は --run で渡していません")
        recorded[name] = condition
    return recorded


def load_run(root: Path, folder: str, condition: str, questions_sha256: str, name: str = "") -> tuple[RunRecord, str]:
    """記録を 1 回だけ読み、その同じバイト列から中身と指紋を得る。

    condition は記録に付いているはずの条件名（--recorded の宣言があればその値）、name は束での名前。
    """
    data = (root / folder / ANSWERS_FILE).read_bytes()
    try:
        record = read_run(data.decode("utf-8"))
    except ValueError as error:
        raise ValueError(f"{folder}/{ANSWERS_FILE}: {error}") from error
    actual = record.header.get("condition")
    if actual != condition:
        if name and name != condition:
            raise ValueError(f"--recorded {name}={condition} と宣言しましたが、記録（{folder}）の条件は {actual} です")
        raise ValueError(f"--run {condition}= に渡した記録（{folder}）の条件は {actual} です。{condition} の記録を渡してください"
                         f"（わざと別の名で呼ぶなら --recorded {condition}={actual} と宣言します）")
    if record.header.get("questions_sha256") != questions_sha256:
        raise ValueError(f"{folder} は、いまの questions.yaml とは別の評価セットで取った記録です")
    return record, hashlib.sha256(data).hexdigest()


def same_settings(runs: dict[str, RunRecord]) -> dict:
    """条件どうしで揃っているべき設定を確かめ、その値を返す（対応表の見出しに写す）。"""
    differing = [
        f"{name}（{', '.join(f'{c}={run.header.get(name)!r}' for c, run in runs.items())}）"
        for name in SAME_SETTINGS
        if len({repr(run.header.get(name)) for run in runs.values()}) != 1
    ]
    if differing:
        raise ValueError(f"記録どうしで設定が揃っていません: {'; '.join(differing)}")
    first = next(iter(runs.values()))
    return {name: first.header.get(name) for name in SAME_SETTINGS}


def check_committed_runs(runs: dict[str, RunRecord]) -> None:
    """未コミットの定義・コードで取った記録は束に入れない（run_eval の見張りの外で取った記録も弾く）。"""
    for condition, run in runs.items():
        dirty = run.header.get("git_dirty")
        if not isinstance(dirty, list):
            raise ValueError(f"{condition} の記録の見出しに git_dirty がありません（未コミットの変更の有無が分からない）")
        if dirty:
            raise ValueError(f"{condition} の記録は未コミットの定義・コードで取られています: {'、'.join(dirty)}")


def check_retrieval_markers(runs: dict[str, RunRecord]) -> None:
    """引用の番号を除いた後も検索の印が残る回答があれば止める。新しい種類の印は、扱いを決めてから束を作る。"""
    found = [
        f"{condition} の {answer.question_id} {answer.repeat} 回目（{name}）"
        for condition, run in runs.items()
        for answer in run.answers
        for name, pattern in RETRIEVAL_MARKERS.items()
        if pattern.search(answer.content)
    ]
    if found:
        shown = "、".join(found[:5]) + (f" ほか {len(found) - 5} 件" if len(found) > 5 else "")
        raise ValueError(
            f"採点者に条件が分かる検索の印が回答にあります（{shown}）。"
            "引用の番号 [n] のほかは除かないので、印の扱いを決めてから束を作ります"
        )


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
