"""agent.yaml のシステムプロンプトで質問一覧を llama-swap に 1 問ずつ投げ、記録を Markdown で残す。

Phase 2（声の設計、RAG なし）の確認用。Phase 3 の評価実行（run-eval）の土台にする。

使い方（llm01 のリポジトリ直下で）:
  .venv/bin/python scripts/probe_voice.py --label voice-draft2
  → evaluations/kukai/phase2-voice-check/<今日>-voice-draft2.md（既にあれば上書きしない）
途中で失敗したときは、それまでの回答を「途中で中断」と記して同じ場所に保存する。
"""
import argparse
import datetime as dt
import os
import re
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Optional

if __package__ in (None, ""):  # スクリプトとして直接実行されたとき、リポジトリ直下を import 経路に加える
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import requests  # noqa: E402
import yaml  # noqa: E402

from scripts.lib.agent import load_agent  # noqa: E402
from scripts.lib.env import load_dotenv  # noqa: E402
from scripts.lib.llm_client import ChatResult, chat_completion  # noqa: E402
from scripts.lib.probe import ProbeRecord, Question, TranscriptHeader, load_questions, render_transcript  # noqa: E402
from scripts.lib.prompt_builder import build_system_prompt, load_parts, resolve_inside  # noqa: E402
from scripts.lib.tokens import TokenCount, count_tokens  # noqa: E402

DEFAULT_AGENT = "agents/kukai/agent.yaml"
DEFAULT_QUESTIONS = "evaluations/kukai/phase2-voice-check/questions.yaml"
DEFAULT_OUT_DIR = "evaluations/kukai/phase2-voice-check"
DEFAULT_LLAMA_SWAP_URL = "http://localhost:8080"
DEFAULT_LLAMA_SWAP_MODEL = "kukai"
DEFAULT_MAX_TOKENS = 800
DEFAULT_TEMPERATURE = 0.7
DEFAULT_CONDITIONS = "RAG なし、思考 OFF"
THINKING_OFF_MARK = "思考 OFF"
LABEL_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")  # ファイル名に埋め込むので区切り文字を許さない
INPUT_ERRORS = (ValueError, OSError, yaml.YAMLError)  # 設定ファイルの不備はすべて終了コード 2


@dataclass(frozen=True)
class ProbeSetup:
    agent_path: str
    system_prompt: str
    questions: list[Question]
    base_url: str
    model: str
    temperature: float
    out_path: Path


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("1 以上の整数を指定してください")
    return number


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="質問一覧を llama-swap に投げ、記録を Markdown で残す")
    parser.add_argument("--label", required=True, help="記録ファイル名の識別子（英数字・ドット・ハイフン・下線）")
    parser.add_argument("--title", default=None, help="記録の見出し（既定: 語り口の確認 <label>）")
    parser.add_argument("--agent", default=DEFAULT_AGENT, help="agent.yaml のパス（リポジトリ直下から）")
    parser.add_argument("--questions", default=DEFAULT_QUESTIONS, help="質問一覧 YAML のパス（リポジトリ直下から）")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="記録の保存先フォルダ")
    parser.add_argument("--max-tokens", type=positive_int, default=DEFAULT_MAX_TOKENS, help="1 問あたりの出力上限")
    parser.add_argument("--conditions", default=DEFAULT_CONDITIONS, help="見出しに記す条件")
    return parser.parse_args(argv)


def run(
    argv: list[str],
    *,
    root: Path,
    env: dict,
    chat: Callable[..., ChatResult],
    token_counter: Callable[..., TokenCount],
    today: str,
) -> int:
    args = parse_args(argv)
    try:
        setup = prepare(args, root, env, today)
    except INPUT_ERRORS as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    tokens = token_counter(setup.system_prompt, base_url=setup.base_url, model=setup.model)
    exactness = "exact" if tokens.exact else "estimated"
    print(
        f"system prompt: {tokens.count} tokens ({exactness}); "
        f"{len(setup.questions)} questions -> {setup.model} at {setup.base_url}"
    )
    if not tokens.exact:
        print("warning: token count is an estimate (llama-swap tokenize was unreachable)", file=sys.stderr)

    header = TranscriptHeader(
        title=args.title or f"語り口の確認 {args.label}",
        date=today,
        conditions=args.conditions,
        temperature=setup.temperature,
        max_tokens=args.max_tokens,
        system_tokens=tokens.count,
        tokens_exact=tokens.exact,
        model=setup.model,
        base_url=setup.base_url,
        agent=setup.agent_path,
    )
    records = ask_all(setup, args.max_tokens, chat, header, root)
    if THINKING_OFF_MARK in header.conditions and any(r.result.reasoning for r in records):
        print(
            f"warning: 条件に「{THINKING_OFF_MARK}」と記していますが、応答に思考（reasoning_content）が含まれています",
            file=sys.stderr,
        )
    save_transcript(setup.out_path, render_transcript(header, records))
    print(f"saved: {display(setup.out_path, root)}")
    return 0


def prepare(args: argparse.Namespace, root: Path, env: dict, today: str) -> ProbeSetup:
    """入力を読み、送信前に確かめられることはすべてここで確かめる（LLM は呼ばない）。"""
    if not LABEL_PATTERN.match(args.label):
        raise ValueError("--label は英数字・ドット・ハイフン・下線のみです")
    out_path = root / args.out_dir / f"{today}-{args.label}.md"
    if out_path.exists():
        raise ValueError(f"記録が既にあります。上書きしません: {display(out_path, root)}")
    agent = load_agent(root, args.agent)
    questions = load_questions(resolve_inside(root, args.questions))
    system_prompt = build_system_prompt(load_parts(root, agent["system_prompt"]["parts"]))
    temperature = float((agent.get("params") or {}).get("temperature", DEFAULT_TEMPERATURE))
    return ProbeSetup(
        agent_path=args.agent,
        system_prompt=system_prompt,
        questions=questions,
        base_url=env.get("LLAMA_SWAP_URL", DEFAULT_LLAMA_SWAP_URL),
        model=env.get("LLAMA_SWAP_MODEL", DEFAULT_LLAMA_SWAP_MODEL),
        temperature=temperature,
        out_path=out_path,
    )


def ask_all(
    setup: ProbeSetup, max_tokens: int, chat: Callable[..., ChatResult], header: TranscriptHeader, root: Path
) -> list[ProbeRecord]:
    """1 問ずつ問う。途中で失敗したら、それまでの回答を「途中で中断」と記して保存してから失敗を伝える。"""
    records: list[ProbeRecord] = []
    try:
        for question in setup.questions:
            result = chat(
                setup.base_url,
                setup.model,
                system=setup.system_prompt,
                user=question.text,
                temperature=setup.temperature,
                max_tokens=max_tokens,
            )
            records = [*records, ProbeRecord(question, result)]
            print(
                f"[{question.kind}] {result.elapsed_seconds:.1f} 秒、{len(result.content.strip())} 字、"
                f"finish={result.finish_reason}"
            )
    except BaseException:
        if records:
            note = f"途中で中断（{len(records)}/{len(setup.questions)} 問まで）"
            save_transcript(setup.out_path, render_transcript(replace(header, note=note), records))
            print(f"saved (partial): {display(setup.out_path, root)}", file=sys.stderr)
        raise
    return records


def save_transcript(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x", encoding="utf-8") as file:  # 同名があれば失敗する（同時実行でも上書きしない）
        file.write(text)


def display(path: Path, root: Path) -> str:
    """表示用にリポジトリ相対へ。外のパスはそのまま返す（表示のために落とさない）。"""
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def main(argv: Optional[list[str]] = None, root: Optional[Path] = None) -> int:
    repo_root = Path(root) if root else Path(__file__).resolve().parents[1]
    env = {**load_dotenv(repo_root / ".env"), **os.environ}  # 環境変数が .env より優先
    try:
        return run(
            sys.argv[1:] if argv is None else argv,
            root=repo_root,
            env=env,
            chat=chat_completion,
            token_counter=count_tokens,
            today=dt.date.today().isoformat(),
        )
    except (requests.RequestException, OSError, yaml.YAMLError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
