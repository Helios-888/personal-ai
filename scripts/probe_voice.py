"""agent.yaml のシステムプロンプトで質問一覧を llama-swap に 1 問ずつ投げ、記録を Markdown で残す。

Phase 2（声の設計、RAG なし）の確認用。Phase 3 の評価実行（run-eval）の土台にする。

使い方（llm01 のリポジトリ直下で）:
  .venv/bin/python scripts/probe_voice.py --label voice-draft2
  → evaluations/kukai/phase2-voice-check/<今日>-voice-draft2.md（既にあれば上書きしない）
"""
import argparse
import datetime as dt
import os
import sys
from pathlib import Path
from typing import Callable, Optional

if __package__ in (None, ""):  # スクリプトとして直接実行されたとき、リポジトリ直下を import 経路に加える
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import requests  # noqa: E402
import yaml  # noqa: E402

from scripts.lib.agent import load_agent  # noqa: E402
from scripts.lib.env import load_dotenv  # noqa: E402
from scripts.lib.llm_client import chat_completion  # noqa: E402
from scripts.lib.probe import ProbeRecord, TranscriptHeader, load_questions, render_transcript  # noqa: E402
from scripts.lib.prompt_builder import build_system_prompt, load_parts  # noqa: E402
from scripts.lib.tokens import TokenCount, count_tokens  # noqa: E402

DEFAULT_AGENT = "agents/kukai/agent.yaml"
DEFAULT_QUESTIONS = "evaluations/kukai/phase2-voice-check/questions.yaml"
DEFAULT_OUT_DIR = "evaluations/kukai/phase2-voice-check"
DEFAULT_LLAMA_SWAP_URL = "http://localhost:8080"
DEFAULT_LLAMA_SWAP_MODEL = "kukai"
DEFAULT_MAX_TOKENS = 800
DEFAULT_TEMPERATURE = 0.7
DEFAULT_CONDITIONS = "RAG なし、思考 OFF"


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="質問一覧を llama-swap に投げ、記録を Markdown で残す")
    parser.add_argument("--label", required=True, help="記録ファイル名の識別子（例: voice-draft2）")
    parser.add_argument("--title", default=None, help="記録の見出し（既定: 語り口の確認 <label>）")
    parser.add_argument("--agent", default=DEFAULT_AGENT, help="agent.yaml のパス（リポジトリ直下から）")
    parser.add_argument("--questions", default=DEFAULT_QUESTIONS, help="質問一覧 YAML のパス")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="記録の保存先フォルダ")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="1 問あたりの出力上限")
    parser.add_argument("--conditions", default=DEFAULT_CONDITIONS, help="見出しに記す条件")
    return parser.parse_args(argv)


def run(
    argv: list[str],
    *,
    root: Path,
    env: dict,
    chat: Callable,
    token_counter: Callable[..., TokenCount],
    today: str,
) -> int:
    args = parse_args(argv)
    out_path = root / args.out_dir / f"{today}-{args.label}.md"
    if out_path.exists():
        print(f"error: 記録が既にあります。上書きしません: {out_path.relative_to(root)}", file=sys.stderr)
        return 2
    try:
        agent = load_agent(root, args.agent)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    questions = load_questions(root / args.questions)
    system_prompt = build_system_prompt(load_parts(root, agent["system_prompt"]["parts"]))
    base_url = env.get("LLAMA_SWAP_URL", DEFAULT_LLAMA_SWAP_URL)
    model = env.get("LLAMA_SWAP_MODEL", DEFAULT_LLAMA_SWAP_MODEL)
    temperature = float((agent.get("params") or {}).get("temperature", DEFAULT_TEMPERATURE))

    tokens = token_counter(system_prompt, base_url=base_url, model=model)
    exactness = "exact" if tokens.exact else "estimated"
    print(f"system prompt: {tokens.count} tokens ({exactness}); {len(questions)} questions -> {model} at {base_url}")
    if not tokens.exact:
        print("warning: token count is an estimate (llama-swap tokenize was unreachable)", file=sys.stderr)

    records: list[ProbeRecord] = []
    for question in questions:
        result = chat(
            base_url,
            model,
            system=system_prompt,
            user=question.text,
            temperature=temperature,
            max_tokens=args.max_tokens,
        )
        records = [*records, ProbeRecord(question, result)]
        print(
            f"[{question.kind}] {result.elapsed_seconds:.1f} 秒、{len(result.content.strip())} 字、"
            f"finish={result.finish_reason}"
        )

    header = TranscriptHeader(
        title=args.title or f"語り口の確認 {args.label}",
        date=today,
        conditions=args.conditions,
        temperature=temperature,
        max_tokens=args.max_tokens,
        system_tokens=tokens.count,
        tokens_exact=tokens.exact,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_transcript(header, records), encoding="utf-8")
    print(f"saved: {out_path.relative_to(root)}")
    return 0


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
