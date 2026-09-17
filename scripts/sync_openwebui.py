"""agents/<agent>/agent.yaml を Open WebUI のカスタムモデルとして登録・更新する。

使い方（リポジトリ直下で）:
  .venv/bin/python scripts/sync_openwebui.py --dry-run   # 差分とトークン数だけ表示
  .venv/bin/python scripts/sync_openwebui.py             # 登録・更新
"""
import argparse
import os
import sys
from pathlib import Path
from typing import Callable

if __package__ in (None, ""):  # スクリプトとして直接実行されたとき、リポジトリ直下を import 経路に加える
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml  # noqa: E402

from scripts.lib.env import load_dotenv  # noqa: E402
from scripts.lib.openwebui_client import OpenWebUIClient  # noqa: E402
from scripts.lib.prompt_builder import build_system_prompt, is_over_budget, load_parts  # noqa: E402
from scripts.lib.sync import apply_plan, build_model_form, plan_sync  # noqa: E402
from scripts.lib.tokens import TokenCount, count_tokens  # noqa: E402

DEFAULT_AGENT = "agents/kukai/agent.yaml"
DEFAULT_OPENWEBUI_URL = "http://localhost:3000"
DEFAULT_LLAMA_SWAP_URL = "http://localhost:8080"
DEFAULT_LLAMA_SWAP_MODEL = "kukai"


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="agent.yaml を Open WebUI に同期する")
    parser.add_argument("--agent", default=DEFAULT_AGENT, help="agent.yaml のパス（リポジトリ直下から）")
    parser.add_argument("--dry-run", action="store_true", help="送信せず差分とトークン数だけ表示する")
    return parser.parse_args(argv)


def run(
    argv: list[str],
    *,
    root: Path,
    env: dict,
    client_factory: Callable[[str, str], object],
    token_counter: Callable[..., TokenCount],
) -> int:
    args = parse_args(argv)

    api_key = env.get("OPENWEBUI_API_KEY", "")
    if not api_key:
        print("error: OPENWEBUI_API_KEY が設定されていません（.env を確認してください）", file=sys.stderr)
        return 2

    agent = yaml.safe_load((root / args.agent).read_text(encoding="utf-8"))
    spec = agent["system_prompt"]
    system_prompt = build_system_prompt(load_parts(root, spec["parts"]))

    tokens = token_counter(
        system_prompt,
        base_url=env.get("LLAMA_SWAP_URL", DEFAULT_LLAMA_SWAP_URL),
        model=env.get("LLAMA_SWAP_MODEL", DEFAULT_LLAMA_SWAP_MODEL),
    )
    budget = int(spec.get("budget_tokens", 0))
    exactness = "exact" if tokens.exact else "estimated"
    print(f"system prompt: {len(spec['parts'])} parts, {tokens.count} tokens ({exactness}) / budget {budget}")
    if budget and is_over_budget(tokens.count, budget):
        print(f"warning: system prompt exceeds budget ({tokens.count} > {budget} tokens)", file=sys.stderr)

    client = client_factory(env.get("OPENWEBUI_URL", DEFAULT_OPENWEBUI_URL), api_key)
    plan = plan_sync(build_model_form(agent, system_prompt), client.get_model(agent["id"]))
    print(f"plan: {plan.action} {agent['id']}")
    if plan.diff:
        print(plan.diff)

    if args.dry_run:
        print("dry-run: nothing sent")
        return 0

    result = apply_plan(plan, client, dry_run=False)
    if result is None:
        print("done: no change")
    else:
        print(f"done: {plan.action}d {result.get('id', agent['id'])}")
    return 0


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    env = {**load_dotenv(root / ".env"), **os.environ}
    return run(sys.argv[1:], root=root, env=env, client_factory=OpenWebUIClient, token_counter=count_tokens)


if __name__ == "__main__":
    sys.exit(main())
