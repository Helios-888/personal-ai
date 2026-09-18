"""凍結した評価セット（evaluations/kukai/questions.yaml）を 1 つの条件で問い、回答を記録する。

設計は docs/specs/2026-09-18-phase3-design.md「比較の土俵」「評価実行スクリプト」。

条件（--condition）:
  B0 … 素のモデル。基底モデルに 1 文のシステムプロンプト（B0_SYSTEM_PROMPT）だけを付けて問う
  B1 … 常時層のみ。agent.yaml の parts から組んだシステムプロンプトで、Knowledge なし
経路（--route）:
  openwebui  … Open WebUI の /api/chat/completions（既定。Phase 4 と同じ経路）。
               B1 は登録済みのカスタムモデル（kukai-ai）に問い、システムプロンプトは送らない
               （Open WebUI が登録済みのものを先頭に足す）。問う前に、登録がリポジトリの定義と同じかを確かめる
  llama-swap … llama-swap の /v1/chat/completions（撤退時の経路）。システムプロンプトを要求に入れて送る

使い方（llm01 のリポジトリ直下で。約 30 分かかるので tmux の中で走らせる）:
  .venv/bin/python scripts/run_eval.py --condition B1
  → evaluations/kukai/baseline/<今日>-B1/answers.jsonl（機械が読む）と transcript.md（人が読む）
  既にあれば上書きしない。回答は 1 件ごとに answers.jsonl へ追記し、最後に完走か中断かの印を書く。
  baseline/ に置く記録は、全問・既定の反復・コミット済みの定義とコードでだけ取れる。
  経路の先行確認（1 問・1 回）の例:
  .venv/bin/python scripts/run_eval.py --condition B1 --only T01 --repeats 1 --out-dir evaluations/kukai/preflight
"""
import argparse
import datetime as dt
import hashlib
import os
import signal
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Optional

if __package__ in (None, ""):  # スクリプトとして直接実行されたとき、リポジトリ直下を import 経路に加える
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import requests  # noqa: E402
import yaml  # noqa: E402

from scripts.lib.agent import load_agent  # noqa: E402
from scripts.lib.cli import LABEL_PATTERN, display, positive_int  # noqa: E402
from scripts.lib.env import load_dotenv  # noqa: E402
from scripts.lib.evaluation import (  # noqa: E402
    AnswerRecord,
    EvalQuestion,
    RunHeader,
    answer_json,
    count_truncated,
    end_json,
    header_json,
    inconsistent_prompt_tokens,
    load_eval_questions,
    plan_calls,
    render_eval_transcript,
    select_questions,
)
from scripts.lib.freeze import read_frozen_text  # noqa: E402
from scripts.lib.git_state import GitState, git_state  # noqa: E402
from scripts.lib.llm_client import DEFAULT_ENDPOINT, ChatResult, chat_completion  # noqa: E402
from scripts.lib.openwebui_client import OpenWebUIClient, OpenWebUIError  # noqa: E402
from scripts.lib.prompt_builder import build_system_prompt, load_parts, resolve_inside  # noqa: E402
from scripts.lib.sync import build_model_form, changed_fields  # noqa: E402
from scripts.lib.tokens import TokenCount, count_tokens  # noqa: E402

B0_SYSTEM_PROMPT = "あなたは空海（774–835）です。空海として一人称で答えてください。"
CONDITIONS = ("B0", "B1")
ROUTE_OPENWEBUI = "openwebui"
ROUTE_LLAMA_SWAP = "llama-swap"
OPENWEBUI_ENDPOINT = "/api/chat/completions"
TEMPLATE_MARK = "{{"  # Open WebUI は {{CURRENT_DATE}} などをシステムプロンプトの中で置き換える
DEFAULT_AGENT = "agents/kukai/agent.yaml"
DEFAULT_QUESTIONS = "evaluations/kukai/questions.yaml"
DEFAULT_OUT_DIR = "evaluations/kukai/baseline"
CODE_DIR = "scripts"  # 記録を作ったコード。未コミットなら基準値にしない
DEFAULT_OPENWEBUI_URL = "http://localhost:3000"
DEFAULT_LLAMA_SWAP_URL = "http://localhost:8080"
DEFAULT_REPEATS = 3
DEFAULT_MAX_TOKENS = 800
DEFAULT_TEMPERATURE = 0.7
MAX_TRUNCATED_RATE = 0.05  # 設計の成功基準 3：finish=length（出力枠切れ）は 5% 以下
ANSWERS_FILE = "answers.jsonl"
TRANSCRIPT_FILE = "transcript.md"
INTERRUPTED_EXIT_CODE = 130
INPUT_ERRORS = (ValueError, OSError, yaml.YAMLError)  # 送る前に分かる不備はすべて終了コード 2


@dataclass(frozen=True)
class Target:
    """送信先。system が None なら要求にシステムプロンプトを入れない。"""

    base_url: str
    endpoint: str
    model: str
    headers: Optional[dict] = field(repr=False)  # API キーを含む。表示に出さない
    system: Optional[str]


@dataclass(frozen=True)
class EvalSetup:
    target: Target
    system_prompt: str  # モデルが実際に受け取るシステムプロンプト（指紋を記録する）
    base_model: str
    questions: list[EvalQuestion]
    questions_sha256: str
    temperature: float
    git: GitState
    out_dir: Path


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="凍結した評価セットを 1 つの条件で問い、回答を記録する")
    parser.add_argument("--condition", required=True, choices=CONDITIONS, help="B0（素のモデル）か B1（常時層のみ）")
    parser.add_argument("--route", choices=(ROUTE_OPENWEBUI, ROUTE_LLAMA_SWAP), default=ROUTE_OPENWEBUI)
    parser.add_argument("--label", default=None, help="保存先フォルダ名の末尾に足す識別子（英数字・ドット・ハイフン・下線）")
    parser.add_argument("--repeats", type=positive_int, default=DEFAULT_REPEATS, help="1 問あたりの反復回数")
    parser.add_argument("--only", default=None, help="問う問いの id をカンマ区切りで（経路の先行確認用）")
    parser.add_argument("--agent", default=DEFAULT_AGENT, help="agent.yaml のパス（リポジトリ直下から）")
    parser.add_argument("--questions", default=DEFAULT_QUESTIONS, help="凍結した評価セットのパス")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="記録フォルダを作る場所")
    parser.add_argument("--max-tokens", type=positive_int, default=DEFAULT_MAX_TOKENS, help="1 回答あたりの出力上限")
    return parser.parse_args(argv)


def run(
    argv: list[str],
    *,
    root: Path,
    env: dict,
    chat: Callable[..., ChatResult],
    token_counter: Callable[..., TokenCount],
    client_factory: Callable[[str, str], object],
    git_state: Callable[[Path, list[str]], GitState],
    today: str,
) -> int:
    args = parse_args(argv)
    try:
        setup = prepare(args, root, env, today, client_factory, git_state)
    except requests.RequestException:
        raise  # つながらないのは入力の不備ではない（main で終了コード 1）
    except INPUT_ERRORS as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    tokens = token_counter(
        setup.system_prompt, base_url=env.get("LLAMA_SWAP_URL", DEFAULT_LLAMA_SWAP_URL), model=setup.base_model
    )
    header = build_header(args, setup, tokens, today)
    print(
        f"{header.condition} via {header.route}: {header.model} at {header.url}; "
        f"system prompt {tokens.count} tokens ({'exact' if tokens.exact else 'estimated'}); "
        f"{header.question_count} questions x {header.repeats}"
    )
    if not tokens.exact:
        print("warning: token count is an estimate (llama-swap tokenize was unreachable)", file=sys.stderr)

    records = ask_all(setup, header, chat, root)
    save_new(setup.out_dir / TRANSCRIPT_FILE, render_eval_transcript(header, records))
    report(records)
    print(f"saved: {display(setup.out_dir, root)}")
    return 0


def prepare(
    args: argparse.Namespace,
    root: Path,
    env: dict,
    today: str,
    client_factory: Callable[[str, str], object],
    git_state: Callable[[Path, list[str]], GitState],
) -> EvalSetup:
    """入力を読み、送る前に確かめられることはすべてここで確かめる（LLM は呼ばない）。"""
    if args.label is not None and not LABEL_PATTERN.match(args.label):
        raise ValueError("--label は英数字・ドット・ハイフン・下線のみです")
    only = split_ids(args.only)
    is_baseline = (root / args.out_dir).resolve() == (root / DEFAULT_OUT_DIR).resolve()
    if is_baseline and (only is not None or args.repeats != DEFAULT_REPEATS):
        raise ValueError("絞り込み（--only・--repeats）の記録は baseline/ に置きません。--out-dir を指定してください")
    name = f"{today}-{args.condition}" + (f"-{args.label}" if args.label else "")
    out_dir = root / args.out_dir / name
    if out_dir.exists():
        raise ValueError(f"記録が既にあります。上書きしません: {display(out_dir, root)}")

    agent = load_agent(root, args.agent)
    if args.condition == "B1" and agent.get("knowledge"):
        raise ValueError("B1 は Knowledge なしの条件です。agent.yaml の knowledge が空ではありません")
    questions_text = read_frozen_text(resolve_inside(root, args.questions))
    questions = select_questions(load_eval_questions(questions_text), only)
    parts = agent["system_prompt"]["parts"]
    system_prompt = B0_SYSTEM_PROMPT if args.condition == "B0" else build_system_prompt(load_parts(root, parts))

    tracked = [args.agent, args.questions, CODE_DIR, *(parts if args.condition == "B1" else [])]
    git = git_state(root, tracked)
    if is_baseline and git.dirty:
        raise ValueError(
            "baseline/ の記録は、定義とコードをコミットしてから取ります。未コミットの変更: " + "、".join(git.dirty)
        )
    return EvalSetup(
        target=choose_target(args.condition, args.route, agent, system_prompt, env, client_factory),
        system_prompt=system_prompt,
        base_model=agent["base_model_id"],
        questions=questions,
        questions_sha256=sha256_text(questions_text),
        temperature=float((agent.get("params") or {}).get("temperature", DEFAULT_TEMPERATURE)),
        git=git,
        out_dir=out_dir,
    )


def choose_target(
    condition: str,
    route: str,
    agent: dict,
    system_prompt: str,
    env: dict,
    client_factory: Callable[[str, str], object],
) -> Target:
    if route == ROUTE_LLAMA_SWAP:
        base_url = env.get("LLAMA_SWAP_URL", DEFAULT_LLAMA_SWAP_URL)
        return Target(base_url, DEFAULT_ENDPOINT, agent["base_model_id"], None, system_prompt)

    if TEMPLATE_MARK in system_prompt:
        raise ValueError(
            f"システムプロンプトに {TEMPLATE_MARK} があります。Open WebUI が置き換えるため、"
            "記録する指紋と実際に届くものがずれます"
        )
    api_key = env.get("OPENWEBUI_API_KEY", "")
    if not api_key:
        raise ValueError("OPENWEBUI_API_KEY が設定されていません（.env を確認してください）")
    base_url = env.get("OPENWEBUI_URL", DEFAULT_OPENWEBUI_URL)
    headers = {"Authorization": f"Bearer {api_key}"}
    if condition == "B0":
        return Target(base_url, OPENWEBUI_ENDPOINT, agent["base_model_id"], headers, system_prompt)
    check_registration(client_factory(base_url, api_key), agent, system_prompt)
    return Target(base_url, OPENWEBUI_ENDPOINT, agent["id"], headers, None)


def check_registration(client, agent: dict, system_prompt: str) -> None:
    """B1 を Open WebUI 経由で問う前に、登録済みのモデルがリポジトリの定義と同じであることを確かめる。

    システムプロンプトは Open WebUI が登録済みのものを足すため、ここが食い違うと記録の指紋と実際に送られたものがずれる。
    """
    existing = client.get_model(agent["id"])
    if existing is None:
        raise ValueError(f"Open WebUI に {agent['id']} が登録されていません。先に scripts/sync_openwebui.py で登録してください")
    changes = changed_fields(build_model_form(agent, system_prompt), existing)
    if changes:
        raise ValueError(
            f"Open WebUI の {agent['id']} がリポジトリの定義と食い違います（{', '.join(changes)}）。"
            "scripts/sync_openwebui.py --dry-run で差分を確認してください"
        )


def build_header(args: argparse.Namespace, setup: EvalSetup, tokens: TokenCount, today: str) -> RunHeader:
    return RunHeader(
        condition=args.condition,
        route=args.route,
        url=f"{setup.target.base_url.rstrip('/')}{setup.target.endpoint}",
        model=setup.target.model,
        date=today,
        questions_sha256=setup.questions_sha256,
        system_prompt_sha256=sha256_text(setup.system_prompt),
        system_prompt_sent=setup.target.system is not None,
        system_tokens=tokens.count,
        tokens_exact=tokens.exact,
        temperature=setup.temperature,
        max_tokens=args.max_tokens,
        repeats=args.repeats,
        question_count=len(setup.questions),
        agent=args.agent,
        question_ids=tuple(q.id for q in setup.questions),
        git_commit=setup.git.commit,
        git_dirty=setup.git.dirty,
    )


def ask_all(
    setup: EvalSetup, header: RunHeader, chat: Callable[..., ChatResult], root: Path
) -> list[AnswerRecord]:
    """巡回の順に問い、1 件ごとに answers.jsonl へ追記する。最後に完走か中断かの印を書く。

    途中で失敗したら、中断の印と「途中で中断」の transcript.md を残してから失敗を伝える。
    """
    calls = plan_calls(setup.questions, header.repeats)
    answers_path = setup.out_dir / ANSWERS_FILE
    setup.out_dir.mkdir(parents=True)  # 既にあれば失敗する（同時に走らせても上書きしない）
    append_line(answers_path, header_json(header), create=True)
    records: list[AnswerRecord] = []
    target = setup.target
    try:
        for number, (repeat, question) in enumerate(calls, start=1):
            result = chat(
                target.base_url,
                target.model,
                system=target.system,
                user=question.text,
                temperature=setup.temperature,
                max_tokens=header.max_tokens,
                endpoint=target.endpoint,
                headers=target.headers,
            )
            record = AnswerRecord(repeat, question, result)
            append_line(answers_path, answer_json(header.condition, record))
            records = [*records, record]
            print(
                f"[{number}/{len(calls)} {question.id} r{repeat}] {result.elapsed_seconds:.1f} 秒、"
                f"{len(result.content.strip())} 字、入力 {result.prompt_tokens}、finish={result.finish_reason}",
                flush=True,  # tmux や nohup の下でも進み具合がすぐ見えるように
            )
    except BaseException:
        append_line(answers_path, end_json("interrupted", len(records), inconsistent_prompt_tokens(records)))
        note = f"途中で中断（{len(records)}/{len(calls)} 回答まで）"
        save_new(setup.out_dir / TRANSCRIPT_FILE, render_eval_transcript(replace(header, note=note), records))
        print(f"saved (partial): {display(setup.out_dir, root)}", file=sys.stderr)
        raise
    append_line(answers_path, end_json("complete", len(records), inconsistent_prompt_tokens(records)))
    return records


def report(records: list[AnswerRecord]) -> None:
    truncated = count_truncated(records)
    print(f"finish=length: {truncated}/{len(records)}")
    if records and truncated / len(records) > MAX_TRUNCATED_RATE:
        print(
            f"warning: 出力枠切れが {MAX_TRUNCATED_RATE:.0%} を超えています（設計の成功基準 3）。"
            "max_tokens を変えるなら、両条件を同じ値で取り直してください",
            file=sys.stderr,
        )
    inconsistent = inconsistent_prompt_tokens(records)
    if inconsistent:
        print(
            f"warning: 同じ問いなのに入力トークン数が回ごとに違います（{', '.join(inconsistent)}）。"
            "途中でシステムプロンプトか Open WebUI の設定が変わった可能性があります",
            file=sys.stderr,
        )
    if any(record.result.reasoning for record in records):
        print("warning: 思考 OFF のはずが、応答に思考（reasoning_content）が含まれています", file=sys.stderr)


def split_ids(raw: Optional[str]) -> Optional[list[str]]:
    """--only の値を id の一覧にする。指定したのに空なら誤り（空を「全問」とみなさない）。"""
    if raw is None:
        return None
    ids = [part.strip() for part in raw.split(",") if part.strip()]
    if not ids:
        raise ValueError("--only に問いの id がありません")
    return ids


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def append_line(path: Path, line: str, *, create: bool = False) -> None:
    with open(path, "x" if create else "a", encoding="utf-8", newline="\n") as file:
        file.write(line + "\n")


def save_new(path: Path, text: str) -> None:
    with open(path, "x", encoding="utf-8", newline="\n") as file:  # 同名があれば失敗する（上書きしない）
        file.write(text)


def raise_interrupt(signum, frame) -> None:
    raise KeyboardInterrupt(f"signal {signum}")


def install_interrupt_handlers() -> None:
    """ssh の切断（SIGHUP）や kill（SIGTERM）を Ctrl-C と同じ扱いにし、中断の記録を書いてから止まるようにする。"""
    for name in ("SIGTERM", "SIGHUP"):
        signum = getattr(signal, name, None)  # Windows に SIGHUP は無い
        if signum is not None:
            signal.signal(signum, raise_interrupt)


def main(argv: Optional[list[str]] = None, root: Optional[Path] = None) -> int:
    repo_root = Path(root) if root else Path(__file__).resolve().parents[1]
    env = {**load_dotenv(repo_root / ".env"), **os.environ}  # 環境変数が .env より優先
    install_interrupt_handlers()
    try:
        return run(
            sys.argv[1:] if argv is None else argv,
            root=repo_root,
            env=env,
            chat=chat_completion,
            token_counter=count_tokens,
            client_factory=OpenWebUIClient,
            git_state=git_state,
            today=dt.date.today().isoformat(),
        )
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return INTERRUPTED_EXIT_CODE
    except (OpenWebUIError, requests.RequestException, OSError, yaml.YAMLError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
