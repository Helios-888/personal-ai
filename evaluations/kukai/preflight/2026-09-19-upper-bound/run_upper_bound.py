"""上限の試験（Phase 4 設計書「上限の試験」）。使い捨て、リポジトリ直下から実行する。

正解の行を含む本文を手で渡し、本文のとおりに答えるかを見る。検索が完璧に当たった場合を再現するため、
Open WebUI v0.11.3 が検索結果を渡すのと同じ形にする：
  - system：段 1 の常時層（agent.yaml の parts を連結したもの）
  - user：RAG テンプレートの {{CONTEXT}} に <source id="1" name="..."> 本文 </source> を入れ、改行して問いを続ける
    （既定の RAG_SYSTEM_CONTEXT=False では、middleware.py の apply_source_context_to_messages が
     add_or_update_user_message(..., append=False) で問いの前に付けるため）
経路は llama-swap 直。記録は同じフォルダの answers.jsonl と transcript.md（上書きしない）。

  .venv/bin/python evaluations/kukai/preflight/2026-09-19-upper-bound/run_upper_bound.py
"""
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from scripts.lib.agent import load_agent  # noqa: E402
from scripts.lib.evaluation import load_eval_questions, select_questions  # noqa: E402
from scripts.lib.freeze import read_frozen_text  # noqa: E402
from scripts.lib.git_state import git_state  # noqa: E402
from scripts.lib.llm_client import chat_completion  # noqa: E402
from scripts.lib.prompt_builder import build_system_prompt, load_parts  # noqa: E402

HERE = Path(__file__).resolve().parent
LLAMA_SWAP_URL = "http://localhost:8080"
MODEL = "kukai"
AGENT = "agents/kukai/agent.yaml"
QUESTIONS = "evaluations/kukai/questions.yaml"
TEMPERATURE = 0.7  # agent.yaml の params と同じ
MAX_TOKENS = 2000
REPEATS = 3


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_user_message(template: str, source_name: str, passage: str, question: str) -> str:
    context = f'<source id="1" name="{source_name}">{passage}</source>'
    return template.replace("{{CONTEXT}}", context) + "\n" + question


def main() -> int:
    answers_path, transcript_path = HERE / "answers.jsonl", HERE / "transcript.md"
    if answers_path.exists() or transcript_path.exists():
        raise SystemExit("記録が既にあります。上書きしません")

    template = (HERE / "rag_template.txt").read_text(encoding="utf-8")
    passages = {p["id"]: p for p in yaml.safe_load((HERE / "passages.yaml").read_text(encoding="utf-8"))["passages"]}
    agent = load_agent(ROOT, AGENT)
    parts = agent["system_prompt"]["parts"]
    system_prompt = build_system_prompt(load_parts(ROOT, parts))
    questions = select_questions(load_eval_questions(read_frozen_text(ROOT / QUESTIONS)), list(passages))
    git = git_state(ROOT, [AGENT, *parts, "scripts"])

    header = {
        "date": dt.date.today().isoformat(),
        "route": "llama-swap", "url": LLAMA_SWAP_URL, "model": MODEL,
        "system_prompt_sha256": sha256(system_prompt),
        "rag_template_sha256": sha256(template),
        "passages_sha256": sha256((HERE / "passages.yaml").read_text(encoding="utf-8")),
        "temperature": TEMPERATURE, "max_tokens": MAX_TOKENS, "repeats": REPEATS,
        "question_ids": [q.id for q in questions],
        "git_commit": git.commit, "git_dirty": list(git.dirty),
    }
    lines = [f"# 上限の試験（{header['date']}）", ""]
    lines += [f"- {key}：{value}" for key, value in header.items()]
    records = []
    for repeat in range(1, REPEATS + 1):
        for question in questions:
            passage = passages[question.id]
            user = build_user_message(template, passage["source_name"], passage["text"], question.text)
            result = chat_completion(LLAMA_SWAP_URL, MODEL, system=system_prompt, user=user,
                                     temperature=TEMPERATURE, max_tokens=MAX_TOKENS)
            print(f"[{question.id} r{repeat}] {result.elapsed_seconds:.1f} 秒、入力 {result.prompt_tokens}、"
                  f"出力 {result.completion_tokens}、finish={result.finish_reason}")
            records.append({"id": question.id, "repeat": repeat, "prompt_tokens": result.prompt_tokens,
                            "completion_tokens": result.completion_tokens,
                            "finish_reason": result.finish_reason, "content": result.content})

    with answers_path.open("x", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps({"header": header}, ensure_ascii=False) + "\n")
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    for question in questions:
        lines += ["", f"## {question.id}", "", f"> {question.text}", "",
                  f"本文：{passages[question.id]['source_name']} {passages[question.id]['lines']}"]
        for record in (r for r in records if r["id"] == question.id):
            lines += ["", f"### {record['repeat']} 回目（入力 {record['prompt_tokens']}、出力 "
                      f"{record['completion_tokens']}、finish={record['finish_reason']}）", "", record["content"]]
    with transcript_path.open("x", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    print(f"saved: {answers_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
