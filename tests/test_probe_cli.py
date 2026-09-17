"""probe_voice CLI: 依存（LLM 呼び出し・トークン計測）を差し替えて流れを検証する。"""
from pathlib import Path

import pytest
import yaml

from scripts.lib.llm_client import ChatResult
from scripts.lib.tokens import TokenCount
from scripts.probe_voice import run

AGENT_YAML = {
    "id": "kukai-ai",
    "name": "空海",
    "base_model_id": "kukai",
    "description": "d",
    "system_prompt": {"parts": ["agents/kukai/a.md", "agents/kukai/b.md"], "budget_tokens": 5000},
    "params": {"temperature": 0.3},
}
QUESTIONS_YAML = {"questions": [{"kind": "語調", "text": "問一"}, {"kind": "判断", "text": "問二"}]}


def make_repo(tmp_path: Path) -> Path:
    agent_dir = tmp_path / "agents" / "kukai"
    agent_dir.mkdir(parents=True)
    (agent_dir / "a.md").write_text("alpha\n", encoding="utf-8")
    (agent_dir / "b.md").write_text("beta\n", encoding="utf-8")
    (agent_dir / "agent.yaml").write_text(yaml.safe_dump(AGENT_YAML, allow_unicode=True), encoding="utf-8")
    eval_dir = tmp_path / "evaluations" / "kukai" / "phase2-voice-check"
    eval_dir.mkdir(parents=True)
    (eval_dir / "questions.yaml").write_text(yaml.safe_dump(QUESTIONS_YAML, allow_unicode=True), encoding="utf-8")
    return tmp_path


class RecordingChat:
    def __init__(self):
        self.calls = []

    def __call__(self, base_url, model, *, system, user, temperature, max_tokens):
        self.calls.append(
            dict(base_url=base_url, model=model, system=system, user=user, temperature=temperature, max_tokens=max_tokens)
        )
        return ChatResult(content=f"答 {user}", reasoning="", finish_reason="stop", completion_tokens=5, elapsed_seconds=1.5)


def run_cli(argv, root, *, chat=None, env=None, tokens=TokenCount(3, True)):
    chat = chat if chat is not None else RecordingChat()
    rc = run(
        argv,
        root=root,
        env={} if env is None else env,
        chat=chat,
        token_counter=lambda text, **_: tokens,
        today="2026-09-18",
    )
    return rc, chat


def test_run_writes_a_transcript_named_by_date_and_label(tmp_path):
    root = make_repo(tmp_path)

    rc, _ = run_cli(["--label", "voice-draft2"], root)

    out = root / "evaluations" / "kukai" / "phase2-voice-check" / "2026-09-18-voice-draft2.md"
    assert rc == 0
    assert out.is_file()
    text = out.read_text(encoding="utf-8")
    assert "## [語調] 問一" in text
    assert "答 問一" in text
    assert "## [判断] 問二" in text
    assert "システムプロンプト 3 トークン。" in text


def test_run_sends_the_system_prompt_built_from_agent_parts_with_the_agent_temperature(tmp_path):
    root = make_repo(tmp_path)

    _, chat = run_cli(["--label", "x", "--max-tokens", "640"], root)

    assert [c["user"] for c in chat.calls] == ["問一", "問二"]
    assert all(c["system"] == "alpha\n\nbeta\n" for c in chat.calls)
    assert all(c["temperature"] == 0.3 for c in chat.calls)
    assert all(c["max_tokens"] == 640 for c in chat.calls)


def test_run_targets_llama_swap_from_env_with_localhost_defaults(tmp_path):
    root = make_repo(tmp_path)

    _, chat = run_cli(["--label", "x"], root)
    assert chat.calls[0]["base_url"] == "http://localhost:8080"
    assert chat.calls[0]["model"] == "kukai"

    _, chat = run_cli(["--label", "y"], root, env={"LLAMA_SWAP_URL": "http://x:1", "LLAMA_SWAP_MODEL": "m"})
    assert chat.calls[0]["base_url"] == "http://x:1"
    assert chat.calls[0]["model"] == "m"


def test_run_prints_one_progress_line_per_question_and_the_output_path(tmp_path, capsys):
    root = make_repo(tmp_path)

    run_cli(["--label", "x"], root)

    out = capsys.readouterr().out
    assert "[語調] 1.5 秒" in out
    assert "[判断] 1.5 秒" in out
    assert "2026-09-18-x.md" in out


def test_run_refuses_to_overwrite_an_existing_transcript(tmp_path, capsys):
    root = make_repo(tmp_path)
    existing = root / "evaluations" / "kukai" / "phase2-voice-check" / "2026-09-18-x.md"
    existing.write_text("keep me", encoding="utf-8")

    rc, chat = run_cli(["--label", "x"], root)

    assert rc == 2
    assert existing.read_text(encoding="utf-8") == "keep me"
    assert chat.calls == []
    assert "2026-09-18-x.md" in capsys.readouterr().err


def test_run_requires_a_label(tmp_path):
    with pytest.raises(SystemExit):
        run_cli([], make_repo(tmp_path))


def test_run_warns_when_the_token_count_is_only_an_estimate(tmp_path, capsys):
    root = make_repo(tmp_path)

    rc, _ = run_cli(["--label", "x"], root, tokens=TokenCount(3, False))

    assert rc == 0
    assert "estimat" in capsys.readouterr().err.lower()


def test_run_returns_2_with_a_message_when_agent_yaml_is_malformed(tmp_path, capsys):
    root = make_repo(tmp_path)
    broken = {k: v for k, v in AGENT_YAML.items() if k != "system_prompt"}
    (root / "agents" / "kukai" / "agent.yaml").write_text(yaml.safe_dump(broken), encoding="utf-8")

    rc, chat = run_cli(["--label", "x"], root)

    assert rc == 2
    assert "system_prompt" in capsys.readouterr().err
    assert chat.calls == []


def test_main_merges_dotenv_with_environ_and_wires_real_dependencies(tmp_path, monkeypatch):
    import scripts.probe_voice as cli
    from scripts.lib.llm_client import chat_completion
    from scripts.lib.tokens import count_tokens

    root = make_repo(tmp_path)
    (root / ".env").write_text("LLAMA_SWAP_URL=http://file:1\n", encoding="utf-8")
    monkeypatch.setenv("LLAMA_SWAP_MODEL", "fromenv")
    seen = {}

    def fake_run(argv, *, root, env, chat, token_counter, today):
        seen.update(argv=argv, env=env, chat=chat, token_counter=token_counter, today=today)
        return 0

    monkeypatch.setattr(cli, "run", fake_run)

    assert cli.main(["--label", "x"], root=root) == 0
    assert seen["argv"] == ["--label", "x"]
    assert seen["env"]["LLAMA_SWAP_URL"] == "http://file:1"
    assert seen["env"]["LLAMA_SWAP_MODEL"] == "fromenv"
    assert seen["chat"] is chat_completion
    assert seen["token_counter"] is count_tokens
    assert len(seen["today"]) == len("2026-09-18")


def test_main_turns_network_errors_into_one_line_and_exit_1(tmp_path, monkeypatch, capsys):
    import requests

    import scripts.probe_voice as cli

    root = make_repo(tmp_path)

    def explode(*args, **kwargs):
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(cli, "run", explode)

    assert cli.main(["--label", "x"], root=root) == 1
    err = capsys.readouterr().err
    assert "refused" in err
    assert "Traceback" not in err
