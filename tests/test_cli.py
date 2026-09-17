"""sync_openwebui CLI: 依存（クライアント・トークン計測）を差し替えて流れを検証する。"""
from pathlib import Path

import yaml

from scripts.lib.tokens import TokenCount
from scripts.sync_openwebui import run

AGENT_YAML = {
    "id": "kukai-ai",
    "name": "空海",
    "base_model_id": "kukai",
    "description": "d",
    "system_prompt": {"parts": ["agents/kukai/a.md", "agents/kukai/b.md"], "budget_tokens": 10},
    "params": {"temperature": 0.5},
}


class RecordingClient:
    def __init__(self, existing=None):
        self.existing = existing
        self.calls = []

    def get_model(self, model_id):
        return self.existing

    def create_model(self, form):
        self.calls.append(("create", form))
        return form

    def update_model(self, model_id, form):
        self.calls.append(("update", model_id, form))
        return form


def make_repo(tmp_path: Path) -> Path:
    agent_dir = tmp_path / "agents" / "kukai"
    agent_dir.mkdir(parents=True)
    (agent_dir / "a.md").write_text("alpha\n", encoding="utf-8")
    (agent_dir / "b.md").write_text("beta\n", encoding="utf-8")
    (agent_dir / "agent.yaml").write_text(yaml.safe_dump(AGENT_YAML, allow_unicode=True), encoding="utf-8")
    return tmp_path


def run_cli(argv, tmp_path, *, client, tokens=TokenCount(3, True), env=None):
    created = []

    def client_factory(url, api_key):
        created.append((url, api_key))
        return client

    rc = run(
        argv,
        root=make_repo(tmp_path),
        env={"OPENWEBUI_API_KEY": "k"} if env is None else env,
        client_factory=client_factory,
        token_counter=lambda text, **_: tokens,
    )
    return rc, created


def test_dry_run_reports_plan_and_token_count_without_sending(tmp_path, capsys):
    client = RecordingClient(existing=None)

    rc, _ = run_cli(["--dry-run"], tmp_path, client=client)

    out = capsys.readouterr().out
    assert rc == 0
    assert "plan: create kukai-ai" in out
    assert "3 tokens (exact)" in out
    assert "dry-run" in out
    assert client.calls == []


def test_run_creates_model_with_system_prompt_assembled_from_parts(tmp_path):
    client = RecordingClient(existing=None)

    rc, created = run_cli([], tmp_path, client=client)

    assert rc == 0
    assert created == [("http://localhost:3000", "k")]
    action, form = client.calls[0]
    assert action == "create"
    assert form["params"]["system"] == "alpha\n\nbeta\n"
    assert form["params"]["temperature"] == 0.5


def test_run_reports_noop_when_server_already_matches(tmp_path, capsys):
    from scripts.lib.sync import build_model_form

    agent = {**AGENT_YAML, "knowledge": [], "filters": []}
    client = RecordingClient(existing=build_model_form(agent, "alpha\n\nbeta\n"))

    rc, _ = run_cli([], tmp_path, client=client)

    assert rc == 0
    assert "plan: noop" in capsys.readouterr().out
    assert client.calls == []


def test_run_updates_and_prints_system_prompt_diff_when_server_differs(tmp_path, capsys):
    from scripts.lib.sync import build_model_form

    agent = {**AGENT_YAML, "knowledge": [], "filters": []}
    client = RecordingClient(existing=build_model_form(agent, "alpha\n\nOLD\n"))

    rc, _ = run_cli([], tmp_path, client=client)

    out = capsys.readouterr().out
    assert rc == 0
    assert "plan: update kukai-ai" in out
    assert "-OLD" in out
    assert "+beta" in out
    assert client.calls[0][:2] == ("update", "kukai-ai")


def test_run_warns_when_system_prompt_exceeds_budget_but_still_syncs(tmp_path, capsys):
    client = RecordingClient(existing=None)

    rc, _ = run_cli([], tmp_path, client=client, tokens=TokenCount(11, True))

    captured = capsys.readouterr()
    assert rc == 0
    assert "budget" in captured.err.lower()
    assert "11" in captured.err
    assert len(client.calls) == 1


def test_run_fails_fast_when_api_key_is_missing(tmp_path, capsys):
    client = RecordingClient(existing=None)

    rc, created = run_cli([], tmp_path, client=client, env={})

    assert rc == 2
    assert "OPENWEBUI_API_KEY" in capsys.readouterr().err
    assert created == []
    assert client.calls == []


def test_run_uses_openwebui_url_from_env(tmp_path):
    client = RecordingClient(existing=None)

    _, created = run_cli([], tmp_path, client=client, env={"OPENWEBUI_API_KEY": "k", "OPENWEBUI_URL": "http://x:1"})

    assert created == [("http://x:1", "k")]
