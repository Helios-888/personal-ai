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
        self.refreshed = 0  # モデル一覧の写しを作り直した回数

    def refresh_models(self):
        self.refreshed += 1

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


# --- レビュー指摘への回帰テスト ---


def test_run_returns_2_with_a_message_when_agent_yaml_is_malformed(tmp_path, capsys):
    root = make_repo(tmp_path)
    broken = {k: v for k, v in AGENT_YAML.items() if k != "system_prompt"}
    (root / "agents" / "kukai" / "agent.yaml").write_text(yaml.safe_dump(broken), encoding="utf-8")
    client = RecordingClient(existing=None)

    rc = run([], root=root, env={"OPENWEBUI_API_KEY": "k"}, client_factory=lambda u, k: client, token_counter=None)

    assert rc == 2
    assert "system_prompt" in capsys.readouterr().err
    assert client.calls == []


def test_run_warns_on_stderr_when_token_count_is_only_an_estimate(tmp_path, capsys):
    client = RecordingClient(existing=None)

    rc, _ = run_cli(["--dry-run"], tmp_path, client=client, tokens=TokenCount(3, False))

    captured = capsys.readouterr()
    assert rc == 0
    assert "estimat" in captured.err.lower()


def test_run_prints_changed_fields_when_only_the_name_changed(tmp_path, capsys):
    from scripts.lib.sync import build_model_form

    agent = {**AGENT_YAML, "name": "旧名", "knowledge": [], "filters": []}
    client = RecordingClient(existing=build_model_form(agent, "alpha\n\nbeta\n"))

    rc, _ = run_cli(["--dry-run"], tmp_path, client=client)

    out = capsys.readouterr().out
    assert rc == 0
    assert "plan: update kukai-ai" in out
    assert "changed: name" in out


def test_main_merges_dotenv_with_environ_and_wires_real_dependencies(tmp_path, monkeypatch):
    import scripts.sync_openwebui as cli
    from scripts.lib.openwebui_client import OpenWebUIClient
    from scripts.lib.tokens import count_tokens

    root = make_repo(tmp_path)
    (root / ".env").write_text("OPENWEBUI_API_KEY=fromfile\nOPENWEBUI_URL=http://file:1\n", encoding="utf-8")
    monkeypatch.setenv("OPENWEBUI_URL", "http://env:2")
    seen = {}

    def fake_run(argv, *, root, env, client_factory, token_counter):
        seen.update(argv=argv, root=root, env=env, client_factory=client_factory, token_counter=token_counter)
        return 0

    monkeypatch.setattr(cli, "run", fake_run)

    assert cli.main(["--dry-run"], root=root) == 0
    assert seen["argv"] == ["--dry-run"]
    assert seen["env"]["OPENWEBUI_API_KEY"] == "fromfile"
    assert seen["env"]["OPENWEBUI_URL"] == "http://env:2"  # 環境変数が .env より優先
    assert seen["client_factory"] is OpenWebUIClient
    assert seen["token_counter"] is count_tokens


def test_main_turns_openwebui_and_network_errors_into_one_line_and_exit_1(tmp_path, monkeypatch, capsys):
    import requests

    import scripts.sync_openwebui as cli
    from scripts.lib.openwebui_client import OpenWebUIError

    root = make_repo(tmp_path)
    (root / ".env").write_text("OPENWEBUI_API_KEY=k\n", encoding="utf-8")

    for error in (OpenWebUIError("HTTP 500: boom"), requests.ConnectionError("refused")):
        monkeypatch.setattr(cli, "run", lambda *a, **k: (_ for _ in ()).throw(error))

        assert cli.main([], root=root) == 1
        err = capsys.readouterr().err
        assert str(error) in err
        assert "Traceback" not in err


# --- Knowledge（Phase 4） ---

KNOWLEDGE = [{"name": "kukai-texts", "description": "d", "files": ["knowledge/kukai/primary/primary__x.md"]}]


def run_with_knowledge(tmp_path, *, registered):
    from scripts.lib.knowledge import Registered, render_registered

    root = make_repo(tmp_path)
    agent_dir = root / "agents" / "kukai"
    (agent_dir / "agent.yaml").write_text(
        yaml.safe_dump({**AGENT_YAML, "knowledge": KNOWLEDGE}, allow_unicode=True), encoding="utf-8"
    )
    if registered:
        record = render_registered((Registered("kukai-texts", "kid-1"),))
        (agent_dir / "knowledge-registered.yaml").write_text(record, encoding="utf-8")
    client = RecordingClient(existing=None)
    rc = run([], root=root, env={"OPENWEBUI_API_KEY": "k"}, client_factory=lambda u, k: client,
             token_counter=lambda text, **_: TokenCount(3, True))
    return rc, client


def test_run_links_the_built_knowledge_by_its_registered_id(tmp_path):
    rc, client = run_with_knowledge(tmp_path, registered=True)

    assert rc == 0
    action, form = client.calls[0]
    assert form["meta"]["knowledge"] == [{"id": "kid-1", "name": "kukai-texts", "type": "collection"}]


def test_run_refuses_a_knowledge_that_is_not_built_yet(tmp_path, capsys):
    rc, client = run_with_knowledge(tmp_path, registered=False)

    assert rc == 2
    assert "build_knowledge.py" in capsys.readouterr().err
    assert client.calls == []



def test_run_rebuilds_the_model_cache_after_syncing(tmp_path, capsys):
    # 写しを作り直さないと、API から問うたときに古い Knowledge の設定のまま答える（2026-09-19 の先行確認で起きた）
    client = RecordingClient(existing=None)

    rc, _ = run_cli([], tmp_path, client=client)

    assert rc == 0
    assert client.refreshed == 1
    assert "refreshed" in capsys.readouterr().out


def test_dry_run_does_not_touch_the_model_cache(tmp_path):
    client = RecordingClient(existing=None)

    run_cli(["--dry-run"], tmp_path, client=client)

    assert client.refreshed == 0
