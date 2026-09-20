"""build_knowledge CLI: agent.yaml の Knowledge を Open WebUI に作り、登録の記録を書く流れ（Open WebUI は偽物）。"""
import hashlib
from pathlib import Path

import pytest
import yaml

from scripts.build_knowledge import run
from scripts.lib.knowledge import load_registered
from tests.openwebui_v0113 import EMBEDDING_CONFIG, RETRIEVAL_CONFIG, VERSION

P1 = "knowledge/kukai/primary/primary__即身成仏義.md"
L1 = "knowledge/kukai/later-attribution/later-attribution__御遺告.md"
TEXTS = {P1: "即身成佛義\n", L1: "遺告\n"}
AGENT_YAML = {
    "id": "kukai-ai",
    "name": "空海",
    "base_model_id": "kukai",
    "system_prompt": {"parts": ["agents/kukai/a.md"]},
    "knowledge": [{"name": "kukai-texts", "description": "空海の著作", "files": [P1, L1]}],
    "retrieval": {"embedding_model": "BAAI/bge-m3"},
}
RECORD = "agents/kukai/knowledge-registered.yaml"


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class FakeOpenWebUI:
    def __init__(self, embedding_model="BAAI/bge-m3", status="completed", version="0.11.3"):
        self.embedding_model = embedding_model
        self.status = status
        self.version = version
        self.retrieval = dict(RETRIEVAL_CONFIG)
        self.knowledge: dict[str, dict] = {}  # id → {name, files: {filename: {sha256, file_id}}}
        self.uploaded: dict[str, dict] = {}
        self.calls: list[tuple] = []
        self.corrupt_next_add = False  # 束に入れたときに中身が化けたことにする

    def get_version(self):
        return {**VERSION, "version": self.version}

    def get_embedding_config(self):
        return {**EMBEDDING_CONFIG, "RAG_EMBEDDING_MODEL": self.embedding_model}

    def get_retrieval_config(self):
        return self.retrieval

    def create_knowledge(self, name, description):
        kid = f"kid-{len(self.knowledge) + 1}"
        self.knowledge[kid] = {"name": name, "description": description, "files": {}}
        self.calls.append(("create", name))
        return {"id": kid, "name": name}

    def upload_file(self, filename, content, content_type="text/markdown"):
        fid = f"f{len(self.uploaded) + 1}"
        self.uploaded[fid] = {"filename": filename, "sha256": hashlib.sha256(content).hexdigest()}
        self.calls.append(("upload", filename, content_type))
        return {"id": fid, "filename": filename, "status": True}

    def file_process_status(self, file_id):
        return {"status": self.status}

    def add_file_to_knowledge(self, knowledge_id, file_id):
        upload = self.uploaded[file_id]
        digest = "0" * 64 if self.corrupt_next_add else upload["sha256"]
        self.knowledge[knowledge_id]["files"][upload["filename"]] = {"sha256": digest, "file_id": file_id}
        self.calls.append(("add", knowledge_id, file_id))
        return {"id": knowledge_id}

    def remove_file_from_knowledge(self, knowledge_id, file_id):
        files = self.knowledge[knowledge_id]["files"]
        name = next(n for n, f in files.items() if f["file_id"] == file_id)
        del files[name]
        self.calls.append(("remove", knowledge_id, file_id))
        return {"id": knowledge_id}

    def get_knowledge_files(self, knowledge_id):
        files = self.knowledge[knowledge_id]["files"]
        items = [{"id": f["file_id"], "filename": n, "meta": {"name": n, "file_hash": f["sha256"]}} for n, f in files.items()]
        return {"items": items, "total": len(items)}


def make_repo(tmp_path: Path, agent: dict = AGENT_YAML) -> Path:
    agent_dir = tmp_path / "agents" / "kukai"
    agent_dir.mkdir(parents=True)
    (agent_dir / "a.md").write_text("alpha\n", encoding="utf-8")
    (agent_dir / "agent.yaml").write_text(yaml.safe_dump(agent, allow_unicode=True), encoding="utf-8")
    for relative, text in TEXTS.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode("utf-8"))
    return tmp_path


def run_cli(argv, root, server, env=None):
    made = []

    def factory(url, api_key):
        made.append((url, api_key))
        return server

    rc = run(argv, root=root, env={"OPENWEBUI_API_KEY": "k"} if env is None else env, client_factory=factory)
    return rc, made


def read_record(root: Path):
    return load_registered((root / RECORD).read_text(encoding="utf-8"))


def test_dry_run_shows_the_plan_and_touches_nothing(tmp_path, capsys):
    root, server = make_repo(tmp_path), FakeOpenWebUI()

    rc, _ = run_cli(["--dry-run"], root, server)

    out = capsys.readouterr().out
    assert rc == 0
    assert "plan: create knowledge kukai-texts（2 ファイル）" in out
    assert f"{P1} {sha(TEXTS[P1])[:12]}" in out
    assert "dry-run" in out
    assert server.calls == []
    assert not (root / RECORD).exists()


def test_run_creates_the_knowledge_uploads_the_files_and_writes_the_record(tmp_path, capsys):
    root, server = make_repo(tmp_path), FakeOpenWebUI()

    rc, made = run_cli([], root, server)

    assert rc == 0
    assert made == [("http://localhost:3000", "k")]
    assert server.calls == [
        ("create", "kukai-texts"),
        ("upload", "primary__即身成仏義.md", "text/markdown"), ("add", "kid-1", "f1"),
        ("upload", "later-attribution__御遺告.md", "text/markdown"), ("add", "kid-1", "f2"),
    ]
    (registered,) = read_record(root)
    assert (registered.name, registered.id) == ("kukai-texts", "kid-1")
    assert [(f.path, f.sha256, f.file_id) for f in registered.files] == [
        (P1, sha(TEXTS[P1]), "f1"), (L1, sha(TEXTS[L1]), "f2"),
    ]
    assert registered.ingest["RAG_EMBEDDING_MODEL"] == "BAAI/bge-m3"  # 束を作ったときの取り込みの設定
    assert registered.ingest["CHUNK_SIZE"] == 1000
    assert b"\r" not in (root / RECORD).read_bytes()
    assert not list((root / RECORD).parent.glob("*.tmp"))  # 一時ファイルは置き換えで消える
    assert "done: kukai-texts kid-1、2 ファイル、リポジトリと同じ" in capsys.readouterr().out


def test_a_second_run_changes_nothing(tmp_path, capsys):
    root, server = make_repo(tmp_path), FakeOpenWebUI()
    run_cli([], root, server)
    server.calls.clear()

    rc, _ = run_cli([], root, server)

    assert rc == 0
    assert server.calls == []
    assert "noop: kukai-texts" in capsys.readouterr().out


def test_an_interrupted_build_resumes_with_the_missing_files_only(tmp_path):
    root, server = make_repo(tmp_path), FakeOpenWebUI()
    run_cli([], root, server)
    del server.knowledge["kid-1"]["files"]["later-attribution__御遺告.md"]  # 2 つ目を入れる前に止まった
    server.calls.clear()

    rc, _ = run_cli([], root, server)

    assert rc == 0
    assert server.calls == [("upload", "later-attribution__御遺告.md", "text/markdown"), ("add", "kid-1", "f3")]
    assert [f.file_id for f in read_record(root)[0].files] == ["f1", "f3"]


def test_it_stops_when_a_file_in_the_knowledge_differs_from_the_repository(tmp_path, capsys):
    root, server = make_repo(tmp_path), FakeOpenWebUI()
    run_cli([], root, server)
    (root / P1).write_bytes("書き換えた\n".encode("utf-8"))
    server.calls.clear()

    rc, _ = run_cli([], root, server)

    assert rc == 1
    assert server.calls == []  # 入れ直しは手で決める
    assert "中身が違います" in capsys.readouterr().err


def test_it_keeps_the_knowledge_id_when_the_embedding_fails_and_resumes_into_the_same_knowledge(tmp_path, capsys):
    root, server = make_repo(tmp_path), FakeOpenWebUI(status="failed")

    rc, _ = run_cli([], root, server)

    assert rc == 1
    err = capsys.readouterr().err
    assert "failed" in err and "f1" in err  # 束に入れなかったファイルの id を示す
    assert ("add", "kid-1", "f1") not in server.calls
    assert read_record(root)[0].id == "kid-1"

    server.status, server.calls = "completed", []
    assert run_cli([], root, server)[0] == 0
    assert [c for c in server.calls if c[0] == "create"] == []  # 束を二重に作らない
    assert len(server.knowledge) == 1


def test_it_stops_when_the_ingest_settings_changed_since_the_knowledge_was_made(tmp_path, capsys):
    # 区切り方を変えたまま続きを入れると、1 つの束に 2 種類の区切りが混ざる
    root, server = make_repo(tmp_path), FakeOpenWebUI()
    run_cli([], root, server)
    del server.knowledge["kid-1"]["files"]["later-attribution__御遺告.md"]
    server.retrieval = {**RETRIEVAL_CONFIG, "CHUNK_SIZE": 500}
    server.calls.clear()

    rc, _ = run_cli([], root, server)

    assert rc == 1
    assert server.calls == []
    assert "CHUNK_SIZE：1000 → 500" in capsys.readouterr().err


def test_it_stops_when_the_knowledge_holds_a_file_outside_the_definition(tmp_path, capsys):
    root, server = make_repo(tmp_path), FakeOpenWebUI()
    run_cli([], root, server)
    server.knowledge["kid-1"]["files"]["primary__秘蔵宝鑰.md"] = {"sha256": "cc", "file_id": "f9"}
    server.calls.clear()

    rc, _ = run_cli([], root, server)

    assert rc == 1
    assert server.calls == []
    assert "primary__秘蔵宝鑰.md" in capsys.readouterr().err


def test_it_stops_when_the_knowledge_differs_right_after_adding(tmp_path, capsys):
    root, server = make_repo(tmp_path), FakeOpenWebUI()
    server.corrupt_next_add = True

    rc, _ = run_cli([], root, server)

    assert rc == 1
    assert "入れた後の照合" in capsys.readouterr().err
    assert read_record(root)[0].files == ()  # 照合を通っていないファイルは記録しない


def test_a_broken_record_stops_before_touching_open_webui(tmp_path, capsys):
    root, server = make_repo(tmp_path), FakeOpenWebUI()
    (root / RECORD).write_text("knowledge: [{name: kukai-texts}]\n", encoding="utf-8")

    rc, made = run_cli([], root, server)

    assert rc == 2
    assert made == []
    assert "knowledge-registered.yaml" in capsys.readouterr().err


def test_it_refuses_to_upload_under_a_different_embedding_model(tmp_path, capsys):
    # 英語用の埋め込みのまま入れると、切り替えた後に入れ直しになる
    root, server = make_repo(tmp_path), FakeOpenWebUI(embedding_model="sentence-transformers/all-MiniLM-L6-v2")

    rc, _ = run_cli([], root, server)

    assert rc == 1  # Open WebUI 側の状態なので 1（定義の不備は 2）
    assert server.calls == []
    err = capsys.readouterr().err
    assert "all-MiniLM-L6-v2" in err and "BAAI/bge-m3" in err


def test_it_refuses_an_open_webui_version_whose_source_was_not_read(tmp_path, capsys):
    root, server = make_repo(tmp_path), FakeOpenWebUI(version="0.12.0")

    rc, _ = run_cli([], root, server)

    assert rc == 1
    assert server.calls == []
    assert "0.12.0" in capsys.readouterr().err


@pytest.mark.parametrize(
    "agent, message",
    [
        ({**AGENT_YAML, "knowledge": []}, "knowledge が空"),
        ({k: v for k, v in AGENT_YAML.items() if k != "retrieval"}, "retrieval.embedding_model"),
    ],
    ids=["no-knowledge", "no-embedding-model"],
)
def test_it_refuses_an_incomplete_definition(tmp_path, capsys, agent, message):
    root, server = make_repo(tmp_path, agent), FakeOpenWebUI()

    rc, made = run_cli([], root, server)

    assert rc == 2
    assert made == []
    assert message in capsys.readouterr().err


def test_it_needs_the_api_key(tmp_path, capsys):
    rc, made = run_cli([], make_repo(tmp_path), FakeOpenWebUI(), env={})

    assert rc == 2
    assert made == []
    assert "OPENWEBUI_API_KEY" in capsys.readouterr().err


# --- 1 点だけ入れ直す（--replace、2026-09-20 Phase 5） ---------------------------------


RESTORED = chr(36978) + chr(21578) + chr(65288) + chr(24489) + chr(20803) + chr(65289) + chr(10)


def test_replace_swaps_a_changed_file_and_leaves_the_rest_alone(tmp_path, capsys):
    root, server = make_repo(tmp_path), FakeOpenWebUI()
    run_cli([], root, server)
    (root / L1).write_bytes(RESTORED.encode("utf-8"))
    server.calls.clear()

    rc, _ = run_cli(["--replace", L1], root, server)

    assert rc == 0
    assert ("remove", "kid-1", "f2") in server.calls
    assert server.knowledge["kid-1"]["files"]["later-attribution__御遺告.md"]["sha256"] == sha(RESTORED)
    assert server.knowledge["kid-1"]["files"]["primary__即身成仏義.md"]["file_id"] == "f1"  # ほかは触らない


def test_replace_shows_the_plan_without_touching_anything_on_a_dry_run(tmp_path, capsys):
    root, server = make_repo(tmp_path), FakeOpenWebUI()
    run_cli([], root, server)
    (root / L1).write_bytes(RESTORED.encode("utf-8"))
    server.calls.clear()

    rc, _ = run_cli(["--dry-run", "--replace", L1], root, server)

    assert rc == 0
    assert "replace" in capsys.readouterr().out
    assert server.calls == []


def test_a_changed_file_still_stops_when_replace_was_not_asked_for(tmp_path, capsys):
    root, server = make_repo(tmp_path), FakeOpenWebUI()
    run_cli([], root, server)
    (root / L1).write_bytes(RESTORED.encode("utf-8"))

    rc, _ = run_cli([], root, server)

    assert rc == 1
    assert "中身が違います" in capsys.readouterr().err


def test_replace_refuses_a_path_that_is_not_in_the_bundle(tmp_path, capsys):
    root, server = make_repo(tmp_path), FakeOpenWebUI()

    rc, _ = run_cli(["--replace", "knowledge/kukai/primary/primary__無い.md"], root, server)

    assert rc == 2
    assert "無い" in capsys.readouterr().err
    assert server.calls == []
