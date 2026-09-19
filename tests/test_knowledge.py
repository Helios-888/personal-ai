"""knowledge: agent.yaml の Knowledge の定義、登録の記録、Open WebUI 上の束との照合。"""
import hashlib
from pathlib import Path

import pytest

from scripts.lib.knowledge import (
    KnowledgeSpec,
    Registered,
    RegisteredFile,
    compare,
    header_entry,
    knowledge_refs_for,
    load_registered,
    local_digests,
    model_refs,
    parse_specs,
    registered_path,
    render_registered,
    server_files,
)

P1 = "knowledge/kukai/primary/primary__即身成仏義.md"
L1 = "knowledge/kukai/later-attribution/later-attribution__御遺告.md"
RAW = [{"name": "kukai-texts", "description": "空海の著作", "files": [P1, L1]}]


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def make_files(root: Path, files: dict[str, str]) -> Path:
    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode("utf-8"))
    return root


# --- 定義 ---


def test_parse_specs_reads_name_description_and_files():
    assert parse_specs(RAW) == (KnowledgeSpec("kukai-texts", "空海の著作", (P1, L1)),)


@pytest.mark.parametrize("raw", [None, []])
def test_parse_specs_treats_empty_as_no_knowledge(raw):
    assert parse_specs(raw) == ()


@pytest.mark.parametrize(
    "raw, message",
    [
        ("kukai-texts", "一覧"),
        (["kukai-primary"], "辞書"),  # Phase 3 までの名前だけの書き方は受け付けない
        ([{"description": "d", "files": [P1]}], "name"),
        ([{"name": "k", "files": []}], "files"),
        ([{"name": "k", "description": 1, "files": [P1]}], "description"),
        ([{"name": "k", "files": [P1]}, {"name": "k", "files": [L1]}], "name が重複"),
        ([{"name": "a", "files": [P1]}, {"name": "b", "files": [P1]}], "files が重複"),
        ([{"name": "a", "files": [P1, "knowledge/kukai/other/primary__即身成仏義.md"]}], "ファイル名が重複"),
    ],
    ids=["not-list", "names-only", "no-name", "no-files", "bad-description", "dup-name", "dup-file", "dup-basename"],
)
def test_parse_specs_refuses_malformed_entries(raw, message):
    with pytest.raises(ValueError, match=message):
        parse_specs(raw)


# --- リポジトリのファイル ---


def test_local_digests_fingerprints_the_raw_bytes(tmp_path):
    root = make_files(tmp_path, {P1: "即身\n", L1: "遺告\n"})
    assert local_digests(root, parse_specs(RAW)) == {P1: sha("即身\n"), L1: sha("遺告\n")}


@pytest.mark.parametrize(
    "files, path, message",
    [
        ({}, P1, "ありません"),
        ({"agents/kukai/primary/primary__x.md": "x"}, "agents/kukai/primary/primary__x.md", "knowledge/ の下"),
        ({"knowledge/kukai/primary/later-attribution__x.md": "x"}, "knowledge/kukai/primary/later-attribution__x.md", "区分"),
        ({}, "../outside/primary/primary__x.md", "outside the repository"),
    ],
    ids=["missing", "outside-knowledge", "wrong-prefix", "outside-repo"],
)
def test_local_digests_refuses_files_it_should_not_upload(tmp_path, files, path, message):
    root = make_files(tmp_path, files)
    with pytest.raises(ValueError, match=message):
        local_digests(root, (KnowledgeSpec("k", "", (path,)),))


# --- 登録の記録 ---

REGISTERED = (Registered("kukai-texts", "kid-1", (RegisteredFile(P1, "aa", "f1"), RegisteredFile(L1, "bb", "f2"))),)


def test_registered_record_round_trips_and_says_not_to_edit_by_hand():
    text = render_registered(REGISTERED)
    assert text.startswith("# Open WebUI の Knowledge の登録の記録")
    assert "手で直さない" in text
    assert load_registered(text) == REGISTERED


@pytest.mark.parametrize(
    "text",
    [
        "knowledge: x",
        "knowledge: [{name: k}]",
        "knowledge: [{name: k, id: i, files: [{path: p}]}]",
    ],
    ids=["not-list", "no-id", "file-without-hash"],
)
def test_a_broken_record_stops_with_a_message(text):
    with pytest.raises(ValueError, match="knowledge-registered.yaml"):
        load_registered(text)


def test_registered_record_keeps_the_ingest_settings():
    entry = Registered("kukai-texts", "kid-1", (), {"CHUNK_SIZE": 1000, "RAG_EMBEDDING_MODEL": "BAAI/bge-m3"})
    assert load_registered(render_registered((entry,))) == (entry,)


def test_registered_path_sits_next_to_agent_yaml():
    assert registered_path("agents/kukai/agent.yaml") == "agents/kukai/knowledge-registered.yaml"


def test_model_refs_link_the_registered_ids_as_collections():
    assert model_refs(parse_specs(RAW), REGISTERED) == [{"id": "kid-1", "name": "kukai-texts", "type": "collection"}]


def test_model_refs_refuse_a_knowledge_not_yet_built():
    with pytest.raises(ValueError, match="build_knowledge.py"):
        model_refs(parse_specs(RAW), ())


def test_knowledge_refs_for_reads_the_record_next_to_agent_yaml(tmp_path):
    record = tmp_path / "agents" / "kukai" / "knowledge-registered.yaml"
    record.parent.mkdir(parents=True)
    record.write_text(render_registered(REGISTERED), encoding="utf-8")
    refs = knowledge_refs_for(tmp_path, "agents/kukai/agent.yaml", {"knowledge": RAW})
    assert refs == [{"id": "kid-1", "name": "kukai-texts", "type": "collection"}]
    assert knowledge_refs_for(tmp_path, "agents/kukai/agent.yaml", {"knowledge": []}) == []


# --- Open WebUI 上の束との照合 ---


def listing(*items, total=None) -> dict:
    return {"items": list(items), "total": len(items) if total is None else total}


def item(name: str, file_hash: str, file_id: str = "f") -> dict:
    return {"id": file_id, "filename": name, "meta": {"name": name, "file_hash": file_hash}}


def test_server_files_reads_names_hashes_and_ids():
    got = server_files(listing(item("primary__即身成仏義.md", "aa", "f1")))
    assert got == {"primary__即身成仏義.md": {"sha256": "aa", "file_id": "f1"}}


def test_server_files_refuses_a_listing_cut_by_paging():
    with pytest.raises(ValueError, match="1 ページ"):
        server_files(listing(item("a.md", "x"), total=31))


@pytest.mark.parametrize(
    "raw, message",
    [({"items": []}, "total"), ({"items": [{"id": "f", "meta": {}}], "total": 1}, "名前")],
    ids=["no-total", "no-name"],
)
def test_server_files_refuses_a_listing_it_cannot_trust(raw, message):
    with pytest.raises(ValueError, match=message):
        server_files(raw)


def test_server_files_refuses_two_files_with_the_same_name():
    with pytest.raises(ValueError, match="同じ名前"):
        server_files(listing(item("a.md", "x"), item("a.md", "y")))


def test_compare_finds_missing_changed_and_extra_files():
    spec = parse_specs(RAW)[0]
    digests = {P1: "aa", L1: "bb"}
    on_server = server_files(listing(item("primary__即身成仏義.md", "zz"), item("primary__秘蔵宝鑰.md", "cc")))
    result = compare(spec, digests, on_server)
    assert result.missing == (L1,)
    assert result.changed == (P1,)
    assert result.extra == ("primary__秘蔵宝鑰.md",)
    assert not result.same
    assert len(result.problems("kukai-texts")) == 3


def test_compare_passes_when_the_knowledge_holds_exactly_the_repository_files():
    spec = parse_specs(RAW)[0]
    on_server = server_files(listing(item("primary__即身成仏義.md", "aa"), item("later-attribution__御遺告.md", "bb")))
    assert compare(spec, {P1: "aa", L1: "bb"}, on_server).same


def test_header_entry_lists_file_names_fingerprints_and_server_ids_in_definition_order():
    spec = parse_specs(RAW)[0]
    on_server = server_files(listing(item("primary__即身成仏義.md", "aa", "f1"), item("later-attribution__御遺告.md", "bb", "f2")))
    assert header_entry(spec, "kid-1", {P1: "aa", L1: "bb"}, on_server) == {
        "name": "kukai-texts",
        "id": "kid-1",
        "files": [
            {"name": "primary__即身成仏義.md", "sha256": "aa", "file_id": "f1"},
            {"name": "later-attribution__御遺告.md", "sha256": "bb", "file_id": "f2"},
        ],
    }


# --- 埋め込みモデル ---


def test_expected_embedding_model_reads_retrieval_block():
    from scripts.lib.knowledge import expected_embedding_model

    assert expected_embedding_model({"retrieval": {"embedding_model": " BAAI/bge-m3 "}}) == "BAAI/bge-m3"


@pytest.mark.parametrize("agent", [{}, {"retrieval": None}, {"retrieval": {"embedding_model": ""}}, {"retrieval": "x"}])
def test_expected_embedding_model_is_required(agent):
    from scripts.lib.knowledge import expected_embedding_model

    with pytest.raises(ValueError, match="retrieval.embedding_model"):
        expected_embedding_model(agent)


# --- 束を作ったときの取り込みの設定 ---


def test_ingest_settings_pick_what_shapes_the_stored_chunks():
    from scripts.lib.knowledge import ingest_settings
    from tests.openwebui_v0113 import EMBEDDING_CONFIG, RETRIEVAL_CONFIG

    got = ingest_settings(RETRIEVAL_CONFIG, EMBEDDING_CONFIG)
    assert got == {
        "RAG_EMBEDDING_ENGINE": "",
        "RAG_EMBEDDING_MODEL": "BAAI/bge-m3",
        "CONTENT_EXTRACTION_ENGINE": "",
        "TEXT_SPLITTER": "",
        "ENABLE_MARKDOWN_HEADER_TEXT_SPLITTER": True,
        "CHUNK_SIZE": 1000,
        "CHUNK_OVERLAP": 100,
        "CHUNK_MIN_SIZE_TARGET": 0,
    }


def test_ingest_differences_name_each_changed_value():
    from scripts.lib.knowledge import ingest_differences

    assert ingest_differences({"CHUNK_SIZE": 1000, "TOP": 1}, {"CHUNK_SIZE": 500, "TOP": 1}) == ["CHUNK_SIZE：1000 → 500"]
    assert ingest_differences(None, {"CHUNK_SIZE": 500}) == ["記録に取り込みの設定がありません"]


# --- 1 回答ごとの検索の確かめ（K1） ---

ENTRIES = ({"name": "kukai-texts", "id": "kid-1", "files": [{"name": "a.md", "sha256": "aa", "file_id": "f1"}]},)


def source(knowledge_id="kid-1", file_ids=("f1",)):
    return {
        "source": {"id": knowledge_id, "name": "kukai-texts", "type": "collection"},
        "document": [f"本文{i}" for i, _ in enumerate(file_ids)],
        "metadata": [{"file_id": fid, "name": "a.md"} for fid in file_ids],
    }


def test_retrieval_problem_passes_chunks_from_the_checked_knowledge():
    from scripts.lib.knowledge import retrieval_problem

    assert retrieval_problem((source(file_ids=("f1", "f1", "f1")),), ENTRIES) is None


@pytest.mark.parametrize(
    "sources, message",
    [
        ((), "検索された箇所がありません"),
        ((source(knowledge_id="kid-9"),), "kid-9"),
        ((source(file_ids=("f-old",)),), "f-old"),  # 束の一覧に無いファイルの箇所（古いベクトルの残り）
        (({"source": {"id": "kid-1"}, "document": [], "metadata": []},), "本文"),
        (({"source": {"id": "kid-1"}, "document": ["a", "b"], "metadata": [{"file_id": "f1"}]},), "metadata"),
    ],
    ids=["none", "other-knowledge", "stale-file", "no-document", "metadata-mismatch"],
)
def test_retrieval_problem_reports_retrieval_that_did_not_use_the_checked_knowledge(sources, message):
    from scripts.lib.knowledge import retrieval_problem

    assert message in retrieval_problem(sources, ENTRIES)
