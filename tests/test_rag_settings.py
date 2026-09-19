"""rag_settings: Open WebUI の検索の設定から、記録に残す値と指紋を作る。"""
import pytest

from scripts.lib.rag_settings import rag_snapshot
from tests.openwebui_v0113 import EMBEDDING_CONFIG, RETRIEVAL_CONFIG, TASK_CONFIG


def snapshot(*, retrieval=None, tasks=None, fc="legacy", version="0.11.3"):
    return rag_snapshot({**RETRIEVAL_CONFIG, **(retrieval or {})}, EMBEDDING_CONFIG, {**TASK_CONFIG, **(tasks or {})}, fc,
                        version)


def test_snapshot_keeps_the_values_that_shape_retrieval():
    got = snapshot()
    assert got["RAG_EMBEDDING_MODEL"] == "BAAI/bge-m3"
    assert got["TOP_K"] == 3
    assert got["CHUNK_SIZE"] == 1000
    assert got["ENABLE_RETRIEVAL_QUERY_GENERATION"] is False
    assert got["function_calling"] == "legacy"
    assert got["OPENWEBUI_VERSION"] == "0.11.3"


def test_snapshot_replaces_long_templates_with_their_fingerprints():
    got = snapshot()
    assert got["RAG_TEMPLATE"].startswith("sha256:") and len(got["RAG_TEMPLATE"]) == 7 + 64
    assert got["QUERY_GENERATION_PROMPT_TEMPLATE"].startswith("sha256:")


def test_snapshot_leaves_out_keys_and_unrelated_settings():
    got = snapshot()
    assert "secret" not in str(got)  # 実物の応答には鍵の欄がある（tests/openwebui_v0113.py）
    assert "TITLE_GENERATION_PROMPT_TEMPLATE" not in got
    assert "status" not in got


@pytest.mark.parametrize(
    "changes",
    [dict(retrieval={"RAG_TEMPLATE": "別のテンプレート"}), dict(retrieval={"TOP_K": 5}),
     dict(tasks={"ENABLE_RETRIEVAL_QUERY_GENERATION": True}), dict(fc="native"), dict(version="0.12.0")],
    ids=["template", "top-k", "query-generation", "function-calling", "version"],
)
def test_the_fingerprint_changes_with_any_kept_value(changes):
    assert snapshot(**changes)["sha256"] != snapshot()["sha256"]


def test_the_fingerprint_ignores_unrelated_settings():
    assert snapshot(retrieval={"DOCLING_API_KEY": "other"})["sha256"] == snapshot()["sha256"]


def test_a_missing_setting_stops_instead_of_recording_a_gap():
    retrieval = {k: v for k, v in RETRIEVAL_CONFIG.items() if k != "TOP_K"}
    with pytest.raises(ValueError, match="TOP_K"):
        rag_snapshot(retrieval, EMBEDDING_CONFIG, TASK_CONFIG, "legacy", "0.11.3")
