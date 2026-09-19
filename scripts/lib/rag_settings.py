"""Open WebUI の検索の設定から、評価の記録に残す値と指紋を作る。純粋関数。

設計は docs/specs/2026-09-19-phase4-design.md「K1 の記録に残すもの」。RAG の設定は Open WebUI 全体の設定で、
リポジトリの外にある。問うたときの設定を記録に残し、あとで「どの設定で取った答えか」を確かめられるようにする。

残すのは検索の結果を左右する値だけ（v0.11.3 の GET /api/v1/retrieval/config・/retrieval/embedding・/tasks/config）。
API キーなど関係の無い値は残さない。長いテンプレートは SHA-256 に置き換える。指紋は、置き換える前の値全体から作る。
"""
import hashlib
import json

RETRIEVAL_KEYS = (
    "RAG_TEMPLATE",
    "TOP_K",
    "TOP_K_RERANKER",
    "RELEVANCE_THRESHOLD",
    "ENABLE_RAG_HYBRID_SEARCH",
    "HYBRID_BM25_WEIGHT",
    "RAG_FULL_CONTEXT",
    "BYPASS_EMBEDDING_AND_RETRIEVAL",
    "TEXT_SPLITTER",
    "ENABLE_MARKDOWN_HEADER_TEXT_SPLITTER",
    "CHUNK_SIZE",
    "CHUNK_OVERLAP",
    "CHUNK_MIN_SIZE_TARGET",
    "RAG_RERANKING_ENGINE",
    "RAG_RERANKING_MODEL",
)
EMBEDDING_KEYS = ("RAG_EMBEDDING_ENGINE", "RAG_EMBEDDING_MODEL")
# 資料を検索する前に、Open WebUI は既定でモデルに検索語を作らせる（問いそのものでは検索しない）
TASK_KEYS = ("ENABLE_RETRIEVAL_QUERY_GENERATION", "QUERY_GENERATION_PROMPT_TEMPLATE", "TASK_MODEL", "TASK_MODEL_EXTERNAL")
FINGERPRINTED_KEYS = ("RAG_TEMPLATE", "QUERY_GENERATION_PROMPT_TEMPLATE")


def rag_snapshot(retrieval: dict, embedding: dict, tasks: dict, function_calling: str, version: str) -> dict:
    """記録の見出しに残す検索の設定。

    function_calling はモデルの設定（legacy でないと API からは検索されない）。version は Open WebUI の版で、
    上の挙動や既定のテンプレートは版に依る（空のテンプレートは、その版の既定が使われる）。
    """
    values = {
        **_pick(retrieval, RETRIEVAL_KEYS, "retrieval/config"),
        **_pick(embedding, EMBEDDING_KEYS, "retrieval/embedding"),
        **_pick(tasks, TASK_KEYS, "tasks/config"),
        "function_calling": function_calling,
        "OPENWEBUI_VERSION": version,
    }
    shown = {key: f"sha256:{_sha256(value)}" if key in FINGERPRINTED_KEYS else value for key, value in values.items()}
    return {"sha256": _sha256(json.dumps(values, sort_keys=True, ensure_ascii=False)), **shown}


def _pick(source: dict, keys: tuple[str, ...], where: str) -> dict:
    missing = [key for key in keys if key not in source]
    if missing:
        raise ValueError(f"Open WebUI の {where} に {', '.join(missing)} がありません（版が変わった可能性）")
    return {key: source[key] for key in keys}


def _sha256(value) -> str:
    return hashlib.sha256(str(value if value is not None else "").encode("utf-8")).hexdigest()
