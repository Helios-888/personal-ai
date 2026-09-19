"""Open WebUI v0.11.3 REST の薄いラッパー。

実機で確認した挙動：
- GET  /api/v1/models/model?id=<id> … 無ければ 404、権限なしは 401
- POST /api/v1/models/create        … ModelForm を本文で送る
- POST /api/v1/models/model/update  … 対象は本文の id で指定（クエリではない）

公開ソースで確認した挙動（Phase 4 の Knowledge。routers/files.py・routers/knowledge.py）：
- POST /api/v1/files/?process=true&process_in_background=false … 埋め込みまで終えてから返る。meta.file_hash は送ったバイト列の SHA-256
- POST /api/v1/knowledge/{id}/file/add … 処理済みのファイルを束に入れ、束の側でも埋め込む
- GET  /api/v1/knowledge/{id}/files   … 管理者は limit で 1 ページの件数を広げられる（既定 30）
"""
import re
from typing import Optional

import requests

# 上の挙動と、Phase 4 設計書「着手時に判明したこと」11〜14 は、この版の公開ソースで確かめた。版が変われば確かめ直す
EXPECTED_VERSION = "0.11.3"
UPLOAD_TIMEOUT = 1800  # 秒。コンテナの CPU で埋め込むので、長い著作は数分かかる
_ID = re.compile(r"[A-Za-z0-9-]+")  # パスに入れる id（Open WebUI の uuid）


class OpenWebUIError(RuntimeError):
    """想定外の HTTP 応答。"""


class OpenWebUIAuthError(OpenWebUIError):
    """API キーが無効か、権限がない。"""


class OpenWebUIClient:
    def __init__(self, base_url: str, api_key: str, session=None, timeout: float = 60):
        self._base = base_url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}
        self._session = session if session is not None else requests.Session()
        self._timeout = timeout

    def get_model(self, model_id: str) -> Optional[dict]:
        response = self._session.get(
            f"{self._base}/api/v1/models/model",
            headers=self._headers,
            params={"id": model_id},
            timeout=self._timeout,
        )
        if response.status_code == 404:
            return None
        return self._checked(response)

    def create_model(self, form: dict) -> dict:
        return self._post("/api/v1/models/create", form)

    def update_model(self, model_id: str, form: dict) -> dict:
        return self._post("/api/v1/models/model/update", {**form, "id": model_id})

    def create_knowledge(self, name: str, description: str) -> dict:
        return self._post("/api/v1/knowledge/create", {"name": name, "description": description})

    def upload_file(self, filename: str, content: bytes, content_type: str = "text/markdown") -> dict:
        """ファイルを送り、Open WebUI が本文を取り出して埋め込み終えるまで待つ。"""
        response = self._session.post(
            f"{self._base}/api/v1/files/",
            headers=self._headers,
            params={"process": "true", "process_in_background": "false"},
            files={"file": (filename, content, content_type)},
            timeout=UPLOAD_TIMEOUT,
        )
        return self._checked(response)

    def file_process_status(self, file_id: str) -> dict:
        return self._get(f"/api/v1/files/{_checked_id(file_id)}/process/status")

    def add_file_to_knowledge(self, knowledge_id: str, file_id: str) -> dict:
        return self._post(f"/api/v1/knowledge/{_checked_id(knowledge_id)}/file/add", {"file_id": file_id},
                          timeout=UPLOAD_TIMEOUT)

    def get_knowledge_files(self, knowledge_id: str, limit: int = 100) -> dict:
        return self._get(f"/api/v1/knowledge/{_checked_id(knowledge_id)}/files", params={"page": 1, "limit": limit})

    def get_retrieval_config(self) -> dict:
        return self._get("/api/v1/retrieval/config")

    def get_embedding_config(self) -> dict:
        return self._get("/api/v1/retrieval/embedding")

    def get_task_config(self) -> dict:
        return self._get("/api/v1/tasks/config")

    def get_version(self) -> dict:
        return self._get("/api/version")

    def _get(self, path: str, params: Optional[dict] = None) -> dict:
        response = self._session.get(f"{self._base}{path}", headers=self._headers, params=params, timeout=self._timeout)
        return self._checked(response)

    def _post(self, path: str, body: dict, timeout: Optional[float] = None) -> dict:
        response = self._session.post(
            f"{self._base}{path}", headers=self._headers, json=body, timeout=timeout or self._timeout
        )
        return self._checked(response)

    @staticmethod
    def _checked(response) -> dict:
        if response.status_code in (401, 403):
            raise OpenWebUIAuthError(f"authentication failed (HTTP {response.status_code}): {response.text}")
        if response.status_code >= 400:
            raise OpenWebUIError(f"HTTP {response.status_code}: {response.text}")
        try:
            payload = response.json()
        except ValueError as error:
            raise OpenWebUIError(
                f"HTTP {response.status_code}: response is not JSON: {str(response.text)[:200]}"
            ) from error
        if not isinstance(payload, dict):
            raise OpenWebUIError(f"HTTP {response.status_code}: unexpected JSON payload: {str(payload)[:200]}")
        return payload


def _checked_id(value: str) -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ValueError(f"Open WebUI の id として使えない文字があります: {value!r}")
    return value
