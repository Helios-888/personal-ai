"""Open WebUI v0.11.3 REST（/api/v1/models）の薄いラッパー。

実機で確認した挙動：
- GET  /api/v1/models/model?id=<id> … 無ければ 404、権限なしは 401
- POST /api/v1/models/create        … ModelForm を本文で送る
- POST /api/v1/models/model/update  … 対象は本文の id で指定（クエリではない）
"""
from typing import Optional

import requests


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

    def _post(self, path: str, body: dict) -> dict:
        response = self._session.post(
            f"{self._base}{path}", headers=self._headers, json=body, timeout=self._timeout
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
