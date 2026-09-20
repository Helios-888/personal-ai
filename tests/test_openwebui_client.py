"""openwebui_client: Open WebUI v0.11.3 の REST を薄く包む。HTTP は差し替え可能な session で検証する。"""
import pytest

from scripts.lib.openwebui_client import OpenWebUIAuthError, OpenWebUIClient, OpenWebUIError


class FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload
        self.text = str(payload)

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.uploads = []  # （files, timeout）。multipart で送ったもの
        self.timeouts = []

    def get(self, url, headers=None, params=None, timeout=None):
        self.calls.append(("GET", url, headers, params, None))
        self.timeouts.append(timeout)
        return self.responses.pop(0)

    def post(self, url, headers=None, params=None, json=None, timeout=None, files=None):
        self.calls.append(("POST", url, headers, params, json))
        self.timeouts.append(timeout)
        if files is not None:
            self.uploads.append(files)
        return self.responses.pop(0)


def make_client(*responses):
    session = FakeSession(responses)
    return OpenWebUIClient("http://owui:3000/", "KEY", session=session), session


def test_get_model_returns_model_dict_and_sends_bearer_key():
    client, session = make_client(FakeResponse(200, {"id": "kukai-ai", "params": {"system": "S"}}))

    model = client.get_model("kukai-ai")

    assert model == {"id": "kukai-ai", "params": {"system": "S"}}
    method, url, headers, params, _ = session.calls[0]
    assert (method, url, params) == ("GET", "http://owui:3000/api/v1/models/model", {"id": "kukai-ai"})
    assert headers["Authorization"] == "Bearer KEY"


def test_get_model_returns_none_when_server_says_not_found():
    client, _ = make_client(FakeResponse(404, {"detail": "Not found"}))

    assert client.get_model("kukai-ai") is None


def test_get_model_raises_auth_error_on_401():
    client, _ = make_client(FakeResponse(401, {"detail": "Unauthorized"}))

    with pytest.raises(OpenWebUIAuthError):
        client.get_model("kukai-ai")


def test_create_model_posts_form_to_create_endpoint():
    form = {"id": "kukai-ai", "name": "空海"}
    client, session = make_client(FakeResponse(200, {"id": "kukai-ai"}))

    assert client.create_model(form) == {"id": "kukai-ai"}
    method, url, _, _, body = session.calls[0]
    assert (method, url, body) == ("POST", "http://owui:3000/api/v1/models/create", form)


def test_update_model_posts_form_with_id_in_body_to_update_endpoint():
    form = {"id": "kukai-ai", "name": "空海"}
    client, session = make_client(FakeResponse(200, {"id": "kukai-ai"}))

    assert client.update_model("kukai-ai", form) == {"id": "kukai-ai"}
    method, url, _, _, body = session.calls[0]
    assert (method, url) == ("POST", "http://owui:3000/api/v1/models/model/update")
    assert body["id"] == "kukai-ai"


class NonJsonResponse(FakeResponse):
    def json(self):
        raise ValueError("not json")


def test_non_json_success_body_raises_error_instead_of_crashing():
    client, _ = make_client(NonJsonResponse(200, "<html>proxy page</html>"))

    with pytest.raises(OpenWebUIError) as excinfo:
        client.get_model("kukai-ai")

    assert "JSON" in str(excinfo.value)


def test_null_json_body_on_success_raises_error():
    client, _ = make_client(FakeResponse(200, None))

    with pytest.raises(OpenWebUIError):
        client.update_model("kukai-ai", {"id": "kukai-ai"})


def test_unexpected_status_raises_error_mentioning_status_and_body():
    client, _ = make_client(FakeResponse(500, {"detail": "boom"}))

    with pytest.raises(OpenWebUIError) as excinfo:
        client.create_model({"id": "x"})

    assert "500" in str(excinfo.value)
    assert "boom" in str(excinfo.value)


# --- Knowledge と RAG 設定（Phase 4） ---

KID = "0b1c2d3e-aaaa-bbbb-cccc-1234567890ab"


def test_create_knowledge_posts_name_and_description():
    client, session = make_client(FakeResponse(200, {"id": KID, "name": "kukai-texts"}))

    assert client.create_knowledge("kukai-texts", "空海の著作")["id"] == KID
    method, url, _, _, body = session.calls[0]
    assert (method, url, body) == ("POST", "http://owui:3000/api/v1/knowledge/create",
                                   {"name": "kukai-texts", "description": "空海の著作"})


def test_upload_file_sends_multipart_and_waits_for_the_embedding():
    client, session = make_client(FakeResponse(200, {"id": "f1", "status": True}))

    assert client.upload_file("primary__即身成仏義.md", "本文\n".encode("utf-8"))["id"] == "f1"
    method, url, headers, params, body = session.calls[0]
    assert (method, url, body) == ("POST", "http://owui:3000/api/v1/files/", None)
    assert params == {"process": "true", "process_in_background": "false"}  # 埋め込みが終わってから返る
    assert "Content-Type" not in headers  # multipart の境界は requests が付ける
    assert session.uploads == [{"file": ("primary__即身成仏義.md", "本文\n".encode("utf-8"), "text/markdown")}]
    assert session.timeouts[0] >= 600  # CPU の埋め込みは著作によって数分かかる


def test_file_process_status_reads_the_status():
    client, session = make_client(FakeResponse(200, {"status": "completed"}))

    assert client.file_process_status("f1") == {"status": "completed"}
    assert session.calls[0][1] == "http://owui:3000/api/v1/files/f1/process/status"


def test_add_file_to_knowledge_posts_the_file_id():
    client, session = make_client(FakeResponse(200, {"id": KID, "files": []}))

    client.add_file_to_knowledge(KID, "f1")
    method, url, _, _, body = session.calls[0]
    assert (method, url, body) == ("POST", f"http://owui:3000/api/v1/knowledge/{KID}/file/add", {"file_id": "f1"})
    assert session.timeouts[0] >= 600  # 束の側でも埋め込みが走る


def test_get_knowledge_files_asks_for_one_large_page():
    client, session = make_client(FakeResponse(200, {"items": [], "total": 0}))

    assert client.get_knowledge_files(KID) == {"items": [], "total": 0}
    method, url, _, params, _ = session.calls[0]
    assert (method, url) == ("GET", f"http://owui:3000/api/v1/knowledge/{KID}/files")
    assert params == {"page": 1, "limit": 100}


@pytest.mark.parametrize("bad", ["../models", "a/b", "", "x?y=1"])
def test_ids_in_paths_are_checked(bad):
    client, session = make_client()

    with pytest.raises(ValueError):
        client.get_knowledge_files(bad)
    with pytest.raises(ValueError):
        client.file_process_status(bad)
    assert session.calls == []


@pytest.mark.parametrize(
    "method, path",
    [
        ("get_retrieval_config", "/api/v1/retrieval/config"),
        ("get_embedding_config", "/api/v1/retrieval/embedding"),
        ("get_task_config", "/api/v1/tasks/config"),
        ("get_version", "/api/version"),
    ],
)
def test_settings_are_read_with_get(method, path):
    client, session = make_client(FakeResponse(200, {"status": True}))

    assert getattr(client, method)() == {"status": True}
    assert session.calls[0][:2] == ("GET", f"http://owui:3000{path}")


def test_the_expected_version_is_the_one_whose_source_was_read():
    from scripts.lib.openwebui_client import EXPECTED_VERSION

    assert EXPECTED_VERSION == "0.11.3"


def test_refresh_models_rebuilds_the_model_cache_without_returning_the_list():
    # Open WebUI は問いを受けたとき、モデルの Knowledge を手元の写し（app.state.MODELS）から読む。写しは GET /api/models で作り直される。
    # 一覧にはこのプロジェクト以外のモデルも載るので、中身は返さない
    client, session = make_client(FakeResponse(200, {"data": [{"id": "other-model"}]}))

    assert client.refresh_models() is None
    assert session.calls[0][:2] == ("GET", "http://owui:3000/api/models")


def test_refresh_models_raises_on_failure_without_echoing_the_body():
    client, _ = make_client(FakeResponse(500, {"data": [{"id": "other-model"}]}))

    with pytest.raises(OpenWebUIError) as excinfo:
        client.refresh_models()
    assert "other-model" not in str(excinfo.value)


def test_remove_file_from_knowledge_posts_the_file_id():
    client, session = make_client(FakeResponse(200, {"id": KID, "files": []}))

    client.remove_file_from_knowledge(KID, "f1")
    method, url, _, _, body = session.calls[0]
    assert (method, url, body) == ("POST", f"http://owui:3000/api/v1/knowledge/{KID}/file/remove", {"file_id": "f1"})


def test_remove_file_from_knowledge_checks_the_knowledge_id():
    client, session = make_client()

    with pytest.raises(ValueError):
        client.remove_file_from_knowledge("../models", "f1")
    assert session.calls == []
