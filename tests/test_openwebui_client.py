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

    def get(self, url, headers=None, params=None, timeout=None):
        self.calls.append(("GET", url, headers, params, None))
        return self.responses.pop(0)

    def post(self, url, headers=None, params=None, json=None, timeout=None):
        self.calls.append(("POST", url, headers, params, json))
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
