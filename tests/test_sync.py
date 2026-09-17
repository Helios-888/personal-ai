"""sync: agent.yaml から ModelForm を組み立て、作成・更新・変更なしを判定して実行する。"""
from scripts.lib.sync import apply_plan, build_model_form, plan_sync

AGENT = {
    "id": "kukai-ai",
    "name": "空海",
    "base_model_id": "kukai",
    "description": "desc",
    "params": {"temperature": 0.7},
    "knowledge": [],
    "filters": [],
}


class RecordingClient:
    def __init__(self):
        self.calls = []

    def create_model(self, form):
        self.calls.append(("create", form))
        return {"id": form["id"]}

    def update_model(self, model_id, form):
        self.calls.append(("update", model_id, form))
        return {"id": model_id}


def test_build_model_form_maps_agent_yaml_to_openwebui_model_form():
    form = build_model_form(AGENT, "SYSTEM")

    assert form == {
        "id": "kukai-ai",
        "base_model_id": "kukai",
        "name": "空海",
        "meta": {"description": "desc", "knowledge": [], "filterIds": []},
        "params": {"system": "SYSTEM", "temperature": 0.7},
        "is_active": True,
    }


def test_plan_is_create_when_model_does_not_exist():
    plan = plan_sync(build_model_form(AGENT, "S"), existing=None)

    assert plan.action == "create"


def test_plan_is_noop_when_existing_model_already_matches_ignoring_extra_server_fields():
    form = build_model_form(AGENT, "S")
    existing = {
        **form,
        "user_id": "u1",
        "updated_at": 1,
        "meta": {**form["meta"], "profile_image_url": "/static/x.png"},
    }

    plan = plan_sync(form, existing)

    assert plan.action == "noop"


def test_plan_is_update_with_unified_diff_when_system_prompt_changed():
    desired = build_model_form(AGENT, "old line\nsame\n")
    existing = build_model_form(AGENT, "new line\nsame\n")

    plan = plan_sync(desired, existing)

    assert plan.action == "update"
    assert "-new line" in plan.diff
    assert "+old line" in plan.diff


def test_plan_is_update_when_only_name_changed():
    desired = build_model_form({**AGENT, "name": "空海（改）"}, "S")
    existing = build_model_form(AGENT, "S")

    assert plan_sync(desired, existing).action == "update"


def test_apply_create_sends_create_model():
    client = RecordingClient()
    plan = plan_sync(build_model_form(AGENT, "S"), existing=None)

    apply_plan(plan, client, dry_run=False)

    assert client.calls == [("create", plan.form)]


def test_apply_update_sends_update_model_with_id():
    client = RecordingClient()
    plan = plan_sync(build_model_form(AGENT, "new"), build_model_form(AGENT, "old"))

    apply_plan(plan, client, dry_run=False)

    assert client.calls == [("update", "kukai-ai", plan.form)]


def test_apply_noop_sends_nothing():
    client = RecordingClient()
    form = build_model_form(AGENT, "S")

    apply_plan(plan_sync(form, form), client, dry_run=False)

    assert client.calls == []


def test_apply_dry_run_sends_nothing_even_when_create_is_needed():
    client = RecordingClient()
    plan = plan_sync(build_model_form(AGENT, "S"), existing=None)

    apply_plan(plan, client, dry_run=True)

    assert client.calls == []


# --- レビュー指摘への回帰テスト ---


def test_build_model_form_rejects_system_inside_agent_params():
    import pytest

    with pytest.raises(ValueError) as excinfo:
        build_model_form({**AGENT, "params": {"system": "handwritten"}}, "assembled")

    assert "params.system" in str(excinfo.value)


def test_plan_is_update_when_a_param_was_removed_from_agent_yaml():
    desired = build_model_form({**AGENT, "params": {}}, "S")
    existing = build_model_form(AGENT, "S")  # server still has temperature 0.7

    assert plan_sync(desired, existing).action == "update"


def test_plan_ignores_is_active_and_meta_fields_the_repo_does_not_manage():
    form = build_model_form(AGENT, "S")
    existing = {
        **form,
        "is_active": False,
        "meta": {**form["meta"], "capabilities": {"vision": False}, "tags": [{"name": "x"}]},
    }

    assert plan_sync(form, existing).action == "noop"


def test_update_form_keeps_existing_meta_fields_the_repo_does_not_manage():
    desired = build_model_form(AGENT, "new")
    existing = {
        **build_model_form(AGENT, "old"),
        "meta": {"description": "desc", "knowledge": [], "filterIds": [], "capabilities": {"vision": False}},
    }

    plan = plan_sync(desired, existing)

    assert plan.action == "update"
    assert plan.form["meta"]["capabilities"] == {"vision": False}
    assert plan.form["meta"]["description"] == "desc"
    assert plan.form["params"]["system"] == "new"


def test_plan_lists_changed_fields_so_a_name_only_update_is_explained():
    desired = build_model_form({**AGENT, "name": "空海（改）"}, "S")
    existing = build_model_form(AGENT, "S")

    plan = plan_sync(desired, existing)

    assert plan.changes == ("name",)
    assert plan.diff == ""
