"""agent: agent.yaml の読み込みと形の検査（2 つの CLI が共有する要）。"""
import pytest
import yaml

from scripts.lib.agent import load_agent, validate_agent

VALID = {
    "id": "kukai-ai",
    "name": "空海",
    "base_model_id": "kukai",
    "system_prompt": {"parts": ["agents/kukai/system.md"]},
    "params": {"temperature": 0.7},
}


def test_validate_rejects_a_document_that_is_not_a_mapping():
    assert validate_agent(None) is not None
    assert validate_agent(["a"]) is not None


def test_validate_names_every_missing_required_key():
    problem = validate_agent({"id": "x"})

    assert "name" in problem
    assert "base_model_id" in problem
    assert "system_prompt" in problem


def test_validate_rejects_parts_that_is_a_single_string_instead_of_a_list():
    problem = validate_agent({**VALID, "system_prompt": {"parts": "agents/kukai/system.md"}})

    assert problem is not None
    assert "parts" in problem


def test_validate_rejects_an_empty_parts_list_and_non_string_entries():
    assert "parts" in validate_agent({**VALID, "system_prompt": {"parts": []}})
    assert "parts" in validate_agent({**VALID, "system_prompt": {"parts": ["a.md", 3]}})


def test_validate_rejects_empty_or_non_string_identity_fields():
    assert "id" in validate_agent({**VALID, "id": ""})
    assert "id" in validate_agent({**VALID, "id": None})
    assert "name" in validate_agent({**VALID, "name": "   "})


def test_validate_rejects_a_temperature_that_is_not_a_number_between_0_and_2():
    assert "temperature" in validate_agent({**VALID, "params": {"temperature": "ぬるめ"}})
    assert "temperature" in validate_agent({**VALID, "params": {"temperature": 3}})
    assert "temperature" in validate_agent({**VALID, "params": {"temperature": True}})
    assert validate_agent({**VALID, "params": {"temperature": 0}}) is None
    assert validate_agent({**VALID, "params": {}}) is None


def test_validate_accepts_the_reference_shape():
    assert validate_agent(VALID) is None


def test_load_agent_returns_the_mapping_when_the_file_is_valid(tmp_path):
    (tmp_path / "agent.yaml").write_text(yaml.safe_dump(VALID, allow_unicode=True), encoding="utf-8")

    assert load_agent(tmp_path, "agent.yaml") == VALID


def test_load_agent_raises_value_error_with_the_problem_when_invalid(tmp_path):
    (tmp_path / "agent.yaml").write_text("id: x\n", encoding="utf-8")

    with pytest.raises(ValueError) as excinfo:
        load_agent(tmp_path, "agent.yaml")

    assert "system_prompt" in str(excinfo.value)


def test_load_agent_refuses_a_path_outside_the_repository(tmp_path):
    outside = tmp_path.parent / "outside-agent.yaml"
    outside.write_text(yaml.safe_dump(VALID), encoding="utf-8")

    with pytest.raises(ValueError) as excinfo:
        load_agent(tmp_path, "../outside-agent.yaml")

    assert "outside" in str(excinfo.value)
