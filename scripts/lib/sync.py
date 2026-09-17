"""同期の判定と実行。

agent.yaml → Open WebUI の ModelForm を組み立て、既存モデルと比べて
create / update / noop のどれかを決める。HTTP は client に委ねる。
"""
import difflib
from dataclasses import dataclass
from typing import Literal, Optional, Protocol

Action = Literal["create", "update", "noop"]


class ModelClient(Protocol):
    def create_model(self, form: dict) -> dict: ...

    def update_model(self, model_id: str, form: dict) -> dict: ...


@dataclass(frozen=True)
class SyncPlan:
    action: Action
    form: dict
    diff: str = ""


def build_model_form(agent: dict, system_prompt: str) -> dict:
    """agent.yaml の内容を Open WebUI v0.11.3 の ModelForm の形にする。"""
    return {
        "id": agent["id"],
        "base_model_id": agent["base_model_id"],
        "name": agent["name"],
        "meta": {
            "description": agent.get("description", ""),
            "knowledge": list(agent.get("knowledge") or []),
            "filterIds": list(agent.get("filters") or []),
        },
        "params": {"system": system_prompt, **(agent.get("params") or {})},
        "is_active": True,
    }


def plan_sync(desired: dict, existing: Optional[dict]) -> SyncPlan:
    if existing is None:
        return SyncPlan("create", desired)
    if _matches(desired, existing):
        return SyncPlan("noop", desired)
    return SyncPlan("update", desired, diff=_system_prompt_diff(existing, desired))


def apply_plan(plan: SyncPlan, client: ModelClient, *, dry_run: bool) -> Optional[dict]:
    if dry_run or plan.action == "noop":
        return None
    if plan.action == "create":
        return client.create_model(plan.form)
    return client.update_model(plan.form["id"], plan.form)


def _matches(desired: dict, existing: dict) -> bool:
    """desired にある項目だけを比べる（サーバー側が付け足す項目は無視）。"""
    for key, value in desired.items():
        current = existing.get(key)
        if isinstance(value, dict):
            if not isinstance(current, dict):
                return False
            if any(current.get(sub_key) != sub_value for sub_key, sub_value in value.items()):
                return False
        elif current != value:
            return False
    return True


def _system_prompt_diff(existing: dict, desired: dict) -> str:
    before = (existing.get("params") or {}).get("system") or ""
    after = (desired.get("params") or {}).get("system") or ""
    lines = difflib.unified_diff(
        before.splitlines(), after.splitlines(), fromfile="openwebui", tofile="repo", lineterm=""
    )
    return "\n".join(lines)
