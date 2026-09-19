"""同期の判定と実行。

agent.yaml → Open WebUI の ModelForm を組み立て、既存モデルと比べて
create / update / noop のどれかを決める。HTTP は client に委ねる。

比較するのはリポジトリが管理する項目だけ：name、base_model_id、meta の
description / knowledge / filterIds、params 全体。サーバーが付け足す項目
（profile_image_url、capabilities、tags、is_active など）は比較せず、更新時は
既存値を引き継ぐ。
"""
import difflib
from dataclasses import dataclass
from typing import Literal, Optional, Protocol, Sequence

Action = Literal["create", "update", "noop"]
COMPARED_TOP_LEVEL_KEYS = ("name", "base_model_id")
MANAGED_META_KEYS = ("description", "knowledge", "filterIds")


class ModelClient(Protocol):
    def create_model(self, form: dict) -> dict: ...

    def update_model(self, model_id: str, form: dict) -> dict: ...


@dataclass(frozen=True)
class SyncPlan:
    action: Action
    form: dict
    diff: str = ""
    changes: tuple[str, ...] = ()


def build_model_form(agent: dict, system_prompt: str, knowledge_refs: Sequence[dict] = ()) -> dict:
    """agent.yaml の内容を Open WebUI v0.11.3 の ModelForm の形にする。

    meta.knowledge には、agent.yaml の束の定義ではなく、Open WebUI に作った束の参照（knowledge.model_refs）を入れる。
    """
    params = dict(agent.get("params") or {})
    if "system" in params:
        raise ValueError("agent.yaml の params.system は指定できません（system_prompt.parts から組み立てます）")
    return {
        "id": agent["id"],
        "base_model_id": agent["base_model_id"],
        "name": agent["name"],
        "meta": {
            "description": agent.get("description", ""),
            "knowledge": [dict(ref) for ref in knowledge_refs],
            "filterIds": list(agent.get("filters") or []),
        },
        "params": {**params, "system": system_prompt},
        "is_active": True,
    }


def plan_sync(desired: dict, existing: Optional[dict]) -> SyncPlan:
    if existing is None:
        return SyncPlan("create", desired)
    changes = changed_fields(desired, existing)
    if not changes:
        return SyncPlan("noop", desired)
    return SyncPlan(
        "update",
        _merge_for_update(desired, existing),
        diff=_system_prompt_diff(existing, desired),
        changes=changes,
    )


def apply_plan(plan: SyncPlan, client: ModelClient, *, dry_run: bool) -> Optional[dict]:
    if dry_run or plan.action == "noop":
        return None
    if plan.action == "create":
        return client.create_model(plan.form)
    return client.update_model(plan.form["id"], plan.form)


def changed_fields(desired: dict, existing: dict) -> tuple[str, ...]:
    """リポジトリ管理下の項目のうち、サーバー側と異なるものの名前を返す。"""
    changed: list[str] = []
    for key in COMPARED_TOP_LEVEL_KEYS:
        if existing.get(key) != desired.get(key):
            changed.append(key)
    existing_meta = existing.get("meta") or {}
    for key in MANAGED_META_KEYS:
        if _normalize(existing_meta.get(key)) != _normalize(desired["meta"].get(key)):
            changed.append(f"meta.{key}")
    if (existing.get("params") or {}) != desired["params"]:
        changed.append("params")
    return tuple(changed)


def _normalize(value):
    """サーバーが None で返す空値をリポジトリ側の空値と同一視する。"""
    return value if value is not None else _EMPTY


class _Empty:
    """None / [] / "" を同じ空値として比べるための番兵。"""

    def __eq__(self, other):
        return other is None or other == [] or other == "" or isinstance(other, _Empty)

    __hash__ = None


_EMPTY = _Empty()


def _merge_for_update(desired: dict, existing: dict) -> dict:
    """更新時は、リポジトリが管理しない meta 項目（画像・capabilities 等）を既存値から引き継ぐ。"""
    merged_meta = {**(existing.get("meta") or {}), **desired["meta"]}
    return {**desired, "meta": merged_meta}


def _system_prompt_diff(existing: dict, desired: dict) -> str:
    before = (existing.get("params") or {}).get("system") or ""
    after = (desired.get("params") or {}).get("system") or ""
    lines = difflib.unified_diff(
        before.splitlines(), after.splitlines(), fromfile="openwebui", tofile="repo", lineterm=""
    )
    return "\n".join(lines)
