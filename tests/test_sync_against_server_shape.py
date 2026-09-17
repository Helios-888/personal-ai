"""実機（Open WebUI v0.11.3、2026-09-17）が GET /api/v1/models/model で返した形に対する冪等性の固定。

サーバーは profile_image_url / capabilities を null、access_grants を []、user を null で返し、
user_id / created_at / updated_at / write_access を付け足す。この形に対して plan が noop になることを守る。
"""
from scripts.lib.sync import build_model_form, plan_sync

AGENT = {
    "id": "kukai-ai",
    "name": "空海",
    "base_model_id": "kukai",
    "description": "空海（弘法大師）の著作と判断様式を再現する対話モデル。RAG 設計と評価手法の学習用。",
    "params": {"temperature": 0.7},
    "knowledge": [],
    "filters": [],
}
SYSTEM = "# 役割\n\nわたしは空海である。\n"

SERVER_RESPONSE = {
    "id": "kukai-ai",
    "user_id": "bc10a428-9b7c-4e7e-8c7d-d16deb9ff3fe",
    "base_model_id": "kukai",
    "name": "空海",
    "params": {"temperature": 0.7, "system": SYSTEM},
    "meta": {
        "profile_image_url": None,
        "description": AGENT["description"],
        "capabilities": None,
        "knowledge": [],
        "filterIds": [],
    },
    "access_grants": [],
    "is_active": True,
    "updated_at": 1789651415,
    "created_at": 1789651415,
    "user": None,
    "write_access": True,
}


def test_server_shape_after_create_is_reported_as_noop():
    assert plan_sync(build_model_form(AGENT, SYSTEM), SERVER_RESPONSE).action == "noop"


def test_server_shape_with_changed_prompt_is_reported_as_update_of_params_only():
    plan = plan_sync(build_model_form(AGENT, SYSTEM + "追記\n"), SERVER_RESPONSE)

    assert plan.action == "update"
    assert plan.changes == ("params",)
    assert plan.form["meta"]["profile_image_url"] is None  # 既存の未管理項目を引き継いで送る
