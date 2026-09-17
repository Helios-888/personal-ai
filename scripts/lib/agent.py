"""agents/<agent>/agent.yaml の読み込みと形の検査（sync_openwebui と probe_voice が共有する）。"""
from pathlib import Path
from typing import Optional

import yaml

from scripts.lib.prompt_builder import resolve_inside

REQUIRED_AGENT_KEYS = ("id", "name", "base_model_id", "system_prompt")
IDENTITY_KEYS = ("id", "name", "base_model_id")
TEMPERATURE_RANGE = (0, 2)


def validate_agent(agent) -> Optional[str]:
    """agent.yaml の形を確かめ、問題があればその説明を返す。"""
    if not isinstance(agent, dict):
        return "agent.yaml が辞書形式ではありません"
    missing = [key for key in REQUIRED_AGENT_KEYS if key not in agent]
    if missing:
        return f"agent.yaml に必須項目がありません: {', '.join(missing)}"
    for key in IDENTITY_KEYS:
        if not isinstance(agent[key], str) or not agent[key].strip():
            return f"agent.yaml の {key} は空でない文字列である必要があります"
    spec = agent["system_prompt"]
    parts = spec.get("parts") if isinstance(spec, dict) else None
    if not isinstance(parts, list) or not parts or not all(isinstance(p, str) for p in parts):
        return "agent.yaml の system_prompt.parts はファイルパスの一覧（1 件以上）である必要があります"
    return _validate_params(agent.get("params"))


def _validate_params(params) -> Optional[str]:
    if params is None:
        return None
    if not isinstance(params, dict):
        return "agent.yaml の params は辞書である必要があります"
    temperature = params.get("temperature")
    if temperature is None:
        return None
    low, high = TEMPERATURE_RANGE
    is_number = isinstance(temperature, (int, float)) and not isinstance(temperature, bool)
    if not is_number or not low <= temperature <= high:
        return f"agent.yaml の params.temperature は {low}〜{high} の数値である必要があります"
    return None


def load_agent(root: Path, relative: str) -> dict:
    """agent.yaml を読み、形が正しければ辞書を返す。リポジトリ外のパスや形の問題は ValueError。"""
    path = resolve_inside(root, relative)
    agent = yaml.safe_load(path.read_text(encoding="utf-8"))
    problem = validate_agent(agent)
    if problem:
        raise ValueError(problem)
    return agent
