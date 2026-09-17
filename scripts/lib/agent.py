"""agents/<agent>/agent.yaml の読み込みと形の検査。"""
from pathlib import Path
from typing import Optional

import yaml

REQUIRED_AGENT_KEYS = ("id", "name", "base_model_id", "system_prompt")


def validate_agent(agent) -> Optional[str]:
    """agent.yaml の形を確かめ、問題があればその説明を返す。"""
    if not isinstance(agent, dict):
        return "agent.yaml が辞書形式ではありません"
    missing = [key for key in REQUIRED_AGENT_KEYS if key not in agent]
    if missing:
        return f"agent.yaml に必須項目がありません: {', '.join(missing)}"
    spec = agent["system_prompt"]
    parts = spec.get("parts") if isinstance(spec, dict) else None
    if not isinstance(parts, list) or not parts or not all(isinstance(p, str) for p in parts):
        return "agent.yaml の system_prompt.parts はファイルパスの一覧（1 件以上）である必要があります"
    return None


def load_agent(root: Path, relative: str) -> dict:
    """agent.yaml を読み、形が正しければ辞書を返す。問題があれば ValueError。"""
    agent = yaml.safe_load((Path(root) / relative).read_text(encoding="utf-8"))
    problem = validate_agent(agent)
    if problem:
        raise ValueError(problem)
    return agent
