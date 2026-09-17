"""システムプロンプトの組み立て（純粋関数）。

agent.yaml の system_prompt.parts に列挙されたファイルを、その順に空行 1 つで連結する。
"""
from pathlib import Path


def build_system_prompt(parts: list[str]) -> str:
    """各 part の末尾空白を落とし、空行 1 つで連結して末尾に改行 1 つを付ける。"""
    cleaned = [part.rstrip() for part in parts]
    if not cleaned:
        return ""
    return "\n\n".join(cleaned) + "\n"


def load_parts(root: Path, relative_paths: list[str]) -> list[str]:
    """root からの相対パスを順に読み、本文のリストを返す。無いファイルは名前を添えて失敗する。"""
    base = Path(root).resolve()
    texts: list[str] = []
    for relative in relative_paths:
        path = (base / relative).resolve()
        if not path.is_relative_to(base):
            raise ValueError(f"system prompt part is outside the repository: {relative}")
        if not path.is_file():
            raise FileNotFoundError(f"system prompt part not found: {relative}")
        texts.append(path.read_text(encoding="utf-8"))
    return texts


def is_over_budget(token_count: int, budget: int) -> bool:
    return token_count > budget
