""".env の最小パーサー（依存ライブラリを増やさない）。"""
from pathlib import Path


def load_dotenv(path: Path) -> dict[str, str]:
    """KEY=VALUE 行を辞書にする。コメント行・空行は無視。ファイルが無ければ空辞書。"""
    file = Path(path)
    if not file.is_file():
        return {}
    values: dict[str, str] = {}
    for raw_line in file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = _unquote(value.strip())
    return values


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    return value
