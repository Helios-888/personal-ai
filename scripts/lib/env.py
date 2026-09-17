""".env の最小パーサー（依存ライブラリを増やさない）。"""
from pathlib import Path

_INLINE_COMMENT_MARKERS = (" #", "\t#")


def load_dotenv(path: Path) -> dict[str, str]:
    """KEY=VALUE 行を辞書にする。コメント行・空行・`export ` 接頭辞・行末コメントを扱う。ファイルが無ければ空辞書。"""
    file = Path(path)
    if not file.is_file():
        return {}
    values: dict[str, str] = {}
    for raw_line in file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, _, raw_value = line.partition("=")
        values[key.strip()] = _clean_value(raw_value.strip())
    return values


def _clean_value(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    for marker in _INLINE_COMMENT_MARKERS:
        if marker in value:
            value = value.split(marker, 1)[0].rstrip()
    return value
