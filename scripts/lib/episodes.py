"""episodes（判断様式の層）の読み込みと、ホールドアウト漏洩の検出。

伏せた episode の判断が principles.md（常時層）に載ると、Phase 3 のホールドアウト試験は
「覚えているか」を測るだけの試験になり無効化する。folder_mismatches と holdout_leaks が
それを機械的に弾く。
"""
from dataclasses import dataclass, field
from pathlib import Path

import yaml

EPISODES_DIR = Path("agents") / "kukai" / "episodes"
HOLDOUT_DIRNAME = "holdout"
FRONTMATTER_FENCE = "---"
REQUIRED_KEYS = ("id", "title", "year", "holdout", "principle", "fact_reliability", "sources", "verify")
# knowledge/_schema/metadata.yaml の reliability と同じ語のうち、episode に使うもの
ALLOWED_RELIABILITY = ("primary_text", "academic_research", "later_attribution", "commentary_premodern", "reference")


@dataclass(frozen=True)
class Episode:
    path: Path
    id: str
    title: str
    year: int
    holdout: bool
    principle: str
    fact_reliability: str
    sources: list[str]
    verify: list[str]
    confirmed: list[str] = field(default_factory=list)
    holdout_markers: list[str] = field(default_factory=list)

    @property
    def in_holdout_folder(self) -> bool:
        return self.path.parent.name == HOLDOUT_DIRNAME


def load_episodes(root: Path) -> list[Episode]:
    """episodes/ と episodes/holdout/ の Markdown を番号順に読む。形が崩れていればファイル名を添えて失敗する。"""
    base = Path(root) / EPISODES_DIR
    paths = sorted(
        (p for p in base.rglob("*.md") if p.name != "README.md"),
        key=lambda p: p.name,
    )
    return [_parse(path) for path in paths]


def folder_mismatches(episodes: list[Episode]) -> list[Path]:
    """holdout の旗と置き場所が食い違う episode のパスを返す。"""
    return [e.path for e in episodes if e.holdout != e.in_holdout_folder]


def holdout_leaks(episodes: list[Episode], principles_text: str) -> list[str]:
    """伏せた episode の目印か判断文が常時層に現れていれば、その説明を返す。"""
    leaks: list[str] = []
    for episode in episodes:
        if not episode.holdout:
            continue
        for term in (*episode.holdout_markers, episode.principle):
            if term and term in principles_text:
                leaks.append(f"{episode.id}（{episode.path.name}）の「{term}」が principles.md に現れています")
    return leaks


def unverified_principles(episodes: list[Episode], principles_text: str) -> list[str]:
    """verify が残ったままの episode の判断文が常時層に載っていれば、その説明を返す。"""
    return [
        f"{episode.id}（{episode.path.name}）は verify が {len(episode.verify)} 件残ったまま principles.md に載っています"
        for episode in episodes
        if episode.verify and episode.principle and episode.principle in principles_text
    ]


def _parse(path: Path) -> Episode:
    data = _frontmatter(path)
    missing = [key for key in REQUIRED_KEYS if key not in data]
    if missing:
        raise ValueError(f"{path.name}: frontmatter に必須項目がありません: {', '.join(missing)}")
    if data["fact_reliability"] not in ALLOWED_RELIABILITY:
        raise ValueError(
            f"{path.name}: fact_reliability は {', '.join(ALLOWED_RELIABILITY)} のいずれかである必要があります"
        )
    if data["holdout"] and not data.get("holdout_markers"):
        raise ValueError(f"{path.name}: 伏せる episode には holdout_markers（常時層に出してはならない語）が要ります")
    return Episode(
        path=path,
        id=str(data["id"]),
        title=str(data["title"]),
        year=int(data["year"]),
        holdout=bool(data["holdout"]),
        principle=str(data["principle"]),
        fact_reliability=str(data["fact_reliability"]),
        sources=list(data["sources"] or []),
        verify=list(data["verify"] or []),
        confirmed=list(data.get("confirmed") or []),
        holdout_markers=[str(m) for m in (data.get("holdout_markers") or [])],
    )


def _frontmatter(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    if not text.startswith(FRONTMATTER_FENCE):
        raise ValueError(f"{path.name}: 先頭に frontmatter（--- で囲んだ YAML）がありません")
    parts = text.split(FRONTMATTER_FENCE, 2)
    if len(parts) < 3:
        raise ValueError(f"{path.name}: frontmatter が --- で閉じられていません")
    data = yaml.safe_load(parts[1])
    if not isinstance(data, dict):
        raise ValueError(f"{path.name}: frontmatter が辞書形式ではありません")
    return data
