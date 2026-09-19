"""Open WebUI の Knowledge（資料の束）の定義・登録の記録・照合。純粋関数（ファイルの読み込みを除く）。

設計は docs/specs/2026-09-19-phase4-design.md「段 2」。agent.yaml の knowledge に束の名前と入れるファイルを書き、
scripts/build_knowledge.py が Open WebUI に束を作って、束の id と入れたファイルの指紋を登録の記録
（agent.yaml と同じフォルダの knowledge-registered.yaml）に書く。同期（sync_openwebui.py）はその id をモデルに紐付け、
評価（run_eval.py）は Open WebUI 上のファイルの指紋がリポジトリのファイルと同じかを確かめてから問う。

Open WebUI v0.11.3 の公開ソースで確かめたこと：
- アップロードしたファイルの meta.file_hash は、送ったバイト列の SHA-256（routers/files.py upload_file_handler）
- モデルの meta.knowledge の項目は、type が collection なら id の束を検索する（retrieval/utils.py get_sources_from_items）
"""
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

from scripts.lib.prompt_builder import resolve_inside

KNOWLEDGE_ROOT = "knowledge"  # 入れるファイルはこの下に限る
REGISTERED_FILE = "knowledge-registered.yaml"
_REGISTERED_HEAD = (
    "# Open WebUI の Knowledge の登録の記録。scripts/build_knowledge.py が書く（手で直さない）\n"
    "# id は Open WebUI が付けた束の id。sha256 はアップロードしたファイルの指紋（Open WebUI の meta.file_hash と同じ）\n"
)


@dataclass(frozen=True)
class KnowledgeSpec:
    """agent.yaml に書く束の定義。"""

    name: str
    description: str
    files: tuple[str, ...]  # リポジトリ直下からのパス


@dataclass(frozen=True)
class RegisteredFile:
    path: str
    sha256: str
    file_id: str


@dataclass(frozen=True)
class Registered:
    """Open WebUI に作った束。ingest は束を作ったときの取り込みの設定（区切り方・埋め込み）。"""

    name: str
    id: str
    files: tuple[RegisteredFile, ...] = ()
    ingest: Optional[dict] = None


# 取り込み（本文の取り出し・区切り・埋め込み）の結果を決める設定。束の箇所はこの設定で作られ、後から設定を変えても
# 入れ直すまで変わらない。そこで束を作ったときの値を記録し、続きを入れるときと評価で問う前に今の値と照らす
INGEST_RETRIEVAL_KEYS = (
    "CONTENT_EXTRACTION_ENGINE",
    "TEXT_SPLITTER",
    "ENABLE_MARKDOWN_HEADER_TEXT_SPLITTER",
    "CHUNK_SIZE",
    "CHUNK_OVERLAP",
    "CHUNK_MIN_SIZE_TARGET",
)
INGEST_EMBEDDING_KEYS = ("RAG_EMBEDDING_ENGINE", "RAG_EMBEDDING_MODEL")


def parse_specs(raw) -> tuple[KnowledgeSpec, ...]:
    """agent.yaml の knowledge を読む。空なら Knowledge なし。"""
    if raw in (None, []):
        return ()
    if not isinstance(raw, list):
        raise ValueError("agent.yaml の knowledge は一覧である必要があります")
    specs = tuple(_spec(index, entry) for index, entry in enumerate(raw, start=1))
    _refuse_duplicates("knowledge の name ", [s.name for s in specs])
    _refuse_duplicates("knowledge の files ", [path for s in specs for path in s.files])
    _refuse_duplicates("knowledge のファイル名", [Path(path).name for s in specs for path in s.files])
    return specs


def _spec(index: int, entry) -> KnowledgeSpec:
    if not isinstance(entry, dict):
        raise ValueError(f"knowledge の {index} 番目は name・description・files を持つ辞書である必要があります")
    name, description, files = entry.get("name"), entry.get("description", ""), entry.get("files")
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"knowledge の {index} 番目に name（空でない文字列）がありません")
    if not isinstance(description, str):
        raise ValueError(f"knowledge {name} の description は文字列である必要があります")
    if not isinstance(files, list) or not files or not all(isinstance(f, str) and f.strip() for f in files):
        raise ValueError(f"knowledge {name} の files はファイルパスの一覧（1 件以上）である必要があります")
    return KnowledgeSpec(name.strip(), description, tuple(files))


def _refuse_duplicates(what: str, values: list[str]) -> None:
    duplicated = sorted({v for v in values if values.count(v) > 1})
    if duplicated:
        raise ValueError(f"{what}が重複しています: {', '.join(duplicated)}")


def expected_embedding_model(agent: dict) -> str:
    """agent.yaml の retrieval.embedding_model。Knowledge を入れる前と評価で問う前に、Open WebUI の設定と照らす。"""
    retrieval = agent.get("retrieval")
    model = retrieval.get("embedding_model") if isinstance(retrieval, dict) else None
    if not isinstance(model, str) or not model.strip():
        raise ValueError("agent.yaml の retrieval.embedding_model（Open WebUI の埋め込みモデル名）がありません")
    return model.strip()


def local_digests(root: Path, specs: tuple[KnowledgeSpec, ...]) -> dict[str, str]:
    """入れるファイルの（パス → SHA-256）。knowledge/ の外・存在しない・区分の名と置き場が合わないファイルは止める。

    区分はフォルダ名で、ファイル名の頭にも「<区分>__」として入れる（rules.md 規則 4。回答が資料名から区分を読む）。
    """
    digests = {}
    for spec in specs:
        for relative in spec.files:
            path = resolve_inside(root, relative)
            if (root / KNOWLEDGE_ROOT).resolve() not in path.parents:
                raise ValueError(f"knowledge {spec.name} のファイルは {KNOWLEDGE_ROOT}/ の下に置きます: {relative}")
            if not path.is_file():
                raise ValueError(f"knowledge {spec.name} のファイルがありません: {relative}")
            if not path.name.startswith(f"{path.parent.name}__"):
                raise ValueError(f"ファイル名の頭が区分（フォルダ名 {path.parent.name}）と合いません: {relative}")
            digests[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return digests


def registered_path(agent_path: str) -> str:
    return (Path(agent_path).parent / REGISTERED_FILE).as_posix()


def load_registered(text: str) -> tuple[Registered, ...]:
    data = yaml.safe_load(text) or {}
    entries = data.get("knowledge") if isinstance(data, dict) else None
    if not isinstance(entries, list):
        raise ValueError(f"{REGISTERED_FILE} の knowledge が一覧ではありません")
    try:
        return tuple(
            Registered(
                name=entry["name"],
                id=entry["id"],
                files=tuple(RegisteredFile(f["path"], f["sha256"], f["file_id"]) for f in entry.get("files") or []),
                ingest=entry.get("ingest"),
            )
            for entry in entries
        )
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError(f"{REGISTERED_FILE} の形が壊れています（{error!r}）。束の id は Open WebUI の画面で確かめてください") from error


def render_registered(registered: tuple[Registered, ...]) -> str:
    data = {
        "knowledge": [
            {
                "name": r.name,
                "id": r.id,
                **({"ingest": dict(r.ingest)} if r.ingest is not None else {}),
                "files": [{"path": f.path, "sha256": f.sha256, "file_id": f.file_id} for f in r.files],
            }
            for r in registered
        ]
    }
    return _REGISTERED_HEAD + yaml.safe_dump(data, allow_unicode=True, sort_keys=False)


def ingest_settings(retrieval: dict, embedding: dict) -> dict:
    """今の取り込みの設定（GET /api/v1/retrieval/config と /retrieval/embedding から）。欠けていれば止める。"""
    picked = {}
    for source, keys, where in ((embedding, INGEST_EMBEDDING_KEYS, "retrieval/embedding"),
                                (retrieval, INGEST_RETRIEVAL_KEYS, "retrieval/config")):
        missing = [key for key in keys if key not in source]
        if missing:
            raise ValueError(f"Open WebUI の {where} に {', '.join(missing)} がありません（版が変わった可能性）")
        picked.update({key: source[key] for key in keys})
    return picked


def ingest_differences(recorded: Optional[dict], current: dict) -> list[str]:
    """束を作ったときの取り込みの設定と今の設定の違い（「キー：前 → 今」）。"""
    if recorded is None:
        return ["記録に取り込みの設定がありません"]
    keys = list(dict.fromkeys([*recorded, *current]))
    return [f"{key}：{recorded.get(key)} → {current.get(key)}" for key in keys if recorded.get(key) != current.get(key)]


def read_registered(root: Path, agent_path: str) -> tuple[Registered, ...]:
    path = root / registered_path(agent_path)
    return load_registered(path.read_text(encoding="utf-8")) if path.exists() else ()


def find_registered(registered: tuple[Registered, ...], name: str) -> Optional[Registered]:
    return next((r for r in registered if r.name == name), None)


def model_refs(specs: tuple[KnowledgeSpec, ...], registered: tuple[Registered, ...]) -> list[dict]:
    """モデルの meta.knowledge に入れる項目。束がまだ作られていなければ止める。"""
    refs = []
    for spec in specs:
        found = find_registered(registered, spec.name)
        if found is None:
            raise ValueError(f"knowledge {spec.name} は Open WebUI にまだありません。先に scripts/build_knowledge.py で作ってください")
        refs.append({"id": found.id, "name": spec.name, "type": "collection"})
    return refs


def knowledge_refs_for(root: Path, agent_path: str, agent: dict) -> list[dict]:
    """agent.yaml の Knowledge を、登録の記録の id でモデルに紐付ける形にする。Knowledge なしなら空。"""
    specs = parse_specs(agent.get("knowledge"))
    return model_refs(specs, read_registered(root, agent_path)) if specs else []


def server_files(listing: dict) -> dict[str, dict]:
    """Open WebUI の束のファイル一覧（GET /api/v1/knowledge/{id}/files）を（ファイル名 → {sha256, file_id}）にする。"""
    items = listing.get("items") or []
    if not isinstance(listing.get("total"), int):
        raise ValueError("束のファイル一覧に total がありません。1 ページに収まったかを確かめられません")
    if listing["total"] != len(items):
        raise ValueError(f"束のファイル一覧が 1 ページに収まっていません（{len(items)}/{listing['total']} 件）")
    found: dict[str, dict] = {}
    for item in items:
        name = item.get("filename") or (item.get("meta") or {}).get("name")
        if not name:
            raise ValueError(f"束のファイル一覧に名前の無い項目があります（id {item.get('id')}）")
        if name in found:
            raise ValueError(f"束に同じ名前のファイルが 2 つあります: {name}")
        found[name] = {"sha256": (item.get("meta") or {}).get("file_hash"), "file_id": item.get("id")}
    return found


@dataclass(frozen=True)
class Comparison:
    missing: tuple[str, ...]  # リポジトリにあって束に無い（パス）
    changed: tuple[str, ...]  # 名前は同じで中身が違う（パス）
    extra: tuple[str, ...]  # 束にあってリポジトリの定義に無い（ファイル名）

    @property
    def same(self) -> bool:
        return not (self.missing or self.changed or self.extra)

    def problems(self, name: str) -> list[str]:
        return [
            *(f"knowledge {name} に {p} がありません" for p in self.missing),
            *(f"knowledge {name} の {p} はリポジトリのファイルと中身が違います" for p in self.changed),
            *(f"knowledge {name} にリポジトリの定義に無いファイルがあります: {n}" for n in self.extra),
        ]


def compare(spec: KnowledgeSpec, digests: dict[str, str], on_server: dict[str, dict]) -> Comparison:
    """束の中身（Open WebUI 側）とリポジトリのファイルを、ファイル名と指紋で照らし合わせる。"""
    wanted = {Path(path).name: path for path in spec.files}
    missing = tuple(path for name, path in wanted.items() if name not in on_server)
    changed = tuple(
        path for name, path in wanted.items() if name in on_server and on_server[name]["sha256"] != digests[path]
    )
    extra = tuple(sorted(name for name in on_server if name not in wanted))
    return Comparison(missing, changed, extra)


def header_entry(spec: KnowledgeSpec, knowledge_id: str, digests: dict[str, str], on_server: dict[str, dict]) -> dict:
    """記録の見出しに残す束の中身（照合を通った後に呼ぶ）。file_id は回答ごとの検索の確かめに使う。"""
    return {
        "name": spec.name,
        "id": knowledge_id,
        "files": [
            {"name": Path(path).name, "sha256": digests[path], "file_id": on_server[Path(path).name]["file_id"]}
            for path in spec.files
        ],
    }


def retrieval_problem(sources, entries: tuple[dict, ...]) -> Optional[str]:
    """K1 の 1 回答で、検索が照合を通った束の今のファイルから行われたかを確かめる。問題があればその説明。

    Open WebUI v0.11.3 は束の検索結果を、source にモデルへ付けた束の項目（id）、箇所ごとの metadata に file_id を入れて返す。
    検索が空振りした・別の束から来た・束の一覧に無いファイル（入れ替える前の残り）の箇所が来た、のどれかなら記録を止める。
    """
    if not sources:
        return "検索された箇所がありません（Open WebUI が資料を検索しなかったか、応答に載せなかった）"
    knowledge_ids = {entry["id"] for entry in entries}
    file_ids = {f["file_id"] for entry in entries for f in entry["files"]}
    for source in sources:
        source_id = (source.get("source") or {}).get("id")
        if source_id not in knowledge_ids:
            return f"照合した束ではない所から検索されました（id {source_id}）"
        documents, metadatas = source.get("document") or [], source.get("metadata") or []
        if not documents:
            return f"束 {source_id} から検索された箇所の本文がありません"
        if len(metadatas) != len(documents):
            return f"束 {source_id} の箇所の数と metadata の数が合いません（{len(documents)} と {len(metadatas)}）"
        stray = [(m or {}).get("file_id") for m in metadatas if (m or {}).get("file_id") not in file_ids]
        if stray:
            return f"束の今のファイルではない箇所が検索されました（file_id {stray[0]}）"
    return None
