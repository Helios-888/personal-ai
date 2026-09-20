"""agent.yaml の knowledge を Open WebUI の Knowledge（資料の束）として作り、ファイルを入れる（Phase 4 手順 5）。

設計は docs/specs/2026-09-19-phase4-design.md「段 2」。束の id・束を作ったときの取り込みの設定・入れたファイルの指紋は、
agent.yaml と同じフォルダの knowledge-registered.yaml に書く（同期と評価がこれを読む。コミットして残す）。
モデルへの紐付けは scripts/sync_openwebui.py が行う。

- 入れる前に、Open WebUI の版が公開ソースを読んだ版と同じか、埋め込みモデルが agent.yaml の retrieval.embedding_model と
  同じかを確かめる（違う埋め込みのまま入れると、切り替えた後に全部入れ直しになる）
- 束を作ったら、ファイルを入れる前に id と取り込みの設定を記録する。途中で止まっても、走らせ直せば同じ束に残りから続く。
  ただし取り込みの設定（区切り方・埋め込み）が束を作ったときと変わっていれば止まる（1 つの束に 2 種類の区切りが混ざるため）
- 束の中のファイルがリポジトリと違う、または定義に無いファイルがあるときは止まる（入れ直しは手で決める）
- 取り込みに失敗したファイルは束に入れず、その id を示して止まる。束の外に残ったファイルは Open WebUI の画面で消せる。
  束ごと作り直すときは、Open WebUI の画面で束を消し、knowledge-registered.yaml から項目を消してから走らせ直す

使い方（llm01 のリポジトリ直下で。API キーが平文で流れるため localhost 向けに使う）:
  .venv/bin/python scripts/build_knowledge.py --dry-run   # 作る束と入れるファイルを表示するだけ
  .venv/bin/python scripts/build_knowledge.py             # 作る・入れる（CPU の埋め込みで 10 分前後）
"""
import argparse
import os
import sys
from pathlib import Path
from typing import Callable, Optional

if __package__ in (None, ""):  # スクリプトとして直接実行されたとき、リポジトリ直下を import 経路に加える
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import requests  # noqa: E402
import yaml  # noqa: E402

from scripts.lib.agent import load_agent  # noqa: E402
from scripts.lib.env import load_dotenv  # noqa: E402
from scripts.lib.knowledge import (  # noqa: E402
    KnowledgeSpec,
    Registered,
    RegisteredFile,
    compare,
    expected_embedding_model,
    find_registered,
    ingest_differences,
    ingest_settings,
    local_digests,
    parse_specs,
    read_registered,
    registered_path,
    render_registered,
    server_files,
)
from scripts.lib.openwebui_client import EXPECTED_VERSION, OpenWebUIClient, OpenWebUIError  # noqa: E402

DEFAULT_AGENT = "agents/kukai/agent.yaml"
DEFAULT_OPENWEBUI_URL = "http://localhost:3000"


class BuildStopped(RuntimeError):
    """Open WebUI 側の状態が想定と違うので止まった（終了コード 1）。"""


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="agent.yaml の knowledge を Open WebUI の Knowledge として作る")
    parser.add_argument("--agent", default=DEFAULT_AGENT, help="agent.yaml のパス（リポジトリ直下から）")
    parser.add_argument("--dry-run", action="store_true", help="作る束と入れるファイルを表示するだけ")
    parser.add_argument("--replace", action="append", default=[], metavar="PATH",
                        help="束の中の 1 点を入れ直す（中身が変わったファイル。何度でも指定できる）")
    return parser.parse_args(argv)


def run(argv: list[str], *, root: Path, env: dict, client_factory: Callable[[str, str], object]) -> int:
    args = parse_args(argv)
    api_key = env.get("OPENWEBUI_API_KEY", "")
    if not api_key:
        print("error: OPENWEBUI_API_KEY が設定されていません（.env を確認してください）", file=sys.stderr)
        return 2
    try:
        agent = load_agent(root, args.agent)
        specs = parse_specs(agent.get("knowledge"))
        if not specs:
            raise ValueError(f"{args.agent} の knowledge が空です")
        expected = expected_embedding_model(agent)
        digests = local_digests(root, specs)
        check_replace(args.replace, specs)
        read_registered(root, args.agent)  # 壊れた記録なら Open WebUI に触る前に止める
    except (ValueError, OSError, yaml.YAMLError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    client = client_factory(env.get("OPENWEBUI_URL", DEFAULT_OPENWEBUI_URL), api_key)
    try:
        ingest = check_server(client, expected)
        for spec in specs:
            build_one(spec, digests, ingest, client, root, args)
    except BuildStopped as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    if args.dry_run:
        print("dry-run: nothing sent")
    return 0


def check_replace(replace: list[str], specs: list) -> None:
    """--replace に渡せるのは、束の定義にあるファイルだけ（打ち間違いを黙って無視しない）。"""
    known = {path for spec in specs for path in spec.files}
    unknown = sorted(set(replace) - known)
    if unknown:
        raise ValueError("--replace に渡したファイルが束の定義にありません: " + "、".join(unknown))


def check_server(client, expected_embedding: str) -> dict:
    """Open WebUI の版と埋め込みモデルを確かめ、今の取り込みの設定を返す。"""
    version = client.get_version().get("version")
    if version != EXPECTED_VERSION:
        raise BuildStopped(
            f"Open WebUI の版は {version} です（挙動を確かめたのは {EXPECTED_VERSION}）。"
            "束の検索のされ方を公開ソースで確かめ直してから、EXPECTED_VERSION を改めてください"
        )
    embedding = client.get_embedding_config()
    if embedding.get("RAG_EMBEDDING_MODEL") != expected_embedding:
        raise BuildStopped(
            f"Open WebUI の埋め込みモデルは {embedding.get('RAG_EMBEDDING_MODEL')} です（agent.yaml は {expected_embedding}）。"
            "先に切り替えてください。このまま入れると、切り替えた後に入れ直しになります"
        )
    return ingest_settings(client.get_retrieval_config(), embedding)


def build_one(spec: KnowledgeSpec, digests: dict[str, str], ingest: dict, client, root: Path,
              args: argparse.Namespace) -> None:
    found = find_registered(read_registered(root, args.agent), spec.name)
    if found is None:
        print(f"plan: create knowledge {spec.name}（{len(spec.files)} ファイル）")
        for path in spec.files:
            print(f"  {path} {digests[path][:12]}")
        if args.dry_run:
            return
        found = Registered(spec.name, client.create_knowledge(spec.name, spec.description)["id"], (), ingest)
        save_record(root, args.agent, found)  # 途中で止まっても、次は同じ束に続きを入れる
        on_server: dict = {}
    else:
        differences = ingest_differences(found.ingest, ingest)
        if differences:
            raise BuildStopped(
                f"knowledge {spec.name} の取り込みの設定が、束を作ったときと違います（" + "、".join(differences) + "）。"
                "設定を戻すか、束を作り直してください"
            )
        on_server = server_files(client.get_knowledge_files(found.id))

    for path in [p for p in spec.files if p in args.replace and Path(p).name in on_server]:
        print(f"plan: replace {path}")
        if not args.dry_run:
            client.remove_file_from_knowledge(found.id, on_server[Path(path).name]["file_id"])
        on_server.pop(Path(path).name)  # 以後は「入っていない」ものとして扱い、下の流れで入れ直す

    comparison = compare(spec, digests, on_server)
    if comparison.changed or comparison.extra:
        raise BuildStopped("。".join(comparison.problems(spec.name)) + "。入れ直すかどうかを決めてから、Open WebUI 側を手で直してください")
    if not comparison.missing:
        print(f"noop: {spec.name} {found.id} はリポジトリと同じ")
        return
    for path in comparison.missing:
        print(f"plan: add {path}")
        if not args.dry_run:
            add_file(client, found.id, root / path)
    if args.dry_run:
        return

    final = server_files(client.get_knowledge_files(found.id))
    result = compare(spec, digests, final)
    if not result.same:
        raise BuildStopped("入れた後の照合で食い違いました。" + "。".join(result.problems(spec.name)))
    files = tuple(RegisteredFile(path, digests[path], final[Path(path).name]["file_id"]) for path in spec.files)
    save_record(root, args.agent, Registered(spec.name, found.id, files, found.ingest))
    print(f"done: {spec.name} {found.id}、{len(files)} ファイル、リポジトリと同じ")


def add_file(client, knowledge_id: str, path: Path) -> None:
    uploaded = client.upload_file(path.name, path.read_bytes())
    status = client.file_process_status(uploaded["id"]).get("status")
    if status != "completed":
        raise BuildStopped(
            f"{path.name} の取り込み（本文の取り出しと埋め込み）が終わっていません: status={status}。"
            f"束には入れていません（アップロードしたファイルの id {uploaded['id']}）"
        )
    client.add_file_to_knowledge(knowledge_id, uploaded["id"])
    print(f"added: {path.name}")


def save_record(root: Path, agent_path: str, entry: Registered) -> None:
    """登録の記録を書く。一時ファイルに書いてから置き換えるので、途中で落ちても前の記録が残る。"""
    current = read_registered(root, agent_path)
    if any(r.name == entry.name for r in current):
        updated = tuple(entry if r.name == entry.name else r for r in current)
    else:
        updated = (*current, entry)
    target = root / registered_path(agent_path)
    temporary = target.with_name(target.name + ".tmp")
    with open(temporary, "w", encoding="utf-8", newline="\n") as file:
        file.write(render_registered(updated))
    os.replace(temporary, target)


def main(argv: Optional[list[str]] = None, root: Optional[Path] = None) -> int:
    repo_root = Path(root) if root else Path(__file__).resolve().parents[1]
    env = {**load_dotenv(repo_root / ".env"), **os.environ}  # 環境変数が .env より優先
    try:
        return run(sys.argv[1:] if argv is None else argv, root=repo_root, env=env, client_factory=OpenWebUIClient)
    except (OpenWebUIError, requests.RequestException, OSError, yaml.YAMLError, ValueError, KeyError) as error:
        print(f"error: {error!r}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
