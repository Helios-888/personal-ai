# personal-ai — 空海AI（Kukai AI）

llm01 上の Open WebUI に「空海AI」を RAG で構築するためのリポジトリ。
**このフォルダのファイルが唯一の正（Source of Truth）** であり、Open WebUI 上のカスタムモデル・Knowledge・Prompts は同期スクリプトによる生成物です。Open WebUI の画面で手編集しません。

## 目的（成果物は空海AIそのものではない）

1. RAG 設計と評価手法の習得（チャンク設計、検索方式の選択、評価セットによる判定、調整の打ち切り判断）
2. 「人格の再現」が知識 RAG と何が違うかの検証

詳細は `docs/パーソナルAI_引き継ぎ書.md`、Phase 0 の確認結果は `docs/phase0_基盤確認_2026-09-17.md`。

## 構成

| パス | 役割 |
|---|---|
| `config/` | 環境（llm01）とコンテキスト予算の定義 |
| `agents/kukai/` | 空海AIの人格定義（system / voice / rules / principles / episodes）とモデル依存メモ |
| `knowledge/kukai/` | 資料。`index.md` が目録層。`translations/` は Git 管理外 |
| `evaluations/kukai/` | 凍結済み評価セットと実行結果 |
| `scripts/` | 同期・投入・評価のスクリプト |
| `tests/` | スクリプトの自動テスト |
| `docs/` | 引き継ぎ書、Phase 報告、設計メモ |
| `logs/` | Git 管理外 |

## 実施フェーズ

| Phase | 内容 | 状態 |
|---|---|---|
| 0 | 基盤確認 | 完了（2026-09-17） |
| 1 | リポジトリ骨格と同期スクリプト | 進行中 |
| 2 | 声の設計（RAG 無し） | 未着手 |
| 3 | 評価セット凍結と基準値取得 | 未着手 |
| 4 | Knowledge 投入 | 未着手 |
| 5 | 調整と打ち切り判断 | 未着手 |

## 同期スクリプトの使い方

```bash
cd /opt/personal-ai
.venv/bin/python scripts/sync_openwebui.py --dry-run   # 送信前に差分だけ表示
.venv/bin/python scripts/sync_openwebui.py             # Open WebUI に登録・更新
```

認証情報は `.env`（Git 管理外）に置く。`.env.example` を参照。

## テスト

```bash
.venv/bin/python -m pytest --cov=scripts --cov-report=term-missing
```
