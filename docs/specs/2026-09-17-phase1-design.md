# Phase 1 設計：リポジトリ骨格と同期スクリプト（2026-09-17）

前提：`docs/パーソナルAI_引き継ぎ書.md` §3・§4・§9、`docs/phase0_基盤確認_2026-09-17.md` §2（10-4）。

## 目的

`agents/kukai/` のファイルを正とし、Open WebUI にカスタムモデル「空海」を **スクリプトで** 登録・更新できるようにする。Phase 1 の完了条件は「`agents/kukai/` が同期スクリプト経由で Open WebUI に登録される」こと。

## 置き場所

- 実体：`/home/aiadmin/personal-ai`（sudo 不要。Windows から `Z:\personal-ai`）。
- `/opt/personal-ai` はシンボリックリンク（引き継ぎ書のパスを維持）。
- GitHub private リポジトリへ push。認証は llm01 で生成した SSH 鍵を GitHub に登録する方式。

## 同期スクリプト `scripts/sync_openwebui.py`

- 入力：`agents/kukai/agent.yaml`。`system_prompt.parts` に列挙したファイルをその順に連結してシステムプロンプトにする（区切りは空行 2 つ）。
- トークン数：llama-swap の `/upstream/<model>/tokenize` で数える（本番モデルと同じ数え方）。到達できないときは文字数 ÷ 1.5 の概算に切り替えて警告を出す。`budget_tokens` 超過は警告（同期は止めない）。
- Open WebUI API（`/api/v1/`）：`GET models/model?id=<id>` で存在確認。あれば `POST models/model/update?id=<id>`、なければ `POST models/create`。送る `ModelForm` は `{id, base_model_id, name, meta{description, knowledge, filterIds}, params{system, temperature…}, is_active}`（v0.11.3 は `access_grants`。省略時は本人のみ）。
- 冪等：既存モデルの `params.system`・`name`・`base_model_id`・`meta.description` が同じなら更新を送らず「変更なし」と表示する。
- `--dry-run`：送信せず、現在のシステムプロンプトとの差分（unified diff）とトークン数だけ表示する。
- 認証：`.env` の `OPENWEBUI_API_KEY`（管理者の API キー）。未設定なら即エラー。

### 構成（小さなファイルに分ける）

| ファイル | 役割 |
|---|---|
| `scripts/sync_openwebui.py` | CLI。引数解釈と表示のみ |
| `scripts/lib/prompt_builder.py` | parts の連結、予算判定（純粋関数） |
| `scripts/lib/tokens.py` | llama-swap tokenize 呼び出しと概算フォールバック |
| `scripts/lib/openwebui_client.py` | Open WebUI REST の薄いラッパー（get_model / create_model / update_model） |
| `scripts/lib/sync.py` | 「作成か更新か変更なしか」を決めて実行する本体 |

### テスト（pytest、カバレッジ 80% 以上）

- prompt_builder：連結順、末尾改行の正規化、予算超過の判定。
- tokens：HTTP 失敗時の概算フォールバックと警告。
- openwebui_client：`requests` を差し替えて、パス・ヘッダ・ボディの形を検証。404 → None。
- sync：存在しない → create、存在し差分あり → update、差分なし → no-op、dry-run → 送信ゼロ。

## 人手が必要な手順（順に 1 手ずつ）

1. 完了：`/opt/personal-ai` リンク作成。
2. Open WebUI 管理画面 → Settings → General で API Keys を有効化 → Settings → Account → API Keys で発行 → `Z:\personal-ai\.env` に `OPENWEBUI_API_KEY=...` を記入。
3. GitHub で空の private リポジトリ `personal-ai` を作成し、llm01 の公開鍵を GitHub アカウントに登録。

## 成功基準

- 同期を 2 回続けて実行し、2 回目が「変更なし」になる。
- Open WebUI のモデル一覧に「空海」が現れ、チャットで選べる。
- GitHub の private リポジトリに push が届き、`git log` と GitHub の画面が一致する。

## やらないこと（Phase 1 では）

- `build-knowledge.py`（Phase 4）、`run-eval.py`（Phase 3）は必要になった時点で作る。
- voice.md / principles.md の本文執筆（Phase 2）。
