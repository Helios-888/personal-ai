# Open WebUI の全体設定の変更（2026-09-19、Phase 4 手順 3・5）

利用者の GO（2026-09-19 夜）で、Open WebUI（v0.11.3）の全体設定を 2 つ変えた。どちらも Open WebUI のすべてのモデルの資料検索に効く。
理由は設計書 `docs/specs/2026-09-19-phase4-design.md` の「着手時に判明したこと」9・13・14 と、`../2026-09-19-embedding/judgment.md`。

| 設定 | 変える前 | 変えた後 | API |
|---|---|---|---|
| 埋め込みモデル（RAG_EMBEDDING_MODEL） | `sentence-transformers/all-MiniLM-L6-v2`（英語用） | `BAAI/bge-m3` | `POST /api/v1/retrieval/embedding/update` |
| 検索の前にモデルに検索語を作らせる（ENABLE_RETRIEVAL_QUERY_GENERATION） | 有効（作業用モデルの指定なし＝問われたモデル自身が作る） | 無効（問いそのもので検索する） | `POST /api/v1/tasks/config/update` |

- 埋め込みモデルのほかの値（エンジン＝既定のローカル、バッチ 1、非同期 有効、同時数 0）は変えていない。切り替えは 101 秒で、その間 Open WebUI は応答しなかった（切り替えの処理がモデルの取得と読み込みを待つため）。
- 検索語の設定は全項目を送り返す形なので、今の値を読んで 1 項目だけ変えて送り、読み直して、変わったのがその 1 項目だけであることを確かめた。
- 変えなかったもの：RAG テンプレート（SHA-256 f2f71411…）、上位 3 件、区切り 1,000 字・重なり 100 字、見出しで分ける、リランカーなし、ハイブリッド検索 off。
- 変えた後の検索の設定の指紋（`scripts/lib/rag_settings.py` の rag_snapshot、function_calling legacy として）：d7514230d8f02a24ccaac02da567d8b32dd37e5d3bd3eead9093fec7ddb128df。変える前は f5a332b93c98495556b1eb4f1b2e099be273c57c0c1cfe0dbf928fc1cbd6bab0。

## 影響

- Open WebUI にこのプロジェクト以外の資料（Knowledge・会話に添付したファイル）があれば、英語用の埋め込みで索引が作られているので、入れ直すまで検索が効かない。
- 検索語の生成を切ったので、画面の会話で「それはなぜ？」のような続きの問いでは、前の話を踏まえた検索にならない（問いの文そのもので検索する）。

## 戻し方

管理者設定の「Documents」タブで「埋め込みモデル」を `sentence-transformers/all-MiniLM-L6-v2` に、「インターフェース」タブで「検索クエリ生成」を有効に戻す（画面の名前は v0.11.3 の公開ソースと日本語訳で確かめた）。
埋め込みを戻すと、空海の資料の束（kukai-texts）は bge-m3 で作ったものなので検索が効かなくなり、`scripts/run_eval.py` の K1 と `scripts/build_knowledge.py` は取り込みの設定の食い違いで止まる。
