# 埋め込みモデルの選定（2026-09-19、Phase 4 手順 3）

Open WebUI の埋め込みモデルはいま `sentence-transformers/all-MiniLM-L6-v2`（英語用）で、漢文・日本語の検索には使えない。
Phase 0 §10-6 の候補 `BAAI/bge-m3` と `cl-nagoya/ruri-v3-310m` を、Open WebUI に触れずに llm01 の CPU で比べる。

- **問い**：`devset.yaml` の練習問題 11 問。評価の 10 問は使わない（試験問題で下見をすると本番の成績が甘く出るため）。
- **箇所**：`knowledge/kukai` の 9 点を、Open WebUI のいまの設定（見出しで分けたうえで 1,000 字・重なり 100 字）に近い形で区切る。
- **命中**：正解の行を含む箇所が上位に入ること。Open WebUI は 1 問に上位 3 件（TOP_K 3）を渡すので、上位 3 件で比べる。
- **選び方（走らせる前に決めた）**：bge-m3 を使う。ruri-v3-310m（前置きなし。Open WebUI は前置きを付けない）の上位 3 件の命中が、bge-m3 を 2 問以上上回ったときだけ ruri にする（Phase 0 の「同等なら bge-m3、ruri が明確に上回れば ruri」と、Phase 3 の「2 問以上の差を明確とする」決めに合わせた）。ruri の前置きありは参考。
- **撤退**：どちらも上位 3 件の命中が 11 問中 5 以下なら、モデルを決めずに止め、区切り方（1 行ごとの行 ID が意味の邪魔をしていないか等）から見直す。
- **環境**：リポジトリ直下の `.venv-embed`（torch CPU・sentence-transformers）と `.cache-hf`（モデルの置き場）。どちらも Git の管理外。
