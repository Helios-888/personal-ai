# 評価の記録 K1（2026-09-19）

- 条件：K1
- 経路：openwebui（http://localhost:3000/api/chat/completions）、モデル kukai-ai
- システムプロンプト：2297 トークン、SHA-256 9af4fcb9e5435e12e7f630d952561559d2058fafd634b0af31f68fb748f677c6。要求には含めない（Open WebUI が登録済みのものを先頭に足す）。
- questions.yaml の SHA-256：2e2ca6f789ced133b35b075243318acdeabedc26d590f7df41c0c831a3d317de
- temperature 0.7、max_tokens 2000、2 問 × 1 回
- 定義：agents/kukai/agent.yaml
- Git：4dea956e1337e7f4fda8abfb625152bf178194b3（定義とコードに未コミットの変更なし）
- 回答 1 件、出力枠切れ（finish=length）0/1
- Knowledge：kukai-texts（9 ファイル、id 91a0e27d-6f21-4a52-acf1-b03f122f8fbf）
- 検索の設定：埋め込み BAAI/bge-m3、上位 3 件、区切り 1000 字・重なり 100 字、検索語の生成 off、function_calling legacy、SHA-256 d7514230d8f02a24ccaac02da567d8b32dd37e5d3bd3eead9093fec7ddb128df

途中で中断（1/2 回答まで）：F02 1 回目：検索された箇所がありません（Open WebUI が資料を検索しなかったか、応答に載せなかった）

## F02 [factual]

> 『即身成仏義』に説かれる四種曼荼羅の名を、四つすべて挙げてください。

### 1 回目（7.4 秒、242 字、入力 2337 トークン、出力 191 トークン、finish=stop）

『即身成仏義』に説かれる四曼、すなわち四種の曼荼羅は、身曼荼羅、語曼荼羅、意曼荼羅、羯磨曼荼羅である。

身曼荼羅とは、仏の姿形そのものであり、語曼荼羅は仏の教えである真言や経文を指す。意曼荼羅は仏の心、すなわち智慧や慈悲の根本であり、羯磨曼荼羅は仏が衆生を救済するために働く行い、つまり儀式や行法を意味する。

この四つは、六大、すなわち地・水・火・風・空・識という宇宙の根本要素から生じ、互いに離れず、即身で仏になる道を示すものである。

――出典：『即身成仏義』（わたしの著作）
