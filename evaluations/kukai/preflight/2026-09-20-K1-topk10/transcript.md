# 評価の記録 K1（2026-09-20）

- 条件：K1
- 経路：openwebui（http://localhost:3000/api/chat/completions）、モデル kukai-ai
- システムプロンプト：2373 トークン、SHA-256 cfe0dae2d92b0771eed357cd1c53ded83aa42a6d5de074a88ad9ddd752e06a0d。要求には含めない（Open WebUI が登録済みのものを先頭に足す）。
- questions.yaml の SHA-256：2e2ca6f789ced133b35b075243318acdeabedc26d590f7df41c0c831a3d317de
- temperature 0.7、max_tokens 2000、1 問 × 1 回
- 定義：agents/kukai/agent.yaml
- Git：17b56f549d8adcd329881ed0783fa0bafbb2fb71。未コミットの変更：M scripts/lib/openwebui_client.py
- 回答 1 件、出力枠切れ（finish=length）0/1
- Knowledge：kukai-texts（10 ファイル、id 91a0e27d-6f21-4a52-acf1-b03f122f8fbf）
- 検索の設定：埋め込み BAAI/bge-m3、上位 10 件、区切り 1000 字・重なり 100 字、検索語の生成 off、function_calling legacy、SHA-256 9378e505e1df921b553c21ca74d2966756ffd9153c1942b38845234636d0fb64

## F13 [factual]

> 最澄から経典を貸してほしいと頼まれて、貸したことはありますか。

### 1 回目（14.1 秒、269 字、入力 11729 トークン、出力 196 トークン、finish=stop）

弘仁4年（813年）11月、最澄から『理趣釈経』を借用したいと求められたことはある。

しかし、わたしはこれを貸さなかった。

『性霊集』巻十に収められた返書の中で、わたしは「秘蔵の奥旨は、文を得ることを貴ばず。ただ心を以って心に伝うるなり」と述べている。

また、「文はこれ糟粕、文はこれ瓦礫、糟粕と瓦礫を受くれば、則ち粋実と至実とを失う。真を棄てて偽を拾うは、愚人の法なり」とも記している。

修行を経ずに経典を得ることは、三昧耶（誓約）を越えることにあたると警告し、貸し出しを断ったのである。

――出典：『性霊集』（わたしの著作）

検索された箇所：later-attribution__御遺告.md（T2431_.77.0414a11–T2431_.77.0414a22）、later-attribution__御遺告.md（T2431_.77.0410a02–T2431_.77.0410a29）、later-attribution__御遺告.md（T2431_.77.0408b24–T2431_.77.0408c20）、later-attribution__御遺告.md（T2431_.77.0411c13–T2431_.77.0412a10）、later-attribution__御遺告.md（T2431_.77.0411a19–T2431_.77.0411b17）、biography__年譜.md、later-attribution__御遺告.md（T2431_.77.0412b06–T2431_.77.0412c03）、later-attribution__御遺告.md（T2431_.77.0408b01–T2431_.77.0408b25）、later-attribution__御遺告.md（T2431_.77.0412c02–T2431_.77.0412c29）、later-attribution__御遺告.md（T2431_.77.0408c19–T2431_.77.0409a16）
