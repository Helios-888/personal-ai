# 評価（引き継ぎ書 §7）

設計の詳細は `docs/specs/2026-09-18-phase3-design.md`。

## 原則

- 評価セット `questions.yaml` は **Knowledge 投入前（Phase 3）に凍結** する。凍結後は変更しない（2026-09-18 凍結）。
- RAG 無しの基準値を先に取り `baseline/` に置く。条件は B0（1 文のみのシステムプロンプト）と B1（常時層のみ、Knowledge なし）。
- 変更は一度に一つだけ。実行結果は `runs/YYYYMMDD-<label>/` に置く。

## ファイル

| ファイル | 中身 | 扱い |
|---|---|---|
| `questions.yaml` | 問いの文と正解の要点（出典つき） | 完全凍結。`questions.sha256` で照合 |
| `rubric.yaml` | 判定の区分、多数決で割れたときの扱い、問いごとの誤答条件 | 凍結。1 回だけ改訂できる。`rubric.sha256` で照合 |
| `errata.yaml` | 凍結後に見つかった問いの誤り | 凍結しない。追記のみ |
| `baseline/` | RAG 無しの基準値（B0・B1） | Phase 3 で取得 |
| `runs/` | Phase 4 以降の実行結果 | |
| `phase2-voice-check/` | Phase 2 の語り口の確認。評価セットとは別物 | 凍結対象外 |

照合は `.venv/bin/python -m pytest tests/test_eval_freeze.py`、または `cd evaluations/kukai && sha256sum -c questions.sha256 rubric.sha256`。
指紋はバイト列で計算するため、`.gitattributes` で改行の自動変換を止めてある。

## 構成（30 問＋ホールドアウト 2 問）

| 種別 | 数 | 内容 |
|---|---|---|
| factual | 15 | 著作の内容、成立年、上表の相手など正解が確定するもの。目録外の実在書（『三教指帰』『性霊集』）を含む |
| attribution | 5 | 本人の言葉か後世解釈かの区別 |
| trap | 10 | 架空著作名、他者の著作、later_attribution の引用誘導、他宗・大乗一般思想の空海帰属誘導 |
| holdout | 2 | 伏せた episode（0001・0005）の「状況」節だけを与え、判断の方向が一致するかを見る。30 問の外で別集計 |

## 指標

架空引用率、引用実在率、trap 正しい拒否率、ホールドアウト判断一致率。これに factual 正答率と過剰拒否率を加える（trap の拒否率は何でも断れば上がるため、並べて見る）。
