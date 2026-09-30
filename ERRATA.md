# 重要な訂正と評価境界 / Material Errata and Evaluation Boundaries

この文書は、提出済み論文の本文を改竄するものではありません。復元コード、ログ、チェックポイント、表の監査で判明した限界を、2026 年の追記として公開するものです。

This document does not rewrite the submitted thesis. It records limitations discovered in the 2026 audit of the recovered code, logs, checkpoints, and tables.

## 訂正一覧 / Findings

### E-01 — ResearchKit の JIS 損失は実質 0 / ResearchKit JIS loss is a no-op

復元実装は `st = fake + (proj - fake).detach()` の後に `L1(st, proj)` を計算します。forward で `st == proj` となり、等号点の L1 勾配も 0 なので、生成器に JIS 近接方向の信号は流れません。`section4_A_trainlog.csv` の保存点 1, 5, 10, 20, 50, 100, 200, 300, 400 epoch で `loss_jis` はすべて `0.0` です。

The recovered straight-through expression makes the forward L1 loss and its gradient zero. The stored training log records `loss_jis=0.0` at every retained epoch. Claims that this term gradually attracted the generator toward the standard set are not supported by that run.

### E-02 — 一つの K-loop は最良候補を選ばない / One K-loop keeps the first sample

ResearchKit `eval_jis_distance` は、2 個目以降の候補に対する分岐が `else: pass` です。したがって、常に 1 個目の候補を保持し、best-of-K ではありません。

The ResearchKit evaluator never replaces the first sample, so that function cannot support a best-of-K claim.

### E-03 — 属性適合と組合せ適合は別物 / Attribute compliance is not tuple compliance

`project_attr_96` は H, B, tw, tf, D, t を個別に丸めます。それぞれのスカラ値が許容集合に入っていても、`(H,B,tw,tf)` または `(D,t)` が表の実在行である保証はありません。公開修正版は完全行に投影します。

Per-attribute rounding can create combinations absent from the table. The corrected code projects complete beam and column tuples.

### E-04 — softmin は非負の距離ではない / The recovered softmin is not a non-negative distance

`strict_v2` / `paper_v1` の `-tau*logsumexp(-d/tau)` は零点を保たず、候補点の密度による entropy bias を持ちます。合法行上の監査値は次の通りでした。

| 集合 / Set | 合法行での loss | 勾配ノルム / Gradient norm |
|---|---:|---:|
| beam | 約 `-0.312` から `0` | max `0.0605` |
| column | `-3.0607` から `-1.1481` | mean `0.2277`, max `0.5316` |

合法な `(D=150, t=6)` でも `grad_t=-0.3347` となり、最小化は `t=6` から密な `t=9` 方向へ動かします。この量を「距離」とは表記しません。修正版は、合法点で loss=0 かつ gradient=0 の exact nearest squared distance を使います。

The expression can be negative and can push a legal row toward a denser neighborhood. The corrected default is an exact nearest squared distance; a historical softmin, if ever retained, must be named `legacy_softmin` and labeled defective.

### E-05 — raw-mm ユークリッド距離の尺度問題 / Raw-mm scale dominance

H/B の数値レンジは厚さより大きく、無標準化の距離では H/B が選択を支配します。修正版は表の座標範囲で標準化し、raw-mm を旧版比較に限定します。

### E-06 — 真値による best-of-K 選択は oracle / Ground-truth best-of-K is an oracle

`legacy_pkg/train_gan_jis.py::eval_gan_recon` は、非公開の正解 `X` との L1 が最小の生成候補を K 個から選びます。実運用時に `X` は分からないため、test best-of-32 `0.0542` 系の値はオラクル診断であり、展開可能な性能指標ではありません。

The recovered evaluator selects candidates using hidden target `X`. These values are optimistic oracle diagnostics, not deployable inference performance. The corrected selector uses projection distance only and reports the oracle under a separate warning key.

### E-07 — 投影後 compliance=1 は学習成果ではない / Post-projection compliance is a guarantee

`pred_q_mm = project_x_mm(pred_mm)` の後に適合率を計算すると、投影が正常な限り 1.0 になります。これは投影演算の保証であり、generator が JIS 制約を学習した証拠ではありません。raw pre-projection combo compliance と post-projection guarantee を必ず分けます。

### E-08 — PaperPipeline はそのままでは再実行不可 / Paper pipeline is incomplete

`GAN_JIS_PaperPipeline.ipynb` は `runs/gan_plain` の不在で停止します。`paper_tools/eval_topk.py` のデフォルト checkpoint パスも、実ファイルの `checkpoints/best.pt` と一致しません。図と top-K 出力は完成していません。

### E-09 — 同一 output は独立追認ではない / Copied outputs are not independent replications

`strict_v2`, `paper_v1`, `ResearchKit/legacy_pkg` の GAN checkpoint はすべて同じ SHA-256 `827a0b5702b1a9bb4d6d69be83a27bae797550f6dd25ae37f581b493599f50c5` です。GAN metrics/preview と DNN checkpoints/metrics/predictions にも完全一致があります。パッケージ数を独立実験回数と数えてはいけません。

### E-10 — 保存表の比較は一貫しない / Stored comparisons conflict

- DNN: MAE `2.9348`, RMSE `6.5087` という復元表があります。
- 一つの表では GAN-plain best-K `2.9733`, GAN-JIS `2.9454` と小さな改善ですが、別の表では GAN-JIS `2.9909` で plain より悪化しています。
- worst-20% は DNN `3.4286` 対 GAN `3.2558` の改善表と、DNN `3.7342` 対 GAN-JIS `4.0643` の悪化表が両方残っています。
- 論文付録の DNN `9.8`、GAN-plain `13.2 / 9.5`、GAN-JIS `13.8 / 9.7` と適合率表に対応する生成ログや一意な集計コードは、復元資料から確定できませんでした。

These conflicts prevent an unqualified claim that GAN-JIS outperforms the baseline.

### E-11 — Colab の best-of-K 出典 / Likely source of the thesis best-of-K narrative

ResearchKit の no-op K 評価とは別に、`01_GAN_colab` 後半のカスタム評価は K 個を実際にサンプリングし、最小値を選びます。そのため、卒論の dproj 表はこの後期セルまたは対応する表生成コードから来た可能性が高いと推定します。ただし、そこでも距離と投影は属性単位であり、complete-tuple 適合ではありません。また K ごとに別の確率サンプルを再生成するセルがあり、同一集合の prefix による厳密な単調比較ではありません。

The later Colab evaluator samples K candidates correctly, unlike the no-op ResearchKit function, but it still uses attribute-wise distance/projection. This is an evidence-based inference about lineage, not proof of the exact source of every thesis figure.

## 論文主張の再整理 / Thesis reconciliation

| 主張 / Claim | 監査後の扱い / Audited status |
|---|---|
| Warm-up 後の JIS loss が分布を誘導 | ResearchKit run では loss/gradient が 0、strict_v2 では biased softmin。未検証 |
| best-of-K で実務的に良案を選べる | 投影距離での選択は実装可能。真値 L1 選択は oracle |
| compliance=1 は学習の成果 | 投影後は演算上の保証。raw 出力の学習証拠ではない |
| GAN-JIS は plain/DNN より一貫して良い | 保存表が矛盾し、一貫した優位は確認できない |
| 分布ピークを再現 | 定量的検定と独立再実行が不足 |

## 公開修正版での対応 / Corrected public behavior

- `exact_nearest_squared_loss`: legal row `loss == 0`, `gradient == 0`
- `JISTables.project`: complete-row projection and idempotence test
- `best_of_k_curve`: fixed candidate prefixes and target-independent score
- `candidate_selection_audit`: raw pre-projection compliance, post-projection guarantee, deployable selector, optional oracle warning are separate
- all public notebooks: no outputs, counts, identity metadata, mounted paths, resource identifiers, or credentials
