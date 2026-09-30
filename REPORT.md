# 監査済み再現レポート / Audited Reproduction Report

監査日 / Audit date: 2026-09-30

## 結論 / Outcome

卒業研究のデータ、コード、Colab、チェックポイント、出力表、論文を復元し、実行可能性を確認しました。ただし、復元された結果はそのままでは性能証拠として公開できません。方法と評価に複数の重要な欠陥があり、出力の一部は異なるパッケージ間でバイト単位で同一でした。

We recovered the graduation-research data, code, Colabs, checkpoints, result tables, and thesis and verified that the core assets execute. The historical results are not publishable as clean performance evidence: the method and evaluation contain material defects, and some outputs are byte-identical across nominally different packages.

## 復元ライン / Recovered lineage

| 系統 / Artifact | 役割 / Role | 判定 / Assessment |
|---|---|---|
| `GAN_raw.ipynb` | 生データから 96 次元 NPZ への前処理 | 最も早い主要証拠の一つ / early primary evidence |
| `GAN.ipynb` | `strict_v2` の実行用ランチャ | 既存パッケージを呼ぶ追認実行 / launcher, not independent source |
| `GAN_JIS_PaperPipeline.ipynb` | `paper_v1` の図表化 | パスと必要 run の不整合で未完走 / broken pipeline |
| `01_GAN_colab.ipynb` | end-to-end 訓練・評価・卒論表 | ResearchKit notebook の拡張実行版 / strongest experiment narrative evidence |
| `ResearchKit/notebooks/01_end2end_colab.ipynb` | 小型 end-to-end 手順 | 16 コードセル中 9 が `01_GAN_colab` と完全一致 |
| `strict` → `strict_v2` | テーブル投影を追加 | 後期の修正パッケージと判断 |
| `paper_v1` | `strict_v2` + 論文用ツール | 多くのコードと実験出力を継承 |
| `ResearchKit/legacy_pkg` | 旧版同梱 | 主要ソースは `strict_v2` とバイト単位で一致 |
| `ResearchKit/src/gan_jis` | 別のモジュール化実装 | 独立書き直しに見えるが、ファイルだけで著作過程は断定不可 |

ファイルの時刻、セル一致、ハッシュに基づく系譜です。これは著作者身元の法的証明ではありません。

## 修正版の設計 / Corrected design

1. 梁は `(H,B,tw,tf)`、柱は `(D,t)` の完全な許可行に投影します。
2. 既存の `-tau*logsumexp(-d/tau)` はデフォルトから外し、`torch.cdist(...).square().min(...)` による正確な最近傍二乗損失を用います。
3. 寸法レンジで座標を標準化し、H/B が厚さを圧倒することを防ぎます。raw-mm 距離は旧版比較だけに残します。
4. best-of-K は同じ候補集合の prefix を使い、非公開の教師値ではなく投影距離で選びます。
5. 生の候補の combo compliance、投影後の保証、オラクル診断を別のキーで報告します。

## 実行結果 / Execution results

### 合成アルゴリズム・スモーク

Seed 2025、12 条件×16 候補で実行しました。

| 指標 / Metric | 結果 / Result |
|---|---:|
| 投影前距離の平均 | 0.3896156847 |
| 投影後距離の平均 | 0.0 |
| 投影後 complete-tuple compliance | 1.0 |
| 投影の冪等性 | true |
| best-of-1 / 2 / 4 / 8 / 16 | 0.389616 / 0.351372 / 0.302820 / 0.212260 / 0.035011 |
| 合法行の loss / 最大勾配 | 0.0 / 0.0 |

### 復元データの実行性スモーク

CPU、seed 2025、train 287 件、2 epochs、batch 64、hidden 16、critic update 1 で完走しました。exact JIS loss は epoch 1 で `0.0162236016`、epoch 2 で `0.0162409164` でした。2 epoch は学習効果を評価する長さではなく、数値データから勾配更新までの経路確認です。

### 既存チェックポイント

SHA-256 `827a0b5702b1a9bb4d6d69be83a27bae797550f6dd25ae37f581b493599f50c5` を `weights_only=True` で読み込み、106 条件×32 候補を検証しました。target-independent な raw-mm tuple L2 は top-1 `137.3548279`、best-of-32 `90.6296616`。真値を使うオラクル L1 は新規実行 `0.0549478009`、保存値 `0.0542332977` でした。後者は推論時に実現できない診断値です。

## Public-data boundary

The recovered dataset has numeric `C`, `X`, and `Wmin` arrays, but the available evidence does not prove a user-generated origin or a right to redistribute them. Its `id` field is an object array and is neither loaded nor published. The standard-derived section tables also lack a documented redistribution grant. The repository therefore ships only synthetic fixtures and private-input instructions. See [DATA_PROVENANCE.md](DATA_PROVENANCE.md).

## Interpretation

The corrected package demonstrates a sound constraint primitive and an executable training path. It does not establish that GAN-JIS outperforms a deterministic baseline, reproduces the submitted thesis figures, or generates structurally feasible designs. Structural analysis, load combinations, code checks, and a target-independent utility/ranking model remain outside the recovered evidence.
