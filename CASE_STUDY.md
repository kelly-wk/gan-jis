# JIS寸法制約付き鋼断面候補の生成 / JIS-Dimension-Constrained Steel-Section Candidate Generation

[紹介ページを開く / Open the presentation page](https://kelly-wk.github.io/gan-jis/)

> **方法修復済み・公開は監査要約のみ / Method repaired · public audit summary only**  
> 公開ケーススタディ / Public case study

## 概要 / Overview

制約付き生成モデルの実装と評価設計を監査し、無効だった制約損失を修復した研究プロトタイプ。

A research prototype that audits constrained generative modeling and repairs an ineffective geometry-constraint loss.

## 主なポイント / Highlights

1. **CP932を明示した入力処理により、文字コード依存の再現性問題を解消。**  
   Made input handling reproducible by explicitly supporting CP932 encoding.
2. **恒等的にゼロとなっていたJIS制約損失を修正し、勾配テストを追加。**  
   Corrected a JIS constraint loss that had been identically zero and added gradient tests.
3. **非確証的な評価結果も保持し、生成候補を構造安全性の証明として扱わない境界を明記。**  
   Retained the non-confirmatory evaluation outcome and explicitly ruled out structural-safety interpretation.

## 研究の流れ / Research Flow

| 段階 / Stage | 内容 / Evidence |
|---|---|
| **課題 / Problem** | 鋼断面候補の生成で、寸法制約が学習へ実際に作用しているかを検証する。<br>Determine whether dimensional constraints genuinely influence training when generating steel-section candidates. |
| **方法 / Method** | 条件付きWGAN-GP、制約損失、複数シードの対応比較、勾配テストを組み合わせる。<br>Combine a conditional WGAN-GP, constraint-aware loss, paired multi-seed comparisons, and gradient tests. |
| **検証 / Validation** | 文字コード処理と損失勾配を監査し、修復前後を対応のある設計で確認する。<br>Audit encoding and loss gradients, then compare pre- and post-repair behavior with a paired design. |
| **成果 / Outcome** | 実装上の欠陥は修復したが、性能改善を確証せず研究プロトタイプとして限定的に位置付ける。<br>The implementation defect was repaired, but improvement was not confirmed; the work remains a bounded research prototype. |

## 使用手法 / Methods

PyTorch, Conditional WGAN-GP, Constraint-aware loss, Paired multi-seed evaluation, Gradient testing

## 限界と適用範囲 / Limitations & Scope

- 元データ、生成物、共同作業物の再配布権が未確定であり、候補の構造性能も検証していない。  
  Redistribution rights for source data, generated artifacts, and collaborative materials remain unresolved, and structural performance is unverified.

## 公開範囲 / Publication Boundary

公開ページは方法監査と修復内容の要約のみ。元データ、コード、生成物、数値結果は非公開で、構造安全性や規格適合性の根拠を提供しない。

The public page is limited to a method-audit and repair summary. Source data, code, generated artifacts, and numeric results remain private, and no structural-safety or standards-compliance evidence is provided.

---

この文書は公開可能な範囲だけで構成されています。  
This document contains only material cleared for public presentation.
