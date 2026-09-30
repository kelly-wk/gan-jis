# JIS 制約付き鋼断面候補生成 / JIS-Constrained Steel-Section Candidate Generation

GAN-JIS 卒業研究の、出典を追跡可能にした監査済み再現候補です。復元した実験には評価漏洩と制約損失の問題があったため、公開コードでは完全な表行投影と正確な最近傍二乗損失を使用します。

This is an audited, provenance-aware reproduction candidate for the GAN-JIS graduation research project. Because the recovered experiment contains evaluation leakage and constraint-loss defects, the public code uses complete-row projection and an exact nearest squared loss.

[監査レポート / Audit report](REPORT.md) · [再現手順 / Reproducibility](REPRODUCIBILITY.md) · [重要な訂正 / Errata](ERRATA.md) · [論文公開ノート / Thesis publication note](THESIS_PUBLICATION_NOTE.md) · [データ境界 / Data provenance](DATA_PROVENANCE.md) · [ライセンス範囲 / License scope](LICENSE_STATUS.md)

## 一目でわかる内容 / At a glance

| 項目 / Item | 内容 / Detail |
|---|---|
| 問題 / Problem | 5 条件から 96 次元の鋼断面候補を生成 / Generate 96-dimensional steel-section candidates from five conditions |
| 手法 / Method | Conditional WGAN-GP, complete-tuple projection, exact nearest squared loss |
| 制約 / Constraint | 梁 `(H, B, tw, tf)` と柱 `(D, t)` を、独立な属性値ではなく許可表の完全行に投影 |
| 言語・技術 / Stack | Python, NumPy, PyTorch, unittest, GitHub Actions |
| 公開境界 / Public boundary | コード、合成テスト、監査文書、プライバシー処理済み論文。実データ、JIS 表、元ノート、チェックポイントは非公開 |

## 確認済みの成果 / Verified outcomes

- 5 件のユニットテストが成功し、合法表行の損失と勾配がともに 0 になることを確認しました。
- 合成スモークテストで、投影後の完全行適合率は `1.0`、投影は冪等でした。
- 同じ候補集合の prefix を使う best-of-K は、`K=1` の `0.389616` から `K=16` の `0.035011` まで非増加でした。
- 復元済みの数値データ 287 件に対する 2 epoch CPU ランが成功しました。これは実行性のスモークテストであり、論文の性能再現ではありません。
- 既存チェックポイントは `weights_only=True` で読み込み、106 条件×32 候補の forward 検証が成功しました。

The smoke results are algorithm checks, not model-quality claims. See [`results/`](results/) for machine-readable records and [`ERRATA.md`](ERRATA.md) before interpreting any historical number.

## クイックスタート / Quick start

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
PYTHONPATH=src .venv/bin/python -m gan_jis.reproduce --profile smoke
```

出力付きノートブックの代わりに、公開用 Notebook は実行履歴と個人情報を除去しています: [`notebooks/gan_jis_reproduction.ipynb`](notebooks/gan_jis_reproduction.ipynb)。

## リポジトリ構成 / Repository map

```text
src/gan_jis/        corrected projection, loss, models, runners
tests/              mathematical and evaluation-boundary tests
notebooks/          sanitized, output-free public notebook
results/            deterministic audit records
scripts/            privacy scan and private-input verifier
docs/               compact project page
thesis/             privacy-redacted submitted thesis (read ERRATA first)
```

## 論文について / About the thesis

投稿時の論文本文は歴史的資料として保存し、結論を後から書き換えていません。**PDF を開く前に必ず [THESIS_PUBLICATION_NOTE.md](THESIS_PUBLICATION_NOTE.md) と [ERRATA.md](ERRATA.md) を読んでください。** その後、[プライバシー処理済み公開候補 PDF](thesis/graduation-thesis-public-candidate.pdf) を参照できます。

The submitted thesis is preserved as a historical document; its conclusions were not silently rewritten. **Read the [publication note](THESIS_PUBLICATION_NOTE.md) and [errata](ERRATA.md) before opening the [privacy-redacted public thesis PDF](thesis/graduation-thesis-public-candidate.pdf).**

## ライセンス / License

本リポジトリの独自・修正コードと文書は [MIT License](LICENSE) です。論文、データ、JIS 由来表、第三者資料は対象外です。詳細は [LICENSE_STATUS.md](LICENSE_STATUS.md) を参照してください。

Independently authored and corrected code and documentation are released under the [MIT License](LICENSE). The thesis, data, JIS-derived tables, and third-party materials are excluded; see [LICENSE_STATUS.md](LICENSE_STATUS.md).

## English summary

The recovered project is materially useful, but its historical outputs are not a clean performance benchmark. The ResearchKit JIS loss evaluates to zero, one K-loop retains the first sample, a later loss is a biased soft minimum, and one reported best-of-K metric selects against hidden ground truth. This repository keeps those facts visible and provides a small corrected implementation with tests. Private research data and standard-derived tables remain excluded until provenance and redistribution rights are documented.
