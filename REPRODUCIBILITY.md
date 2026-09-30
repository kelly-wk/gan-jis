# 再現手順 / Reproducibility

## 1. 公開スモーク / Public smoke profile

検証環境は Python 3.12.14、NumPy 2.1.3、PyTorch 2.5.1 です。依存関係は固定済みで、CPU で実行できます。

The verified environment is Python 3.12.14, NumPy 2.1.3, and PyTorch 2.5.1. Dependencies are pinned, and the smoke profile runs on CPU.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
PYTHONPATH=src .venv/bin/python -m gan_jis.reproduce \
  --profile smoke \
  --seed 2025 \
  --output results/local-smoke.json
```

期待する不変条件:

- tests: 5/5 pass
- legal-row loss: `0.0`
- legal-row gradient max: `0.0`
- projected combo compliance: `1.0`
- projection idempotent: `true`
- best-of-K: non-increasing for the fixed candidate prefixes

Exact floating-point values are recorded in [`results/smoke.json`](results/smoke.json).

## 2. Notebook

[`notebooks/gan_jis_reproduction.ipynb`](notebooks/gan_jis_reproduction.ipynb) は同じスモークを呼び出します。公開版は出力、実行カウント、Colab のユーザー情報、widget state、マウントパス、リソース ID を含みません。

The public notebook is output-free by design. Run it locally to regenerate the displayed dictionary.

## 3. 非公開入力による実行確認 / Private recovered-data execution

次の入力はこのリポジトリに含まれません。

- `dataset_train.npz`, `dataset_val.npz`, `dataset_test.npz`, `dataset_meta.json`
- beam and column reference CSV files
- recovered checkpoint

```bash
PYTHONPATH=src .venv/bin/python -m gan_jis.reproduce \
  --profile recovered \
  --dataset-dir PRIVATE_DATASET_DIR \
  --beam-csv PRIVATE_BEAM_TABLE \
  --column-csv PRIVATE_COLUMN_TABLE \
  --epochs 2 \
  --batch-size 64 \
  --hidden 16 \
  --n-critic 1 \
  --seed 2025 \
  --output results/local-recovered-smoke.json
```

監査環境の CPU 実行は 287 samples を 2 epochs 完走しました。結果は [`results/recovered-data-execution-smoke.json`](results/recovered-data-execution-smoke.json) に保存しています。これは execution smoke であり、性能再現ではありません。

The historical full configuration records 600 GAN epochs, batch size 64, three 256-unit hidden layers, five critic updates per generator update, and a 150-epoch warm-up. A CUDA GPU is recommended for a full rerun. Exact wall time depends on hardware; no full corrected 600-epoch benchmark has been completed in this audit.

## 4. 既存 checkpoint の安全な検証 / Safe checkpoint verification

```bash
PYTHONPATH=src .venv/bin/python scripts/verify_private_checkpoint.py \
  --checkpoint PRIVATE_CHECKPOINT \
  --dataset-dir PRIVATE_DATASET_DIR \
  --beam-csv PRIVATE_BEAM_TABLE \
  --column-csv PRIVATE_COLUMN_TABLE \
  --seed 2025 \
  --k 32
```

The verifier uses `torch.load(..., weights_only=True)` and opens NPZ files with `allow_pickle=False`; it accesses only numeric `C` and `X` arrays. It reports target-independent projection selection separately from hidden-target oracle diagnostics.

## 5. 公開前監査 / Pre-publication audit

```bash
python3 scripts/audit_public_candidate.py .
```

私有識別子リストがある監査環境では `--forbidden-file` も指定します。スキャナーはトークン形式、メール、私有パス、Drive 直リンク、Notebook 出力、実行メタデータ、pickle 系バイナリを fail-closed で検査します。
