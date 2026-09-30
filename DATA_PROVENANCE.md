# データ出典と公開境界 / Data Provenance and Release Boundary

## 復元データ / Recovered arrays

| Split | `C` | `X` | `Wmin` |
|---|---:|---:|---:|
| train | `(287, 5)` float32 | `(287, 96)` float32 | `(287, 1)` float32 |
| validation | `(32, 5)` float32 | `(32, 96)` float32 | `(32, 1)` float32 |
| test | `(106, 5)` float32 | `(106, 96)` float32 | `(106, 1)` float32 |

`C` は 5 条件、`X` は 16 部材位置×6 属性の 96 次元表現、`Wmin` は重量系の値です。各 NPZ には object dtype の `id` もありますが、公開コードは `allow_pickle=False` で `C` / `X` だけを読み、`id` を読み込みません。

`C` contains five conditions; `X` contains 16 positions by six section attributes; `Wmin` is a weight-related value. Every NPZ also contains an object-dtype `id` field. The public runner opens the archive with `allow_pickle=False`, accesses only numeric arrays, and never loads or publishes `id`.

## 権利判定 / Rights determination

現時点の証拠だけでは、`C` / `X` / `Wmin` がすべてユーザー自身の生成物であること、および第三者データや研究室資産を含まず再配布できることを証明できません。したがって、数値部分だけを抽出した場合でも公開しません。

The available evidence does not prove that all `C`, `X`, and `Wmin` values were generated solely by the user or that no lab/third-party rights attach to them. Removing `id` would reduce privacy risk but would not establish redistribution rights. No numeric dataset is therefore included.

## JIS/reference tables

断面表は規格又は規格由来の資料から作成された可能性があり、表値の再配布許諾は復元ファイル内に見つかりません。公開リポジトリには同梱しません。`illustrative_tables()` の小さな値はアルゴリズ試験用の合成 fixture であり、公式 JIS データではありません。

The recovered section tables may derive from a standard or standard-based reference, and no redistribution grant was found. They are excluded. The small values returned by `illustrative_tables()` are synthetic test fixtures, not official JIS data.

## 公開に必要な追加証拠 / Evidence needed before release

1. データ生成者と元ソフトウェアの確定。
2. 研究室・大学・第三者の権利確認。
3. 匿名数値配布に対する明示的な許可とライセンス。
4. object `id` を一度もデシリアライズしない、新規数値専用エクスポート。
5. schema、split、hash、単位、生成手順を含む data card。
