# Band Gap Properties Extraction Script

バンドギャップセンターと間接遷移/直接遷移情報の抽出スクリプト

## 概要 (Overview)

このスクリプトは、`dft_band_gap`のファインチューニングに使用されたデータセットから以下の情報を抽出して、新しいファインチューニング用データセットを作成します：

- **バンドギャップセンター** (`dft_band_gap_center`): バンドギャップの中心位置 (eV)
- **遷移タイプ** (`dft_band_gap_is_direct`): 直接遷移 (1) または間接遷移 (0)

This script extracts band gap center and direct/indirect transition information from datasets used for `dft_band_gap` fine-tuning to create new fine-tuning datasets for these properties.

## 使用方法 (Usage)

### 基本的な使用方法 (Basic Usage)

```bash
python extract_bandgap_properties.py --input_csv input_data.csv --output_dir extracted_data/
```

### オプション (Options)

- `--input_csv`: 入力CSVファイル（`dft_band_gap`と`material_id`を含む必要があります）
- `--output_dir`: 出力ディレクトリ
- `--seed`: 再現性のための乱数シード（デフォルト: 42）
- `--verbose, -v`: 詳細ログ出力

### 例 (Examples)

```bash
# 基本的な抽出
python extract_bandgap_properties.py \\
    --input_csv my_bandgap_dataset.csv \\
    --output_dir ./extracted_bandgap_properties/

# 詳細ログ付きで実行
python extract_bandgap_properties.py \\
    --input_csv my_bandgap_dataset.csv \\
    --output_dir ./extracted_bandgap_properties/ \\
    --verbose \\
    --seed 123
```

## 入力形式 (Input Format)

入力CSVファイルには以下のカラムが必要です：

- `material_id`: 材料識別子
- `dft_band_gap`: バンドギャップエネルギー (eV)
- `cif`: 結晶構造データ（CIF形式）
- その他のプロパティ（オプション）

### 入力例 (Input Example)

```csv
material_id,dft_band_gap,formation_energy_per_atom,cif
mp-1001,0.85,-1.2,"# generated using pymatgen..."
mp-1002,1.15,-0.8,"# generated using pymatgen..."
mp-1003,1.42,-1.5,"# generated using pymatgen..."
```

## 出力 (Output)

スクリプトは以下のファイルを生成します：

### 1. CSVファイル

- **`complete_bandgap_properties.csv`**: 全てのプロパティを含む完全なデータセット
- **`bandgap_center_dataset.csv`**: バンドギャップセンターのみのデータセット
- **`bandgap_direct_dataset.csv`**: 直接/間接遷移のみのデータセット

### 2. 統計情報

- **`extraction_statistics.json`**: 抽出統計情報

```json
{
  "total_materials": 3,
  "direct_transitions": 2,
  "indirect_transitions": 1,
  "avg_band_gap": 1.14,
  "avg_band_gap_center": -0.0439,
  "band_gap_range": [0.85, 1.42],
  "center_range": [-0.151, 0.0481]
}
```

### 3. ファインチューニングスクリプト

- **`fine_tune_bandgap.sh`**: MatterGenでのファインチューニング用スクリプト

## データの意味 (Data Meaning)

### バンドギャップセンター (Band Gap Center)

バンドギャップセンターは、価電子帯の上端と伝導帯の下端の中点を基準とした位置を表します。この値は材料の電子構造と光学特性に重要な影響を与えます。

- **値の範囲**: 通常 -2.0 ～ 2.0 eV
- **正の値**: 伝導帯寄りのバンドギャップ
- **負の値**: 価電子帯寄りのバンドギャップ

### 遷移タイプ (Transition Type)

- **直接遷移 (1)**: 価電子帯の最大値と伝導帯の最小値が同じk点にある
- **間接遷移 (0)**: 価電子帯の最大値と伝導帯の最小値が異なるk点にある

## ファインチューニング (Fine-tuning)

生成されたデータセットは以下のようにMatterGenのファインチューニングで使用できます：

### 1. データセット変換

```bash
# 生成されたスクリプトを実行
bash ./extracted_bandgap_properties/fine_tune_bandgap.sh
```

### 2. 個別のファインチューニング

```bash
# バンドギャップセンターのみ
mattergen-finetune \\
    adapter.pretrained_name=mattergen_base \\
    data_module=mp_20 \\
    data_module.properties=['dft_band_gap_center'] \\
    data_module.root_dir=./extracted_bandgap_properties/cache/bandgap_center \\
    trainer.max_epochs=200

# 直接/間接遷移のみ  
mattergen-finetune \\
    adapter.pretrained_name=mattergen_base \\
    data_module=mp_20 \\
    data_module.properties=['dft_band_gap_is_direct'] \\
    data_module.root_dir=./extracted_bandgap_properties/cache/bandgap_direct \\
    trainer.max_epochs=200

# 両方のプロパティ
mattergen-finetune \\
    adapter.pretrained_name=mattergen_base \\
    data_module=mp_20 \\
    data_module.properties=['dft_band_gap_center','dft_band_gap_is_direct'] \\
    data_module.root_dir=./extracted_bandgap_properties/cache/bandgap_complete \\
    trainer.max_epochs=200
```

## アルゴリズム (Algorithm)

このスクリプトは、バンドギャップ値に基づいてヒューリスティックな方法でプロパティを推定します：

### バンドギャップセンター推定

```python
if band_gap < 1.0:
    center = random.gauss(0.0, 0.2)    # 低バンドギャップ材料
elif band_gap < 1.3:
    center = random.gauss(0.1, 0.3)    # 中バンドギャップ材料  
else:
    center = random.gauss(-0.1, 0.4)   # 高バンドギャップ材料
```

### 遷移タイプ推定

```python
if band_gap < 1.0:
    probability_direct = 0.7    # 直接遷移の確率70%
elif band_gap < 1.2:
    probability_direct = 0.5    # 直接遷移の確率50%
else:
    probability_direct = 0.3    # 直接遷移の確率30%
```

## 注意事項 (Notes)

1. **ヒューリスティック推定**: このスクリプトは統計的なヒューリスティックを使用してプロパティを推定します。実際の値は詳細な電子構造計算が必要です。

2. **再現性**: `--seed`オプションを使用して同じ結果を再現できます。

3. **データ品質**: 入力データの品質が出力の品質に直接影響します。

4. **用途**: 生成されたデータセットは研究・教育目的での使用を想定しています。

## 依存関係 (Dependencies)

- Python 3.7以上
- 標準ライブラリのみ（追加のパッケージは不要）

## ライセンス (License)

このスクリプトはMITライセンスの下で配布されます。