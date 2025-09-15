# MatterGen ファインチューニングガイド

このドキュメントでは、MatterGenで複数の物性を条件とした構造生成を行うために必要なファインチューニング方法について説明します。

## 目次

1. [概要](#概要)
2. [前提条件](#前提条件)
3. [データセットの準備](#データセットの準備)
4. [単一物性でのファインチューニング](#単一物性でのファインチューニング)
5. [複数物性でのファインチューニング](#複数物性でのファインチューニング)
6. [対象コマンドのためのファインチューニング](#対象コマンドのためのファインチューニング)
7. [生成の実行](#生成の実行)
8. [トラブルシューティング](#トラブルシューティング)
9. [ベストプラクティス](#ベストプラクティス)

## 概要

提供されたコマンド：
```bash
python mattergen/scripts/generate.py "$RESULTS_PATH" \
  --model_path="$MODEL_PATH" \
  --batch_size=16 \
  --num_batches=1 \
  --chemical_system_mode contains \
  --properties_to_condition_on="{'chemical_system': ['O'], 'energy_above_hull': 0.05, 'dft_band_gap': 1.4, 'space_group': 221}"
```

このコマンドは以下の物性で条件付けされた構造生成を行います：
- **chemical_system**: ['O'] - 酸素を含む化学系
- **energy_above_hull**: 0.05 - エネルギー安定性指標（0.05 eV/atom）
- **dft_band_gap**: 1.4 - DFT計算によるバンドギャップ（1.4 eV）
- **space_group**: 221 - 結晶の空間群対称性（Pm-3m）

## 前提条件

### システム要件
- Python 3.10以上
- CUDA対応GPU（推奨）
- 十分なディスクスペース（データセットとモデル用）

### インストール
```bash
# uvを使用したインストール（推奨）
pip install uv
uv venv .venv --python 3.10 
source .venv/bin/activate
uv pip install -e .

# Git LFSの設定
git lfs install
git lfs pull
```

## データセットの準備

### 1. データセットの取得と前処理

```bash
# MP-20データセット
git lfs pull -I data-release/mp-20/ --exclude=""
unzip data-release/mp-20/mp_20.zip -d datasets
csv-to-dataset --csv-folder datasets/mp_20/ --dataset-name mp_20 --cache-folder datasets/cache

# より大きなAlex-MP-20データセット（推奨）
git lfs pull -I data-release/alex-mp/alex_mp_20.zip --exclude=""
unzip data-release/alex-mp/alex_mp_20.zip -d datasets
csv-to-dataset --csv-folder datasets/alex_mp_20/ --dataset-name alex_mp_20 --cache-folder datasets/cache
```

### 2. 利用可能な物性の確認

データセットに含まれる物性を確認：
```bash
# MP-20の場合
cat mattergen/conf/data_module/mp_20.yaml

# Alex-MP-20の場合  
cat mattergen/conf/data_module/alex_mp_20.yaml
```

主要な物性：
- `chemical_system`: 化学系
- `energy_above_hull`: ハル上エネルギー
- `dft_band_gap`: DFTバンドギャップ
- `space_group`: 空間群
- `dft_mag_density`: 磁気密度
- `ml_bulk_modulus`: 体積弾性率（ML予測）

## 単一物性でのファインチューニング

### 基本的なファインチューニング

単一物性（例：`dft_band_gap`）でのファインチューニング：

```bash
export PROPERTY=dft_band_gap
export MODEL_NAME=mattergen_base  # または your_base_model_path

mattergen-finetune \
  adapter.pretrained_name=$MODEL_NAME \
  data_module=mp_20 \
  +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.$PROPERTY=$PROPERTY \
  ~trainer.logger \
  data_module.properties=["$PROPERTY"]
```

### カスタムベースモデルの使用

自分で訓練したベースモデルを使用する場合：

```bash
export PROPERTY=dft_band_gap
export MODEL_PATH=/path/to/your/base/model

mattergen-finetune \
  adapter.model_path=$MODEL_PATH \
  data_module=mp_20 \
  +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.$PROPERTY=$PROPERTY \
  ~trainer.logger \
  data_module.properties=["$PROPERTY"]
```

## 複数物性でのファインチューニング

### 2つの物性での例

```bash
export PROPERTY1=dft_band_gap
export PROPERTY2=energy_above_hull
export MODEL_NAME=mattergen_base

mattergen-finetune \
  adapter.pretrained_name=$MODEL_NAME \
  data_module=mp_20 \
  +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.$PROPERTY1=$PROPERTY1 \
  +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.$PROPERTY2=$PROPERTY2 \
  ~trainer.logger \
  data_module.properties=["$PROPERTY1","$PROPERTY2"]
```

### 3つ以上の物性での例

```bash
export PROPERTY1=chemical_system
export PROPERTY2=energy_above_hull  
export PROPERTY3=dft_band_gap
export MODEL_NAME=mattergen_base

mattergen-finetune \
  adapter.pretrained_name=$MODEL_NAME \
  data_module=mp_20 \
  +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.$PROPERTY1=$PROPERTY1 \
  +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.$PROPERTY2=$PROPERTY2 \
  +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.$PROPERTY3=$PROPERTY3 \
  ~trainer.logger \
  data_module.properties=["$PROPERTY1","$PROPERTY2","$PROPERTY3"]
```

## 対象コマンドのためのファインチューニング

目標の生成コマンドに必要な4つの物性すべてでファインチューニングを行います：

### ステップ1: 4物性ファインチューニング

```bash
export PROPERTY1=chemical_system
export PROPERTY2=energy_above_hull
export PROPERTY3=dft_band_gap
export PROPERTY4=space_group
export MODEL_NAME=mattergen_base
export OUTPUT_DIR=outputs/multi_property_finetuning

mattergen-finetune \
  adapter.pretrained_name=$MODEL_NAME \
  data_module=mp_20 \
  +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.$PROPERTY1=$PROPERTY1 \
  +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.$PROPERTY2=$PROPERTY2 \
  +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.$PROPERTY3=$PROPERTY3 \
  +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.$PROPERTY4=$PROPERTY4 \
  ~trainer.logger \
  data_module.properties=["$PROPERTY1","$PROPERTY2","$PROPERTY3","$PROPERTY4"] \
  hydra.run.dir=$OUTPUT_DIR
```

### ステップ2: より大きなデータセットでの訓練（推奨）

より良い結果を得るため、Alex-MP-20データセットを使用：

```bash
mattergen-finetune \
  adapter.pretrained_name=$MODEL_NAME \
  data_module=alex_mp_20 \
  +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.chemical_system=chemical_system \
  +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.energy_above_hull=energy_above_hull \
  +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.dft_band_gap=dft_band_gap \
  +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.space_group=space_group \
  ~trainer.logger \
  data_module.properties=["chemical_system","energy_above_hull","dft_band_gap","space_group"] \
  trainer.accumulate_grad_batches=4 \
  hydra.run.dir=outputs/multi_property_alex_mp
```

### ステップ3: 訓練の監視

訓練の進行状況を監視：

```bash
# ログファイルの確認
tail -f $OUTPUT_DIR/train.log

# チェックポイントの確認
ls -la $OUTPUT_DIR/checkpoints/
```

訓練は通常200エポック程度で完了し、validation lossが収束することを確認してください。

## 生成の実行

### ファインチューニングされたモデルでの生成

```bash
export MODEL_PATH="outputs/multi_property_finetuning"  # ファインチューニング後のモデルパス
export RESULTS_PATH="results/multi_property_generation"

python mattergen/scripts/generate.py "$RESULTS_PATH" \
  --model_path="$MODEL_PATH" \
  --batch_size=16 \
  --num_batches=1 \
  --chemical_system_mode contains \
  --properties_to_condition_on="{'chemical_system': ['O'], 'energy_above_hull': 0.05, 'dft_band_gap': 1.4, 'space_group': 221}"
```

### より多くのサンプル生成

```bash
python mattergen/scripts/generate.py "$RESULTS_PATH" \
  --model_path="$MODEL_PATH" \
  --batch_size=32 \
  --num_batches=5 \
  --chemical_system_mode contains \
  --properties_to_condition_on="{'chemical_system': ['O'], 'energy_above_hull': 0.05, 'dft_band_gap': 1.4, 'space_group': 221}" \
  --diffusion_guidance_factor=2.0
```

### 結果の確認

生成された構造は以下のファイルに保存されます：
- `generated_crystals_cif.zip`: CIFファイル
- `generated_crystals.extxyz`: 単一のXYZファイル
- `generated_trajectories.zip`: 生成軌跡（オプション）

## トラブルシューティング

### 一般的な問題と解決策

#### 1. メモリ不足エラー
```bash
# バッチサイズを小さくする
mattergen-finetune ... trainer.accumulate_grad_batches=8

# または
mattergen-finetune ... data_module.batch_size=64
```

#### 2. 物性データが見つからない
```bash
# データセットに含まれる物性を確認
python -c "
import pandas as pd
df = pd.read_csv('datasets/mp_20/train.csv')
print('Available properties:', df.columns.tolist())
"
```

#### 3. 収束しない訓練
```bash
# 学習率を下げる
mattergen-finetune ... lightning_module.optimizer_partial.lr=1e-6

# エポック数を増やす
mattergen-finetune ... trainer.max_epochs=400
```

#### 4. 生成品質が低い
```bash
# diffusion guidance factorを調整
--diffusion_guidance_factor=1.5  # より多様性重視
--diffusion_guidance_factor=3.0  # より条件適合重視
```

### ログファイルの確認

```bash
# 訓練ログ
grep "val_loss" $OUTPUT_DIR/train.log | tail -10

# エラーログ
grep -i error $OUTPUT_DIR/train.log
```

## ベストプラクティス

### 1. データ品質の確保
- 十分な数の訓練サンプルがあることを確認
- 物性値の分布が偏っていないことを確認
- 欠損値の処理を適切に行う

### 2. 訓練の最適化
- 段階的ファインチューニング：少ない物性から始めて徐々に追加
- 早期停止を使用してオーバーフィッティングを防ぐ
- 定期的にバリデーション損失を確認

### 3. 生成の最適化
- `chemical_system_mode`を適切に設定：
  - `exact`: 指定された元素のみ
  - `contains`: 指定された元素を含む（推奨）
- `diffusion_guidance_factor`を調整：
  - 1.0-2.0: バランス
  - 2.0-4.0: 条件適合重視
  - 0.5-1.0: 多様性重視

### 4. 計算リソースの管理
```bash
# GPU使用量の監視
nvidia-smi -l 1

# ディスクスペースの確認
df -h

# メモリ使用量の監視
free -h
```

### 5. 結果の評価
```bash
# 構造の評価
mattergen-evaluate \
  --structures_path=$RESULTS_PATH \
  --relax=True \
  --structure_matcher='disordered' \
  --save_as="$RESULTS_PATH/metrics.json"
```

### 6. モデルの保存と管理
- チェックポイントを定期的にバックアップ
- 異なる設定での複数のモデルを保持
- 訓練設定を記録して再現可能性を確保

## 高度な設定

### カスタム物性の追加

独自の物性データでファインチューニングを行う場合：

1. `mattergen/common/utils/globals.py`の`PROPERTY_SOURCE_IDS`に物性名を追加
2. データセットCSVに新しい列を追加
3. `csv-to-dataset`スクリプトを再実行
4. `mattergen/conf/lightning_module/diffusion_module/model/property_embeddings/`に設定ファイルを作成

### 完全ファインチューニング vs アダプターファインチューニング

```bash
# 完全ファインチューニング（より多くのパラメータを更新）
mattergen-finetune ... adapter.full_finetuning=true

# アダプターファインチューニング（デフォルト、効率的）
mattergen-finetune ... adapter.full_finetuning=false
```

この包括的なガイドに従うことで、複数の物性を条件とした構造生成のためのMatterGenファインチューニングを成功させることができます。