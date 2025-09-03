# MatterGen Band Gap Crystal Generation - Complete Guide

## 概要 (Overview)
このプロジェクトは、MatterGenの事前訓練済みdft_band_gapモデルを使用して、0.8～1.5 eVの範囲で特定のバンドギャップを持つ結晶を生成するためのスクリプトとツールを提供します。

This project provides scripts and tools for generating crystals with specific band gaps in the range of 0.8-1.5 eV using MatterGen's pre-trained dft_band_gap model.

## 作成されたファイル (Created Files)

### Scripts (スクリプト)
1. **`generate_bandgap_crystals.py`** - 単一のバンドギャップ値で結晶を生成
2. **`generate_bandgap_range.py`** - 複数のバンドギャップ値で結晶を生成  
3. **`run_bandgap_generation.sh`** - シンプルなbashスクリプト
4. **`run_examples.py`** - 使用例を実行するスクリプト

### Documentation (ドキュメント)
1. **`BANDGAP_GENERATION_README.md`** - 詳細なREADME
2. **`COMMAND_REFERENCE.md`** - コマンドリファレンス
3. **`INSTALLATION_SETUP.md`** - このファイル（インストールガイド）

## インストールと設定 (Installation & Setup)

### 前提条件 (Prerequisites)
```bash
# MatterGenリポジトリがクローンされていること
git clone https://github.com/nobkt/mattergen.git
cd mattergen

# Python 3.10以上が必要
python --version  # Should be 3.10+
```

### 依存関係のインストール (Install Dependencies)
```bash
# 基本的なPythonパッケージ
pip install fire pymatgen numpy emmet-core

# または必要に応じて追加のパッケージ
pip install torch torchvision  # GPUを使用する場合
```

### ファイルの実行権限設定 (Set Execute Permissions)
```bash
chmod +x run_bandgap_generation.sh
chmod +x run_examples.py
```

## 基本的な使用方法 (Basic Usage)

### 1. 最もシンプルな方法 (Simplest Method)
```bash
# 1.0 eVのバンドギャップで16個の結晶を生成
./run_bandgap_generation.sh 1.0 results_1_0_eV 16
```

### 2. Pythonスクリプト使用 (Python Script Usage)
```bash
# 単一のバンドギャップ値
python generate_bandgap_crystals.py --band_gap 1.2 --output_dir results_1_2 --batch_size 16

# 複数のバンドギャップ値
python generate_bandgap_range.py --output_base_dir results_range --samples_per_value 32
```

### 3. 直接MatterGenコマンド使用 (Direct MatterGen Command)
```bash
python -m mattergen.scripts.generate results_bg_1_0 \
    --pretrained-name=dft_band_gap \
    --batch_size=16 \
    --num_batches=2 \
    --properties_to_condition_on="{'dft_band_gap': 1.0}" \
    --diffusion_guidance_factor=2.0
```

## パラメータ説明 (Parameter Explanation)

### バンドギャップ範囲 (Band Gap Range)
- **有効範囲**: 0.8 - 1.5 eV
- 半導体材料の実用的な範囲

### 重要なパラメータ (Important Parameters)
- `--band_gap`: 目標バンドギャップ (0.8-1.5 eV)
- `--batch_size`: バッチサイズ (GPUメモリに依存)
- `--guidance_factor`: 誘導係数 (1.0-5.0推奨)
- `--output_dir`: 出力ディレクトリ

## 生成されるファイル (Generated Files)

各実行で以下のファイルが生成されます：
- `generated_crystals_cif.zip`: 個別のCIFファイル
- `generated_crystals.extxyz`: 全構造を含む単一ファイル  
- `generated_trajectories.zip`: ノイズ除去軌跡（オプション）

## 用途別の例 (Application-Specific Examples)

### 太陽電池材料 (Solar Cell Materials)
```bash
python generate_bandgap_range.py --band_gaps 1.0 1.1 1.2 1.3 1.4 1.5 --output_base_dir solar_materials --samples_per_value 50
```

### LED材料 (LED Materials)  
```bash
python generate_bandgap_range.py --band_gaps 0.8 0.9 1.0 1.1 1.2 --output_base_dir led_materials --samples_per_value 50
```

### 小規模テスト (Small Scale Test)
```bash
python generate_bandgap_crystals.py --band_gap 1.0 --output_dir test --batch_size 4 --num_batches 1
```

## トラブルシューティング (Troubleshooting)

### 一般的な問題 (Common Issues)

1. **"No module named 'torch'"**
   ```bash
   pip install torch torchvision
   ```

2. **"CUDA out of memory"**
   ```bash
   # バッチサイズを減らす
   --batch_size=8
   ```

3. **"Band gap validation error"**
   ```bash
   # バンドギャップが0.8-1.5の範囲内であることを確認
   ```

4. **"No module named 'emmet'"**
   ```bash
   pip install emmet-core
   ```

### 動作確認 (Verification)
```bash
# スクリプトが正常に動作するかテスト
python generate_bandgap_crystals.py --help
python generate_bandgap_range.py --help
./run_bandgap_generation.sh
```

### 最小テスト (Minimal Test)
```bash
# 最小限のテスト実行
python generate_bandgap_crystals.py --band_gap 1.0 --output_dir minimal_test --batch_size 1 --num_batches 1
```

## パフォーマンス最適化 (Performance Optimization)

### GPU使用時 (When Using GPU)
```bash
# 大きなバッチサイズで高速化
--batch_size=32

# メモリ不足の場合は小さく
--batch_size=8
```

### ストレージ最適化 (Storage Optimization)
```bash
# 軌跡記録を無効化してディスク使用量を削減
--record_trajectories=False
```

## 例の実行 (Running Examples)
```bash
# 提供されている例を実行
python run_examples.py
```

## 高度な使用法 (Advanced Usage)

### 系統的な研究 (Systematic Study)
```bash
# 多数のバンドギャップ値で系統的に生成
for bg in 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5; do
    ./run_bandgap_generation.sh $bg "systematic_${bg}" 25
done
```

### 誘導係数の影響調査 (Guidance Factor Study)
```bash
# 異なる誘導係数での比較
for guidance in 1.0 2.0 3.0 4.0; do
    python generate_bandgap_crystals.py --band_gap 1.0 --output_dir "guidance_${guidance}" --guidance_factor $guidance
done
```

## サポートと参考文献 (Support & References)

### MatterGen Citation
```bibtex
@article{yang2025mattergen,
  title={MatterGen: a generative model for inorganic materials design},
  author={Yang, Claudio Zeni and others},
  journal={Nature},
  year={2025}
}
```

### 追加情報 (Additional Information)
- MatterGen GitHub: https://github.com/microsoft/mattergen
- バンドギャップ理論に関する参考文献は材料科学の教科書を参照

## 注意事項 (Important Notes)

1. **モデルの制限**: 20原子以下のユニットセル
2. **元素の制限**: 原子番号84以下の非放射性元素
3. **計算資源**: GPU推奨（CPUでも動作可能だが遅い）
4. **出力品質**: 誘導係数が高いほど目標特性に近いが多様性が減る

## 連絡先 (Contact)
問題や質問がある場合は、MatterGenの公式GitHubリポジトリでIssueを作成してください。