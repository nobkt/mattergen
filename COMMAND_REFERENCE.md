# Band Gap Crystal Generation - Command Reference

## Quick Start Commands / クイックスタートコマンド

### 最もシンプルな方法 (Simplest Method)
```bash
# 1.0 eVのバンドギャップで16個の結晶を生成
./run_bandgap_generation.sh 1.0 results_1_0_eV 16
```

### Python Scripts

#### 1. Single Band Gap Generation
```bash
# Basic usage
python generate_bandgap_crystals.py --band_gap 1.0 --output_dir results_1_0

# Generate more crystals
python generate_bandgap_crystals.py --band_gap 1.2 --output_dir results_1_2 --batch_size 16 --num_batches 4

# High guidance for better property adherence
python generate_bandgap_crystals.py --band_gap 0.9 --output_dir results_0_9 --guidance_factor 3.0
```

#### 2. Multiple Band Gap Values
```bash
# Default: 5 evenly spaced values (0.8, 0.95, 1.1, 1.25, 1.4)
python generate_bandgap_range.py --output_base_dir full_range --samples_per_value 32

# Custom band gap values
python generate_bandgap_range.py --band_gaps 0.8 1.0 1.2 1.5 --output_base_dir custom_study --samples_per_value 50

# Large study with high guidance
python generate_bandgap_range.py --output_base_dir large_study --samples_per_value 100 --guidance_factor 2.5
```

### Direct MatterGen Commands

#### Basic Direct Usage
```bash
# Using Python module
python -m mattergen.scripts.generate results_bg_1_0 \
    --pretrained-name=dft_band_gap \
    --batch_size=16 \
    --num_batches=2 \
    --properties_to_condition_on="{'dft_band_gap': 1.0}" \
    --diffusion_guidance_factor=2.0

# If mattergen-generate is installed
mattergen-generate results_bg_1_0 \
    --pretrained-name=dft_band_gap \
    --batch_size=16 \
    --num_batches=2 \
    --properties_to_condition_on="{'dft_band_gap': 1.0}" \
    --diffusion_guidance_factor=2.0
```

## Parameter Combinations / パラメータの組み合わせ

### Small Test (Quick)
```bash
# 4 crystals for quick testing
python generate_bandgap_crystals.py --band_gap 1.0 --output_dir test --batch_size 4 --num_batches 1
```

### Medium Study
```bash
# 32 crystals per band gap value
python generate_bandgap_range.py --band_gaps 0.8 1.0 1.2 1.5 --output_base_dir medium_study --samples_per_value 32
```

### Large Study
```bash
# 100 crystals per band gap value across full range
python generate_bandgap_range.py --output_base_dir large_study --samples_per_value 100 --batch_size 20
```

### High Precision Study
```bash
# High guidance factor for better property adherence
python generate_bandgap_range.py --output_base_dir high_precision --samples_per_value 64 --guidance_factor 4.0
```

## Specific Use Cases / 特定の用途

### Solar Cell Applications (1.0-1.5 eV)
```bash
python generate_bandgap_range.py --band_gaps 1.0 1.1 1.2 1.3 1.4 1.5 --output_base_dir solar_materials --samples_per_value 50
```

### LED Applications (0.8-1.2 eV)
```bash
python generate_bandgap_range.py --band_gaps 0.8 0.9 1.0 1.1 1.2 --output_base_dir led_materials --samples_per_value 50
```

### Wide Band Gap Study
```bash
python generate_bandgap_range.py --band_gaps 1.2 1.3 1.4 1.5 --output_base_dir wide_bandgap --samples_per_value 75
```

### Narrow Band Gap Study
```bash
python generate_bandgap_range.py --band_gaps 0.8 0.85 0.9 0.95 1.0 --output_base_dir narrow_bandgap --samples_per_value 75
```

## Performance Optimization / パフォーマンス最適化

### High Memory GPU
```bash
# Large batch sizes for faster generation
python generate_bandgap_crystals.py --band_gap 1.0 --output_dir results --batch_size 32 --num_batches 4
```

### Limited Memory GPU
```bash
# Smaller batch sizes to avoid OOM
python generate_bandgap_crystals.py --band_gap 1.0 --output_dir results --batch_size 8 --num_batches 8
```

### Storage Optimization
```bash
# Disable trajectory recording to save disk space
python generate_bandgap_range.py --output_base_dir results --samples_per_value 100 --record_trajectories
```

## Advanced Configurations / 高度な設定

### Different Guidance Factors
```bash
# Low guidance (more diverse, less precise)
python generate_bandgap_crystals.py --band_gap 1.0 --output_dir low_guidance --guidance_factor 1.0

# Medium guidance (balanced)
python generate_bandgap_crystals.py --band_gap 1.0 --output_dir med_guidance --guidance_factor 2.0

# High guidance (more precise, less diverse)
python generate_bandgap_crystals.py --band_gap 1.0 --output_dir high_guidance --guidance_factor 4.0
```

### Batch Size Studies
```bash
# Compare different batch sizes
for bs in 8 16 32; do
    python generate_bandgap_crystals.py --band_gap 1.0 --output_dir "batch_${bs}" --batch_size $bs --num_batches 2
done
```

## Troubleshooting Commands / トラブルシューティング

### Test Installation
```bash
# Test if scripts work
python generate_bandgap_crystals.py --help
python generate_bandgap_range.py --help
./run_bandgap_generation.sh
```

### Minimal Test
```bash
# Smallest possible test
python generate_bandgap_crystals.py --band_gap 1.0 --output_dir minimal_test --batch_size 1 --num_batches 1
```

### Check Generated Files
```bash
# List generated files
ls -la results_directory/
unzip -l results_directory/generated_crystals_cif.zip
```

## Environment Variables / 環境変数

```bash
# Set common parameters
export BANDGAP_OUTPUT_BASE="my_bandgap_study"
export BANDGAP_SAMPLES=50
export BANDGAP_GUIDANCE=2.5

# Use in commands
python generate_bandgap_range.py --output_base_dir $BANDGAP_OUTPUT_BASE --samples_per_value $BANDGAP_SAMPLES --guidance_factor $BANDGAP_GUIDANCE
```

## Batch Processing Scripts / バッチ処理スクリプト

### Generate Multiple Studies
```bash
#!/bin/bash
# Generate crystals for different guidance factors
for guidance in 1.0 2.0 3.0 4.0; do
    python generate_bandgap_crystals.py \
        --band_gap 1.0 \
        --output_dir "guidance_study_${guidance}" \
        --guidance_factor $guidance \
        --batch_size 16 \
        --num_batches 2
done
```

### Systematic Band Gap Sweep
```bash
#!/bin/bash
# Generate crystals at many band gap values
for bg in 0.8 0.85 0.9 0.95 1.0 1.05 1.1 1.15 1.2 1.25 1.3 1.35 1.4 1.45 1.5; do
    ./run_bandgap_generation.sh $bg "systematic_bg_${bg}" 20
done
```