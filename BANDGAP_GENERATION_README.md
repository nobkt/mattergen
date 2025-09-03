# Band Gap Crystal Generation Scripts

This directory contains scripts and tools for generating crystals with specific band gaps in the range of 0.8-1.5 eV using MatterGen's pre-trained `dft_band_gap` model.

## バンドギャップ結晶生成スクリプト

このディレクトリには、MatterGenの事前訓練済み`dft_band_gap`モデルを使用して、0.8～1.5 eVの範囲で特定のバンドギャップを持つ結晶を生成するためのスクリプトとツールが含まれています。

## Available Scripts

### 1. `generate_bandgap_crystals.py` - Single Band Gap Generation
Generate crystals with a specific band gap value.

```bash
# Generate 16 crystals with band gap 1.0 eV
python generate_bandgap_crystals.py --band_gap 1.0 --output_dir results_bg_1.0 --batch_size 16

# Generate 32 crystals with band gap 1.2 eV and high guidance
python generate_bandgap_crystals.py --band_gap 1.2 --output_dir results_bg_1.2 --batch_size 16 --num_batches 2 --guidance_factor 3.0
```

**Parameters:**
- `--band_gap`: Target band gap in eV (required, must be 0.8-1.5)
- `--output_dir`: Output directory (required)
- `--batch_size`: Crystals per batch (default: 16)
- `--num_batches`: Number of batches (default: 1)
- `--guidance_factor`: Diffusion guidance factor (default: 2.0)
- `--record_trajectories`: Record denoising trajectories (default: True)

### 2. `generate_bandgap_range.py` - Multiple Band Gap Generation
Generate crystals across multiple band gap values.

```bash
# Generate crystals at 5 evenly spaced band gap values (0.8, 0.95, 1.1, 1.25, 1.4)
python generate_bandgap_range.py --output_base_dir results_bandgap_range --samples_per_value 32

# Generate crystals at specific band gap values
python generate_bandgap_range.py --band_gaps 0.8 1.0 1.2 1.5 --output_base_dir custom_range --samples_per_value 16
```

**Parameters:**
- `--band_gaps`: Specific band gap values (optional, defaults to 5 evenly spaced values)
- `--output_base_dir`: Base output directory (required)
- `--samples_per_value`: Crystals per band gap value (default: 32)
- `--batch_size`: Crystals per batch (default: 16)
- `--guidance_factor`: Diffusion guidance factor (default: 2.0)
- `--record_trajectories`: Record trajectories (default: False to save space)

### 3. `run_bandgap_generation.sh` - Simple Bash Script
Easy-to-use bash script for quick generation.

```bash
# Generate 32 crystals with band gap 1.0 eV
./run_bandgap_generation.sh 1.0 results_1_0_eV 32

# Generate 16 crystals with band gap 1.2 eV and custom guidance
./run_bandgap_generation.sh 1.2 results_1_2_eV 16 3.0
```

**Usage:** `./run_bandgap_generation.sh <band_gap_eV> <output_directory> [num_crystals] [guidance_factor]`

## Direct Command Usage

You can also use the MatterGen generation command directly:

```bash
# Basic usage
python -m mattergen.scripts.generate results_bandgap_1_0 \
    --pretrained-name=dft_band_gap \
    --batch_size=16 \
    --num_batches=2 \
    --properties_to_condition_on="{'dft_band_gap': 1.0}" \
    --diffusion_guidance_factor=2.0

# Or using the mattergen-generate command (if installed)
mattergen-generate results_bandgap_1_0 \
    --pretrained-name=dft_band_gap \
    --batch_size=16 \
    --num_batches=2 \
    --properties_to_condition_on="{'dft_band_gap': 1.0}" \
    --diffusion_guidance_factor=2.0
```

## Output Files

Each generation run creates the following files in the output directory:

- `generated_crystals_cif.zip`: ZIP file containing individual CIF files for each generated structure
- `generated_crystals.extxyz`: Single file containing all generated structures as frames
- `generated_trajectories.zip`: (Optional) ZIP file containing denoising trajectories for each structure

## Parameters Explanation

### Band Gap Range
- **Valid range**: 0.8 - 1.5 eV
- This range corresponds to semiconductors with band gaps suitable for various applications

### Guidance Factor
- **Default**: 2.0
- **Range**: 0.0 (unconditional) to higher values
- Higher values increase adherence to target properties but may reduce diversity
- Recommended range: 1.0 - 5.0

### Batch Size
- **Default**: 16
- Adjust based on GPU memory availability
- Larger batch sizes are more efficient but require more memory

## Example Workflows

### Quick Single Generation
```bash
# Generate 16 crystals with 1.0 eV band gap
./run_bandgap_generation.sh 1.0 quick_test 16
```

### Comprehensive Study
```bash
# Generate crystals across the full range
python generate_bandgap_range.py \
    --output_base_dir comprehensive_study \
    --samples_per_value 50 \
    --guidance_factor 2.5
```

### Custom Band Gap Values
```bash
# Generate crystals at specific interesting band gaps
python generate_bandgap_range.py \
    --band_gaps 0.9 1.1 1.3 \
    --output_base_dir silicon_alternatives \
    --samples_per_value 100 \
    --batch_size 20
```

## Requirements

- MatterGen environment properly installed
- Access to the `dft_band_gap` pre-trained model
- GPU recommended for faster generation
- Python packages: fire, pymatgen, numpy

## Troubleshooting

### Common Issues

1. **"No module named 'torch'"**: Install PyTorch and other MatterGen dependencies
2. **CUDA out of memory**: Reduce batch_size
3. **Band gap validation error**: Ensure band gap is between 0.8 and 1.5 eV
4. **Model download failure**: Check internet connection and HuggingFace access

### Performance Tips

- Use larger batch sizes if GPU memory allows
- Set `record_trajectories=False` to save disk space when not needed
- Use appropriate guidance factors (2.0-3.0) for good property adherence

## Citation

If you use these scripts in your research, please cite the MatterGen paper:

```bibtex
@article{yang2025mattergen,
  title={MatterGen: a generative model for inorganic materials design},
  author={Yang, Claudio Zeni and others},
  journal={Nature},
  year={2025}
}
```