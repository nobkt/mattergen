# Molecular Crystal Fine-tuning for MatterGen

This directory contains scripts for fine-tuning MatterGen on molecular crystal datasets for unconditional structure generation.

## Overview

The provided scripts enable you to:
1. Convert a directory of CIF files (molecular crystals) into MatterGen dataset format
2. Fine-tune a pre-trained MatterGen model for property-free molecular crystal generation
3. Generate new molecular crystal structures without property conditioning

## Scripts

### 1. `molcrystal_finetune.py`
Complete pipeline script that handles the entire process from CIF files to fine-tuned model.

**Usage:**
```bash
python molcrystal_finetune.py --molcrystal_dir ./molcrystal --output_dir ./output
```

**Arguments:**
- `--molcrystal_dir`: Directory containing CIF files (mol1.cif, mol2.cif, etc.)
- `--output_dir`: Output directory for fine-tuned model and intermediate files
- `--dataset_name`: Name for the dataset (default: molcrystal)
- `--train_split`: Fraction of data for training (default: 0.8)
- `--val_split`: Fraction of data for validation (default: 0.1)
- `--skip_dataset_creation`: Skip dataset creation and only run fine-tuning
- `--skip_finetuning`: Only create dataset, skip fine-tuning

### 2. `cif_to_dataset.py`
Utility script for converting CIF files to MatterGen dataset format only.

**Usage:**
```bash
python cif_to_dataset.py --cif_dir ./molcrystal --output_dir ./dataset --dataset_name molcrystal
```

## Prerequisites

### Directory Structure
Your molecular crystal CIF files should be organized as:
```
molcrystal/
├── mol1.cif
├── mol2.cif
├── mol3.cif
└── ...
```

### Dependencies
The scripts require the following packages (typically included in MatterGen environment):
- pandas
- pymatgen
- torch
- torch-geometric
- omegaconf
- tqdm

## Step-by-Step Usage

### Example 1: Complete Pipeline
```bash
# Run the complete pipeline from CIF files to fine-tuned model
python mattergen/scripts/molcrystal_finetune.py \
    --molcrystal_dir ./molcrystal \
    --output_dir ./molcrystal_output \
    --dataset_name my_molcrystal \
    --train_split 0.8 \
    --val_split 0.1
```

This will create:
```
molcrystal_output/
├── csv/                    # CSV files (train.csv, val.csv, test.csv)
├── cache/                  # Cached datasets
├── configs/                # Configuration files
└── model/                  # Fine-tuned model checkpoints
```

### Example 2: Dataset Creation Only
```bash
# Only create the dataset without fine-tuning
python mattergen/scripts/cif_to_dataset.py \
    --cif_dir ./molcrystal \
    --output_dir ./dataset \
    --dataset_name molcrystal
```

### Example 3: Generate Structures with Fine-tuned Model
After fine-tuning, generate new molecular crystals:
```bash
python -m mattergen.scripts.generate \
    --model_path ./molcrystal_output/model \
    --output_path ./generated_crystals \
    --num_batches 5 \
    --batch_size 32
```

## How It Works

### 1. Dataset Conversion
- Reads CIF files from the molcrystal directory
- Converts each structure to CIF string format
- Creates CSV files with material_id and cif columns
- Splits data into train/validation/test sets
- Caches datasets using MatterGen's infrastructure

### 2. Configuration Generation
- Creates data module configuration files
- Sets up fine-tuning configuration for unconditional generation
- Uses MatterGen's adapter pattern for efficient fine-tuning

### 3. Fine-tuning Process
- Loads pre-trained MatterGen model (Materials Project)
- Uses adapter-based fine-tuning for computational efficiency
- Trains model on molecular crystal dataset
- No property conditioning (unconditional generation)

## Configuration Details

### Data Module Configuration
The generated data module config includes:
- No property conditioning (`properties: []`)
- Standard transforms (symmetrize_lattice, set_chemical_system_string)
- Dataset transforms for filtering sparse properties
- Appropriate batch sizes and worker settings

### Fine-tuning Configuration
The fine-tuning config includes:
- Reduced learning rate (5e-6) for fine-tuning
- Shorter training epochs (100 vs 900 for full training)
- Adapter-based approach (not full fine-tuning)
- Uses Materials Project pre-trained model as base

## Output Structure

After running the complete pipeline:

```
output_dir/
├── csv/
│   ├── train.csv           # Training data
│   ├── val.csv             # Validation data
│   └── test.csv            # Test data
├── cache/
│   └── dataset_name/
│       ├── train/          # Cached training dataset
│       ├── val/            # Cached validation dataset
│       └── test/           # Cached test dataset
├── configs/
│   ├── dataset_name.yaml  # Data module configuration
│   └── molcrystal_finetune.yaml  # Fine-tuning configuration
└── model/
    └── [timestamp]/        # Fine-tuned model checkpoints
        ├── checkpoints/
        ├── config.yaml
        └── logs/
```

## Troubleshooting

### Common Issues

1. **ImportError for dependencies**
   ```bash
   pip install pandas pymatgen torch torch-geometric omegaconf
   ```

2. **CIF parsing errors**
   - Ensure CIF files are valid
   - Check for proper formatting and required fields
   - Invalid CIF files will be skipped with warnings

3. **Memory issues during fine-tuning**
   - Reduce batch size in configuration
   - Use fewer training epochs
   - Consider using gradient accumulation

### Configuration Customization

You can modify the generated configuration files before running fine-tuning:
- Adjust batch sizes in the data module config
- Change learning rates and epochs in the fine-tuning config
- Add or modify transforms as needed

## Advanced Usage

### Custom Training Parameters
Modify the fine-tuning configuration to adjust:
- Learning rate: `lightning_module.optimizer_partial.lr`
- Training epochs: `trainer.max_epochs`
- Batch size: `data_module.batch_size.train`

### Using Different Base Models
Change the pre-trained model in the adapter configuration:
- `adapter.pretrained_name`: Use different pre-trained models
- `adapter.model_path`: Use custom model checkpoints

## Performance Notes

- Fine-tuning typically takes significantly less time than training from scratch
- Dataset size affects training time (larger datasets take longer)
- GPU acceleration is recommended for fine-tuning
- The adapter approach requires less memory than full fine-tuning