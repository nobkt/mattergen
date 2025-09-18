#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Script for fine-tuning MatterGen on molecular crystal datasets.

This script handles the complete pipeline from CIF files in a directory
to fine-tuned model for molecular crystal structure generation without properties.

Usage:
    python molcrystal_finetune.py --molcrystal_dir ./molcrystal --output_dir ./output

The molcrystal directory should contain CIF files named mol1.cif, mol2.cif, etc.
"""

import argparse
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional

import pandas as pd
from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter

from mattergen.common.data.dataset import CrystalDataset
from mattergen.common.utils.eval_utils import extract_structures_from_folder
from mattergen.common.globals import PROJECT_ROOT

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def cif_directory_to_csv(molcrystal_dir: str, output_csv: str, train_split: float = 0.8, val_split: float = 0.1) -> None:
    """
    Convert CIF files from a directory to CSV format for dataset creation.
    
    Args:
        molcrystal_dir: Directory containing CIF files (mol1.cif, mol2.cif, etc.)
        output_csv: Base path for output CSV files (train.csv, val.csv, test.csv)
        train_split: Fraction of data for training
        val_split: Fraction of data for validation (remainder goes to test)
    """
    logger.info(f"Loading CIF files from {molcrystal_dir}")
    
    # Load all structures from the directory
    structures = extract_structures_from_folder(molcrystal_dir)
    
    if not structures:
        raise ValueError(f"No valid CIF files found in {molcrystal_dir}")
    
    logger.info(f"Found {len(structures)} structures")
    
    # Create dataframe with structures
    data = []
    for i, structure in enumerate(structures):
        # Convert structure to CIF string
        cif_writer = CifWriter(structure)
        cif_string = str(cif_writer)
        
        # Create material ID
        material_id = f"molcrystal_{i:06d}"
        
        data.append({
            'material_id': material_id,
            'cif': cif_string
        })
    
    df = pd.DataFrame(data)
    
    # Split the data
    n_total = len(df)
    n_train = int(n_total * train_split)
    n_val = int(n_total * val_split)
    
    # Shuffle and split
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    train_df = df_shuffled[:n_train]
    val_df = df_shuffled[n_train:n_train+n_val]
    test_df = df_shuffled[n_train+n_val:]
    
    # Save CSV files
    output_dir = Path(output_csv).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_csv = output_dir / "train.csv"
    val_csv = output_dir / "val.csv"
    test_csv = output_dir / "test.csv"
    
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)
    
    logger.info(f"Created datasets: Train ({len(train_df)}), Val ({len(val_df)}), Test ({len(test_df)})")
    logger.info(f"Saved CSV files: {train_csv}, {val_csv}, {test_csv}")


def create_dataset_cache(csv_dir: str, dataset_name: str, cache_dir: str) -> None:
    """
    Create cached datasets from CSV files.
    
    Args:
        csv_dir: Directory containing train.csv, val.csv, test.csv
        dataset_name: Name for the dataset
        cache_dir: Base directory for cached datasets
    """
    logger.info("Creating cached datasets from CSV files")
    
    csv_path = Path(csv_dir)
    cache_path = Path(cache_dir) / dataset_name
    
    # Process each split
    for split in ['train', 'val', 'test']:
        csv_file = csv_path / f"{split}.csv"
        if csv_file.exists():
            logger.info(f"Processing {split} dataset")
            split_cache_path = cache_path / split
            
            CrystalDataset.from_csv(
                csv_path=str(csv_file),
                cache_path=str(split_cache_path)
            )
        else:
            logger.warning(f"CSV file {csv_file} not found, skipping {split} dataset")


def create_data_module_config(dataset_name: str, cache_dir: str, config_output_path: str) -> None:
    """
    Create a data module configuration file for the molecular crystal dataset.
    
    Args:
        dataset_name: Name of the dataset
        cache_dir: Base directory containing cached datasets
        config_output_path: Path to save the data module config
    """
    config_content = f"""_target_: mattergen.common.data.datamodule.CrystDataModule
_recursive_: true
properties: []  # No properties for unconditional generation

transforms:
- _target_: mattergen.common.data.transform.symmetrize_lattice
  _partial_: true
- _target_: mattergen.common.data.transform.set_chemical_system_string
  _partial_: true

dataset_transforms: 
  - _target_: mattergen.common.data.dataset_transform.filter_sparse_properties
    _partial_: true

average_density: 0.05771451654022283  # atoms/Angstrom**3
root_dir: {cache_dir}/{dataset_name}

train_dataset:
  _target_: mattergen.common.data.dataset.CrystalDataset.from_cache_path
  cache_path: ${{data_module.root_dir}}/train
  properties: ${{data_module.properties}}
  transforms: ${{data_module.transforms}}
  dataset_transforms: ${{data_module.dataset_transforms}}

val_dataset:
  _target_: mattergen.common.data.dataset.CrystalDataset.from_cache_path
  cache_path: ${{data_module.root_dir}}/val
  properties: ${{data_module.properties}}
  transforms: ${{data_module.transforms}}
  dataset_transforms: ${{data_module.dataset_transforms}}

test_dataset:
  _target_: mattergen.common.data.dataset.CrystalDataset.from_cache_path
  cache_path: ${{data_module.root_dir}}/test
  properties: ${{data_module.properties}}
  transforms: ${{data_module.transforms}}
  dataset_transforms: ${{data_module.dataset_transforms}}

num_workers:
  train: 0
  val: 0
  test: 0

batch_size:
  train: ${{eval:'(512 // ${{trainer.accumulate_grad_batches}}) // (${{trainer.devices}} * ${{trainer.num_nodes}})'}}
  val: ${{eval:'(64 // (${{trainer.devices}} * ${{trainer.num_nodes}}))'}}
  test: ${{eval:'(64 // (${{trainer.devices}} * ${{trainer.num_nodes}}))'}}

max_epochs: 900
"""
    
    Path(config_output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(config_output_path, 'w') as f:
        f.write(config_content)
    
    logger.info(f"Created data module config: {config_output_path}")


def create_finetune_config(data_module_config_name: str, config_output_path: str) -> None:
    """
    Create a fine-tuning configuration file for molecular crystals.
    
    Args:
        data_module_config_name: Name of the data module config (without .yaml)
        config_output_path: Path to save the fine-tuning config
    """
    config_content = f"""hydra:
  run:
    dir: ${{oc.env:OUTPUT_DIR,outputs/molcrystal_finetune/${{now:%Y-%m-%d}}/${{now:%H-%M-%S}}}}

defaults:
  - data_module: {data_module_config_name}
  - trainer: default
  - lightning_module: default
  - adapter: default

# Fine-tuning specific settings
trainer:
  max_epochs: 100  # Reduced epochs for fine-tuning
  logger:
    job_type: molcrystal_finetune

lightning_module:
  optimizer_partial:
    lr: 5e-6  # Lower learning rate for fine-tuning

# Adapter settings for unconditional generation
adapter:
  pretrained_name: mattergen_mp  # Use the Materials Project pretrained model
  full_finetuning: false  # Use adapter-based fine-tuning
  property_embeddings_adapt: {{}}  # No additional properties for unconditional generation
"""
    
    Path(config_output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(config_output_path, 'w') as f:
        f.write(config_content)
    
    logger.info(f"Created fine-tuning config: {config_output_path}")


def run_finetune(config_path: str, output_dir: str) -> None:
    """
    Run the fine-tuning process using the MatterGen finetune script.
    
    Args:
        config_path: Path to the fine-tuning configuration
        output_dir: Output directory for the fine-tuned model
    """
    import subprocess
    import sys
    
    logger.info("Starting fine-tuning process")
    
    # Set environment variable for output directory
    env = os.environ.copy()
    env['OUTPUT_DIR'] = output_dir
    
    # Run the fine-tuning script
    cmd = [
        sys.executable, '-m', 'mattergen.scripts.finetune',
        f'--config-path={Path(config_path).parent}',
        f'--config-name={Path(config_path).stem}'
    ]
    
    logger.info(f"Running command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, env=env, check=True)
        logger.info("Fine-tuning completed successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Fine-tuning failed with exit code {e.returncode}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune MatterGen for molecular crystal generation"
    )
    parser.add_argument(
        "--molcrystal_dir",
        type=str,
        required=True,
        help="Directory containing CIF files (mol1.cif, mol2.cif, etc.)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for fine-tuned model and intermediate files"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="molcrystal",
        help="Name for the dataset (default: molcrystal)"
    )
    parser.add_argument(
        "--train_split",
        type=float,
        default=0.8,
        help="Fraction of data for training (default: 0.8)"
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.1,
        help="Fraction of data for validation (default: 0.1)"
    )
    parser.add_argument(
        "--skip_dataset_creation",
        action="store_true",
        help="Skip dataset creation and only run fine-tuning"
    )
    parser.add_argument(
        "--skip_finetuning",
        action="store_true",
        help="Only create dataset, skip fine-tuning"
    )
    
    args = parser.parse_args()
    
    # Create output directory structure
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_dir = output_dir / "csv"
    cache_dir = output_dir / "cache"
    config_dir = output_dir / "configs"
    model_dir = output_dir / "model"
    
    if not args.skip_dataset_creation:
        logger.info("=== STEP 1: Creating dataset from CIF files ===")
        
        # Convert CIF files to CSV
        cif_directory_to_csv(
            molcrystal_dir=args.molcrystal_dir,
            output_csv=str(csv_dir / "dataset.csv"),
            train_split=args.train_split,
            val_split=args.val_split
        )
        
        # Create cached datasets
        create_dataset_cache(
            csv_dir=str(csv_dir),
            dataset_name=args.dataset_name,
            cache_dir=str(cache_dir)
        )
        
        # Create configuration files
        data_module_config_path = config_dir / f"{args.dataset_name}.yaml"
        create_data_module_config(
            dataset_name=args.dataset_name,
            cache_dir=str(cache_dir.absolute()),
            config_output_path=str(data_module_config_path)
        )
        
        finetune_config_path = config_dir / "molcrystal_finetune.yaml"
        create_finetune_config(
            data_module_config_name=args.dataset_name,
            config_output_path=str(finetune_config_path)
        )
        
        logger.info("Dataset creation completed")
    
    if not args.skip_finetuning:
        logger.info("=== STEP 2: Fine-tuning MatterGen model ===")
        
        finetune_config_path = config_dir / "molcrystal_finetune.yaml"
        if not finetune_config_path.exists():
            raise FileNotFoundError(f"Fine-tuning config not found: {finetune_config_path}")
        
        run_finetune(
            config_path=str(finetune_config_path),
            output_dir=str(model_dir)
        )
        
        logger.info("Fine-tuning completed")
    
    logger.info("=== PIPELINE COMPLETED ===")
    logger.info(f"Output directory: {output_dir}")
    if not args.skip_dataset_creation:
        logger.info(f"Dataset cache: {cache_dir}")
        logger.info(f"Configurations: {config_dir}")
    if not args.skip_finetuning:
        logger.info(f"Fine-tuned model: {model_dir}")
    
    # Print usage instructions
    print("\n" + "="*60)
    print("FINE-TUNING COMPLETED!")
    print("="*60)
    print("\nTo generate molecular crystals with your fine-tuned model:")
    print(f"python -m mattergen.scripts.generate \\")
    print(f"    --model_path {model_dir} \\")
    print(f"    --output_path ./generated_crystals \\")
    print(f"    --num_batches 5 \\")
    print(f"    --batch_size 32")
    print("\nThis will generate unconditional molecular crystal structures")
    print("without requiring any property conditions.")
    print("="*60)


if __name__ == "__main__":
    main()