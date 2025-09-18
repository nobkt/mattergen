#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Utility script to convert a directory of CIF files to MatterGen dataset format.

This script converts CIF files (mol1.cif, mol2.cif, etc.) from a directory
into the CSV and cached dataset format used by MatterGen.

Usage:
    python cif_to_dataset.py --cif_dir ./molcrystal --output_dir ./dataset --dataset_name molcrystal
"""

import argparse
import logging
from pathlib import Path

from mattergen.scripts.molcrystal_finetune import (
    cif_directory_to_csv,
    create_dataset_cache,
    create_data_module_config
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Convert CIF files to MatterGen dataset format"
    )
    parser.add_argument(
        "--cif_dir",
        type=str,
        required=True,
        help="Directory containing CIF files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for dataset files"
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
    
    args = parser.parse_args()
    
    # Create output directory structure
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_dir = output_dir / "csv"
    cache_dir = output_dir / "cache"
    config_dir = output_dir / "configs"
    
    logger.info("=== Converting CIF files to dataset format ===")
    
    # Convert CIF files to CSV
    cif_directory_to_csv(
        molcrystal_dir=args.cif_dir,
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
    
    # Create configuration file
    data_module_config_path = config_dir / f"{args.dataset_name}.yaml"
    create_data_module_config(
        dataset_name=args.dataset_name,
        cache_dir=str(cache_dir.absolute()),
        config_output_path=str(data_module_config_path)
    )
    
    logger.info("=== Dataset conversion completed ===")
    logger.info(f"CSV files: {csv_dir}")
    logger.info(f"Cached dataset: {cache_dir}")
    logger.info(f"Data module config: {data_module_config_path}")
    
    print("\n" + "="*50)
    print("DATASET CONVERSION COMPLETED!")
    print("="*50)
    print(f"\nDataset files created in: {output_dir}")
    print(f"- CSV files: {csv_dir}")
    print(f"- Cached dataset: {cache_dir}")
    print(f"- Config file: {data_module_config_path}")
    print("\nYou can now use this dataset for fine-tuning with:")
    print(f"python molcrystal_finetune.py --skip_dataset_creation --output_dir ./finetune_output")
    print("="*50)


if __name__ == "__main__":
    main()