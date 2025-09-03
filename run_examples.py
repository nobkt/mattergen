#!/usr/bin/env python3
"""
Example script demonstrating band gap crystal generation

This script shows various ways to generate crystals with band gaps
in the 0.8-1.5 eV range using the provided scripts.

バンドギャップ結晶生成の使用例を示すスクリプトです。
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print('='*60)
    
    result = subprocess.run(cmd, shell=True, capture_output=False)
    
    if result.returncode == 0:
        print(f"✓ {description} completed successfully")
        return True
    else:
        print(f"✗ {description} failed with exit code {result.returncode}")
        return False

def main():
    print("MatterGen Band Gap Crystal Generation Examples")
    print("=" * 50)
    
    # Create examples directory
    examples_dir = "bandgap_examples"
    os.makedirs(examples_dir, exist_ok=True)
    print(f"Examples will be saved in: {examples_dir}/")
    
    # Example 1: Single band gap generation (small test)
    print(f"\nExample 1: Generate 4 crystals with band gap 1.0 eV (quick test)")
    cmd1 = f"python generate_bandgap_crystals.py --band_gap 1.0 --output_dir {examples_dir}/test_1_0_eV --batch_size 4 --num_batches 1"
    success1 = run_command(cmd1, "Single band gap generation")
    
    if not success1:
        print("Skipping remaining examples due to failure...")
        return
    
    # Example 2: Range generation with default values
    print(f"\nExample 2: Generate crystals across band gap range (default 5 values)")
    cmd2 = f"python generate_bandgap_range.py --output_base_dir {examples_dir}/range_default --samples_per_value 8"
    success2 = run_command(cmd2, "Range generation with defaults")
    
    # Example 3: Custom band gap values
    print(f"\nExample 3: Generate crystals at specific band gap values")
    cmd3 = f"python generate_bandgap_range.py --band_gaps 0.9 1.1 1.3 --output_base_dir {examples_dir}/custom_values --samples_per_value 8"
    success3 = run_command(cmd3, "Custom band gap values")
    
    # Example 4: Using bash script
    print(f"\nExample 4: Using bash script for quick generation")
    cmd4 = f"./run_bandgap_generation.sh 1.2 {examples_dir}/bash_example 8"
    success4 = run_command(cmd4, "Bash script generation")
    
    # Summary
    print(f"\n{'='*60}")
    print("EXAMPLES SUMMARY")
    print('='*60)
    
    examples = [
        ("Single band gap (1.0 eV)", success1),
        ("Range generation (default)", success2),
        ("Custom band gaps", success3),
        ("Bash script (1.2 eV)", success4)
    ]
    
    successful = sum(1 for _, success in examples if success)
    total = len(examples)
    
    for desc, success in examples:
        status = "✓" if success else "✗"
        print(f"  {status} {desc}")
    
    print(f"\nCompleted {successful}/{total} examples successfully")
    
    if successful > 0:
        print(f"\nGenerated crystals can be found in:")
        print(f"  {examples_dir}/")
        print(f"\nEach directory contains:")
        print(f"  - generated_crystals_cif.zip")
        print(f"  - generated_crystals.extxyz")
        print(f"  - generated_trajectories.zip (if enabled)")
        
        print(f"\nTo examine the structures:")
        print(f"  cd {examples_dir}/test_1_0_eV")
        print(f"  unzip generated_crystals_cif.zip")
        print(f"  # CIF files can be opened with visualization software like VESTA, PyMOL, or Avogadro")

if __name__ == "__main__":
    main()