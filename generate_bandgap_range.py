#!/usr/bin/env python3
"""
Comprehensive Band Gap Crystal Generation Script for MatterGen

This script generates multiple batches of crystals across different band gap values
in the range of 0.8-1.5 eV using the pre-trained dft_band_gap model.

Usage examples:
    # Generate crystals at 5 different band gap values (0.8, 0.95, 1.1, 1.25, 1.4)
    python generate_bandgap_range.py --output_base_dir results_bandgap_range --samples_per_value 32

    # Generate crystals at specific band gap values
    python generate_bandgap_range.py --band_gaps 0.8 1.0 1.2 1.5 --output_base_dir custom_range --samples_per_value 16

バンドギャップ0.8～1.5の範囲で複数の値の結晶を生成するためのスクリプトです。
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
import numpy as np

def validate_band_gap(value):
    """Validate that band gap is in the range 0.8-1.5 eV"""
    fvalue = float(value)
    if not (0.8 <= fvalue <= 1.5):
        raise argparse.ArgumentTypeError(f"Band gap must be between 0.8 and 1.5 eV, got {fvalue}")
    return fvalue

def main():
    parser = argparse.ArgumentParser(
        description="Generate crystals across multiple band gap values using MatterGen",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--band_gaps",
        type=validate_band_gap,
        nargs="+",
        help="Specific band gap values in eV (each must be between 0.8 and 1.5). If not specified, uses 5 evenly spaced values."
    )
    
    parser.add_argument(
        "--output_base_dir",
        type=str,
        required=True,
        help="Base output directory for generated crystals (subdirectories will be created for each band gap)"
    )
    
    parser.add_argument(
        "--samples_per_value",
        type=int,
        default=32,
        help="Number of crystals to generate per band gap value (default: 32)"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Number of crystals to generate per batch (default: 16)"
    )
    
    parser.add_argument(
        "--guidance_factor",
        type=float,
        default=2.0,
        help="Diffusion guidance factor - higher values give better adherence to target property (default: 2.0)"
    )
    
    parser.add_argument(
        "--record_trajectories",
        action="store_true",
        default=False,
        help="Record denoising trajectories (default: False to save space)"
    )
    
    args = parser.parse_args()
    
    # Determine band gap values to use
    if args.band_gaps:
        band_gaps = sorted(args.band_gaps)
    else:
        # Default: 5 evenly spaced values from 0.8 to 1.5
        band_gaps = np.linspace(0.8, 1.5, 5).tolist()
        band_gaps = [round(bg, 2) for bg in band_gaps]  # Round to 2 decimal places
    
    print(f"Generating crystals for band gaps: {band_gaps} eV")
    print(f"Samples per band gap value: {args.samples_per_value}")
    print(f"Total crystals to generate: {len(band_gaps) * args.samples_per_value}")
    print(f"Base output directory: {args.output_base_dir}")
    print(f"Guidance factor: {args.guidance_factor}")
    print()
    
    # Calculate number of batches needed
    num_batches = (args.samples_per_value + args.batch_size - 1) // args.batch_size
    
    # Create base output directory
    os.makedirs(args.output_base_dir, exist_ok=True)
    
    # Generate crystals for each band gap value
    success_count = 0
    for i, band_gap in enumerate(band_gaps):
        print(f"[{i+1}/{len(band_gaps)}] Generating crystals with band gap {band_gap} eV...")
        
        # Create output directory for this band gap
        output_dir = os.path.join(args.output_base_dir, f"bandgap_{band_gap:.2f}eV")
        
        # Build command
        cmd_parts = [
            "python", "-m", "mattergen.scripts.generate",
            output_dir,
            "--pretrained-name=dft_band_gap",
            f"--batch_size={args.batch_size}",
            f"--num_batches={num_batches}",
            f"--properties_to_condition_on={{'dft_band_gap': {band_gap}}}",
            f"--diffusion_guidance_factor={args.guidance_factor}",
        ]
        
        if args.record_trajectories:
            cmd_parts.append("--record_trajectories=True")
        else:
            cmd_parts.append("--record_trajectories=False")
        
        cmd = " ".join(cmd_parts)
        print(f"  Command: {cmd}")
        
        # Execute the command using subprocess to avoid shell interpretation issues
        result = subprocess.run(cmd_parts, shell=False)
        exit_code = result.returncode
        
        if exit_code == 0:
            print(f"  ✓ Successfully generated crystals for band gap {band_gap} eV")
            success_count += 1
        else:
            print(f"  ✗ Failed to generate crystals for band gap {band_gap} eV (exit code: {exit_code})")
        
        print()
    
    # Summary
    print("=" * 60)
    print("GENERATION SUMMARY")
    print("=" * 60)
    print(f"Successfully generated crystals for {success_count}/{len(band_gaps)} band gap values")
    
    if success_count > 0:
        print(f"\nResults saved in: {args.output_base_dir}")
        print(f"Directory structure:")
        for band_gap in band_gaps:
            bg_dir = f"bandgap_{band_gap:.2f}eV"
            print(f"  {bg_dir}/")
            print(f"    ├── generated_crystals_cif.zip")
            print(f"    ├── generated_crystals.extxyz")
            if args.record_trajectories:
                print(f"    └── generated_trajectories.zip")
        
        print(f"\nTotal crystals generated: ~{success_count * args.samples_per_value}")
    
    if success_count != len(band_gaps):
        print(f"\n⚠ Warning: {len(band_gaps) - success_count} band gap values failed to generate")
        sys.exit(1)
    else:
        print("\n🎉 All band gap generations completed successfully!")

if __name__ == "__main__":
    main()