#!/usr/bin/env python3
"""
Band Gap Crystal Generation Script for MatterGen

This script generates crystals with band gaps in the range of 0.8-1.5 eV
using the pre-trained dft_band_gap model.

Usage examples:
    # Generate 16 crystals with band gap 1.0 eV
    python generate_bandgap_crystals.py --band_gap 1.0 --output_dir results_bg_1.0 --batch_size 16

    # Generate 32 crystals with band gap 1.2 eV with high guidance
    python generate_bandgap_crystals.py --band_gap 1.2 --output_dir results_bg_1.2 --batch_size 16 --num_batches 2 --guidance_factor 3.0

バンドギャップが0.8～1.5の範囲の結晶を生成するためのスクリプトです。
"""

import argparse
import os
import sys
from pathlib import Path

def validate_band_gap(value):
    """Validate that band gap is in the range 0.8-1.5 eV"""
    fvalue = float(value)
    if not (0.8 <= fvalue <= 1.5):
        raise argparse.ArgumentTypeError(f"Band gap must be between 0.8 and 1.5 eV, got {fvalue}")
    return fvalue

def main():
    parser = argparse.ArgumentParser(
        description="Generate crystals with specific band gaps using MatterGen",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--band_gap", 
        type=validate_band_gap, 
        required=True,
        help="Target band gap in eV (must be between 0.8 and 1.5)"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for generated crystals"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Number of crystals to generate per batch (default: 16)"
    )
    
    parser.add_argument(
        "--num_batches",
        type=int,
        default=1,
        help="Number of batches to generate (default: 1)"
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
        default=True,
        help="Record denoising trajectories (default: True)"
    )
    
    args = parser.parse_args()
    
    # Total number of crystals to generate
    total_crystals = args.batch_size * args.num_batches
    print(f"Generating {total_crystals} crystals with band gap {args.band_gap} eV")
    print(f"Output directory: {args.output_dir}")
    print(f"Guidance factor: {args.guidance_factor}")
    
    # Create the mattergen-generate command
    cmd_parts = [
        "python", "-m", "mattergen.scripts.generate",
        args.output_dir,
        "--pretrained-name=dft_band_gap",
        f"--batch_size={args.batch_size}",
        f"--num_batches={args.num_batches}",
        f"--properties_to_condition_on={{'dft_band_gap': {args.band_gap}}}",
        f"--diffusion_guidance_factor={args.guidance_factor}",
    ]
    
    if args.record_trajectories:
        cmd_parts.append("--record_trajectories=True")
    else:
        cmd_parts.append("--record_trajectories=False")
    
    cmd = " ".join(cmd_parts)
    print(f"\nExecuting command:")
    print(cmd)
    print()
    
    # Execute the command
    exit_code = os.system(cmd)
    
    if exit_code == 0:
        print(f"\n✓ Successfully generated crystals with band gap {args.band_gap} eV")
        print(f"  Results saved in: {args.output_dir}")
        print(f"  Generated files:")
        print(f"    - generated_crystals_cif.zip")
        print(f"    - generated_crystals.extxyz")
        if args.record_trajectories:
            print(f"    - generated_trajectories.zip")
    else:
        print(f"\n✗ Failed to generate crystals (exit code: {exit_code})")
        sys.exit(1)

if __name__ == "__main__":
    main()