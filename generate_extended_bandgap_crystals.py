#!/usr/bin/env python3
"""
Extended Band Gap Crystal Generation Script for MatterGen

This script generates crystals with comprehensive band gap conditioning including:
- Band gap energy value (0.8-1.5 eV) 
- Band gap center value (VBM-CBM center position)
- Transition type (direct=1 or indirect=0)

Usage examples:
    # Generate crystals with band gap 1.0 eV, center at 0.5 eV, direct transition
    python generate_extended_bandgap_crystals.py \
        --band_gap 1.0 \
        --band_gap_center 0.5 \
        --is_direct 1 \
        --output_dir results_bg_extended \
        --batch_size 16

    # Generate crystals with band gap 1.2 eV, center at -0.2 eV, indirect transition  
    python generate_extended_bandgap_crystals.py \
        --band_gap 1.2 \
        --band_gap_center -0.2 \
        --is_direct 0 \
        --output_dir results_bg_indirect \
        --batch_size 16

バンドギャップ、バンドギャップセンター値、および直接/間接遷移を含む包括的なバンドギャップ条件で結晶を生成するためのスクリプトです。
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

def validate_band_gap(value):
    """Validate that band gap is in the range 0.8-1.5 eV"""
    fvalue = float(value)
    if not (0.8 <= fvalue <= 1.5):
        raise argparse.ArgumentTypeError(f"Band gap must be between 0.8 and 1.5 eV, got {fvalue}")
    return fvalue

def validate_is_direct(value):
    """Validate that is_direct is 0 or 1"""
    ivalue = int(value)
    if ivalue not in [0, 1]:
        raise argparse.ArgumentTypeError(f"is_direct must be 0 (indirect) or 1 (direct), got {ivalue}")
    return ivalue

def main():
    parser = argparse.ArgumentParser(
        description="Generate crystals with extended band gap properties using MatterGen",
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
        "--band_gap_center", 
        type=float, 
        help="Band gap center value in eV (VBM-CBM center position)"
    )
    
    parser.add_argument(
        "--is_direct", 
        type=validate_is_direct, 
        help="Transition type: 0 for indirect, 1 for direct"
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
    
    # Build properties dictionary
    properties = {"dft_band_gap": args.band_gap}
    
    if args.band_gap_center is not None:
        properties["dft_band_gap_center"] = args.band_gap_center
        
    if args.is_direct is not None:
        properties["dft_band_gap_is_direct"] = args.is_direct
    
    # Total number of crystals to generate
    total_crystals = args.batch_size * args.num_batches
    print(f"Generating {total_crystals} crystals with properties:")
    print(f"  Band gap: {args.band_gap} eV")
    if args.band_gap_center is not None:
        print(f"  Band gap center: {args.band_gap_center} eV") 
    if args.is_direct is not None:
        transition_type = "direct" if args.is_direct == 1 else "indirect"
        print(f"  Transition type: {transition_type}")
    print(f"Output directory: {args.output_dir}")
    print(f"Guidance factor: {args.guidance_factor}")
    
    # Create the mattergen-generate command
    cmd_parts = [
        "python", "-m", "mattergen.scripts.generate",
        args.output_dir,
        "--pretrained-name=dft_band_gap",  # Use the base band gap model
        f"--batch_size={args.batch_size}",
        f"--num_batches={args.num_batches}",
        f"--properties_to_condition_on={properties}",
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
    
    # Execute the command using subprocess to avoid shell interpretation issues
    result = subprocess.run(cmd_parts, shell=False)
    exit_code = result.returncode
    
    if exit_code == 0:
        print(f"\n✓ Successfully generated crystals with extended band gap properties")
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