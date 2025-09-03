#!/bin/bash
# 
# Simple bash script to generate crystals with band gaps in the 0.8-1.5 eV range
# バンドギャップが0.8～1.5の範囲の結晶を生成するためのシンプルなスクリプト
#
# Usage examples:
#   ./run_bandgap_generation.sh 1.0 results_1_0_eV 32
#   ./run_bandgap_generation.sh 1.2 results_1_2_eV 16
#

# Check if enough arguments are provided
if [ $# -lt 2 ]; then
    echo "Usage: $0 <band_gap_eV> <output_directory> [num_crystals] [guidance_factor]"
    echo ""
    echo "Arguments:"
    echo "  band_gap_eV      - Band gap value in eV (must be between 0.8 and 1.5)"
    echo "  output_directory - Directory to save generated crystals"
    echo "  num_crystals     - Number of crystals to generate (default: 32)"
    echo "  guidance_factor  - Diffusion guidance factor (default: 2.0)"
    echo ""
    echo "Examples:"
    echo "  $0 1.0 results_1_0_eV 32 2.0"
    echo "  $0 1.2 results_1_2_eV 16"
    echo ""
    exit 1
fi

# Parse arguments
BAND_GAP=$1
OUTPUT_DIR=$2
NUM_CRYSTALS=${3:-32}  # Default to 32 crystals
GUIDANCE_FACTOR=${4:-2.0}  # Default guidance factor

# Validate band gap range
if (( $(echo "$BAND_GAP < 0.8" | bc -l) )); then
    echo "Error: Band gap must be >= 0.8 eV, got $BAND_GAP"
    exit 1
fi

if (( $(echo "$BAND_GAP > 1.5" | bc -l) )); then
    echo "Error: Band gap must be <= 1.5 eV, got $BAND_GAP"
    exit 1
fi

# Calculate batch configuration (use batch size of 16)
BATCH_SIZE=16
NUM_BATCHES=$(( (NUM_CRYSTALS + BATCH_SIZE - 1) / BATCH_SIZE ))

echo "=========================================="
echo "MatterGen Band Gap Crystal Generation"
echo "=========================================="
echo "Band gap target: $BAND_GAP eV"
echo "Output directory: $OUTPUT_DIR"
echo "Number of crystals: $NUM_CRYSTALS"
echo "Batch size: $BATCH_SIZE"
echo "Number of batches: $NUM_BATCHES"
echo "Guidance factor: $GUIDANCE_FACTOR"
echo ""

# Run the generation command
echo "Starting crystal generation..."
echo ""

python -m mattergen.scripts.generate \
    "$OUTPUT_DIR" \
    --pretrained-name=dft_band_gap \
    --batch_size=$BATCH_SIZE \
    --num_batches=$NUM_BATCHES \
    --properties_to_condition_on="{'dft_band_gap': $BAND_GAP}" \
    --diffusion_guidance_factor=$GUIDANCE_FACTOR \
    --record_trajectories=True

# Check if generation was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Successfully generated crystals!"
    echo "Results saved in: $OUTPUT_DIR"
    echo ""
    echo "Generated files:"
    echo "  - generated_crystals_cif.zip    (CIF files for each structure)"
    echo "  - generated_crystals.extxyz     (All structures in one file)"
    echo "  - generated_trajectories.zip    (Denoising trajectories)"
    echo ""
    echo "You can extract and examine the structures using:"
    echo "  unzip $OUTPUT_DIR/generated_crystals_cif.zip"
    echo ""
else
    echo ""
    echo "❌ Crystal generation failed!"
    echo "Please check the error messages above."
    exit 1
fi