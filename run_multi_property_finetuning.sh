#!/bin/bash

# MatterGen Multi-Property Fine-tuning Script
# This script demonstrates how to fine-tune MatterGen for the specific generation command:
# python mattergen/scripts/generate.py "$RESULTS_PATH" --model_path="$MODEL_PATH" --batch_size=16 --num_batches=1 --chemical_system_mode contains --properties_to_condition_on="{'chemical_system': ['O'], 'energy_above_hull': 0.05, 'dft_band_gap': 1.4, 'space_group': 221}"

set -e  # Exit on any error

echo "=== MatterGen Multi-Property Fine-tuning Script ==="
echo "Target properties: chemical_system, energy_above_hull, dft_band_gap, space_group"
echo ""

# Configuration
BASE_MODEL="mattergen_base"
DATASET="mp_20"  # Use "alex_mp_20" for larger dataset
OUTPUT_DIR="outputs/multi_property_finetuning_$(date +%Y%m%d_%H%M%S)"
BATCH_SIZE=64
MAX_EPOCHS=200
LEARNING_RATE=5e-6

# Properties for the target generation command
PROPERTY1="chemical_system"
PROPERTY2="energy_above_hull"
PROPERTY3="dft_band_gap"
PROPERTY4="space_group"

echo "Configuration:"
echo "  Base model: $BASE_MODEL"
echo "  Dataset: $DATASET"
echo "  Output directory: $OUTPUT_DIR"
echo "  Properties: $PROPERTY1, $PROPERTY2, $PROPERTY3, $PROPERTY4"
echo ""

# Step 1: Prepare dataset if not already done
echo "Step 1: Preparing dataset..."
if [ ! -d "datasets/cache/$DATASET" ]; then
    echo "Dataset not found. Preparing $DATASET dataset..."
    
    if [ "$DATASET" = "mp_20" ]; then
        echo "Downloading MP-20 dataset..."
        git lfs pull -I data-release/mp-20/ --exclude=""
        unzip -o data-release/mp-20/mp_20.zip -d datasets
        csv-to-dataset --csv-folder datasets/mp_20/ --dataset-name mp_20 --cache-folder datasets/cache
    elif [ "$DATASET" = "alex_mp_20" ]; then
        echo "Downloading Alex-MP-20 dataset..."
        git lfs pull -I data-release/alex-mp/alex_mp_20.zip --exclude=""
        unzip -o data-release/alex-mp/alex_mp_20.zip -d datasets
        csv-to-dataset --csv-folder datasets/alex_mp_20/ --dataset-name alex_mp_20 --cache-folder datasets/cache
    fi
else
    echo "Dataset $DATASET already prepared."
fi

# Step 2: Verify properties are available
echo ""
echo "Step 2: Verifying properties are available in dataset..."
python3 -c "
import pandas as pd
import sys

try:
    df = pd.read_csv('datasets/${DATASET}/train.csv')
    available_props = df.columns.tolist()
    required_props = ['${PROPERTY1}', '${PROPERTY2}', '${PROPERTY3}', '${PROPERTY4}']
    
    print('Available properties:', len(available_props))
    missing_props = [prop for prop in required_props if prop not in available_props]
    
    if missing_props:
        print('ERROR: Missing properties in dataset:', missing_props)
        print('Available properties:', available_props)
        sys.exit(1)
    else:
        print('All required properties found in dataset.')
        for prop in required_props:
            non_null_count = df[prop].count()
            total_count = len(df)
            print(f'  {prop}: {non_null_count}/{total_count} non-null values ({non_null_count/total_count*100:.1f}%)')
        
except Exception as e:
    print(f'Error checking dataset: {e}')
    sys.exit(1)
"

# Step 3: Run fine-tuning
echo ""
echo "Step 3: Starting multi-property fine-tuning..."
echo "This may take several hours depending on your hardware."

# Create the fine-tuning command
FINETUNE_CMD="mattergen-finetune \
  adapter.pretrained_name=$BASE_MODEL \
  data_module=$DATASET \
  +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.$PROPERTY1=$PROPERTY1 \
  +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.$PROPERTY2=$PROPERTY2 \
  +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.$PROPERTY3=$PROPERTY3 \
  +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.$PROPERTY4=$PROPERTY4 \
  ~trainer.logger \
  data_module.properties=[\"$PROPERTY1\",\"$PROPERTY2\",\"$PROPERTY3\",\"$PROPERTY4\"] \
  data_module.batch_size=$BATCH_SIZE \
  trainer.max_epochs=$MAX_EPOCHS \
  lightning_module.optimizer_partial.lr=$LEARNING_RATE \
  hydra.run.dir=$OUTPUT_DIR"

# Add gradient accumulation for larger datasets to avoid memory issues
if [ "$DATASET" = "alex_mp_20" ]; then
    FINETUNE_CMD="$FINETUNE_CMD trainer.accumulate_grad_batches=4"
fi

echo "Executing fine-tuning command:"
echo "$FINETUNE_CMD"
echo ""

# Execute the command
eval $FINETUNE_CMD

# Step 4: Verify training completed successfully
echo ""
echo "Step 4: Verifying training completion..."

if [ -d "$OUTPUT_DIR" ] && [ -d "$OUTPUT_DIR/checkpoints" ]; then
    BEST_CKPT=$(find "$OUTPUT_DIR/checkpoints" -name "*.ckpt" | grep -E "(best|last)" | head -1)
    if [ -n "$BEST_CKPT" ]; then
        echo "Training completed successfully!"
        echo "Model saved at: $OUTPUT_DIR"
        echo "Best checkpoint: $BEST_CKPT"
        
        # Show training stats if available
        if [ -f "$OUTPUT_DIR/train.log" ]; then
            echo ""
            echo "Final training statistics:"
            tail -5 "$OUTPUT_DIR/train.log"
        fi
    else
        echo "WARNING: No checkpoint files found. Training may have failed."
        exit 1
    fi
else
    echo "ERROR: Output directory or checkpoints not found. Training failed."
    exit 1
fi

# Step 5: Provide example generation command
echo ""
echo "Step 5: Example generation command:"
echo ""
echo "export MODEL_PATH=\"$OUTPUT_DIR\""
echo "export RESULTS_PATH=\"results/multi_property_generation_\$(date +%Y%m%d_%H%M%S)\""
echo ""
echo "python mattergen/scripts/generate.py \"\$RESULTS_PATH\" \\"
echo "  --model_path=\"\$MODEL_PATH\" \\"
echo "  --batch_size=16 \\"
echo "  --num_batches=1 \\"
echo "  --chemical_system_mode contains \\"
echo "  --properties_to_condition_on=\"{'chemical_system': ['O'], 'energy_above_hull': 0.05, 'dft_band_gap': 1.4, 'space_group': 221}\""
echo ""

# Step 6: Create a quick generation test
echo "Step 6: Testing generation with a small sample..."
RESULTS_PATH="results/test_generation_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_PATH"

TEST_CMD="python mattergen/scripts/generate.py \"$RESULTS_PATH\" \
  --model_path=\"$OUTPUT_DIR\" \
  --batch_size=2 \
  --num_batches=1 \
  --chemical_system_mode contains \
  --properties_to_condition_on=\"{'chemical_system': ['O'], 'energy_above_hull': 0.05, 'dft_band_gap': 1.4, 'space_group': 221}\""

echo "Running test generation:"
echo "$TEST_CMD"
echo ""

if eval $TEST_CMD; then
    echo ""
    echo "Test generation completed successfully!"
    echo "Results saved to: $RESULTS_PATH"
    
    # Count generated structures
    if [ -f "$RESULTS_PATH/generated_crystals.extxyz" ]; then
        STRUCTURE_COUNT=$(grep -c "Lattice" "$RESULTS_PATH/generated_crystals.extxyz" || echo "0")
        echo "Generated structures: $STRUCTURE_COUNT"
    fi
    
    echo ""
    echo "=== Fine-tuning and Testing Complete ==="
    echo "Your model is ready for generating structures with the specified properties."
    echo "Model location: $OUTPUT_DIR"
    echo "Test results: $RESULTS_PATH"
else
    echo "Test generation failed. Please check the model and try again."
    exit 1
fi