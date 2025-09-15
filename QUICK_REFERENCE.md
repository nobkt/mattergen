# MatterGen Multi-Property Fine-tuning Quick Reference

## Target Generation Command
```bash
python mattergen/scripts/generate.py "$RESULTS_PATH" \
  --model_path="$MODEL_PATH" \
  --batch_size=16 \
  --num_batches=1 \
  --chemical_system_mode contains \
  --properties_to_condition_on="{'chemical_system': ['O'], 'energy_above_hull': 0.05, 'dft_band_gap': 1.4, 'space_group': 221}"
```

## Required Properties
- `chemical_system`: ['O'] (Oxygen-containing systems)
- `energy_above_hull`: 0.05 (eV/atom above convex hull)
- `dft_band_gap`: 1.4 (eV, DFT-computed band gap)
- `space_group`: 221 (Pm-3m space group)

## Quick Setup

### 1. Prepare Dataset
```bash
# MP-20 (recommended for testing)
git lfs pull -I data-release/mp-20/ --exclude=""
unzip data-release/mp-20/mp_20.zip -d datasets
csv-to-dataset --csv-folder datasets/mp_20/ --dataset-name mp_20 --cache-folder datasets/cache

# Alex-MP-20 (recommended for production)
git lfs pull -I data-release/alex-mp/alex_mp_20.zip --exclude=""
unzip data-release/alex-mp/alex_mp_20.zip -d datasets
csv-to-dataset --csv-folder datasets/alex_mp_20/ --dataset-name alex_mp_20 --cache-folder datasets/cache
```

### 2. Fine-tune Model
```bash
export OUTPUT_DIR="outputs/multi_property_$(date +%Y%m%d_%H%M%S)"

mattergen-finetune \
  adapter.pretrained_name=mattergen_base \
  data_module=mp_20 \
  +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.chemical_system=chemical_system \
  +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.energy_above_hull=energy_above_hull \
  +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.dft_band_gap=dft_band_gap \
  +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.space_group=space_group \
  ~trainer.logger \
  data_module.properties=["chemical_system","energy_above_hull","dft_band_gap","space_group"] \
  trainer.max_epochs=200 \
  hydra.run.dir=$OUTPUT_DIR
```

### 3. Generate Structures
```bash
export MODEL_PATH="$OUTPUT_DIR"
export RESULTS_PATH="results/multi_property_generation"

python mattergen/scripts/generate.py "$RESULTS_PATH" \
  --model_path="$MODEL_PATH" \
  --batch_size=16 \
  --num_batches=1 \
  --chemical_system_mode contains \
  --properties_to_condition_on="{'chemical_system': ['O'], 'energy_above_hull': 0.05, 'dft_band_gap': 1.4, 'space_group': 221}"
```

## Alternative Configurations

### For Larger Dataset (Better Results)
```bash
mattergen-finetune \
  adapter.pretrained_name=mattergen_base \
  data_module=alex_mp_20 \
  +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.chemical_system=chemical_system \
  +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.energy_above_hull=energy_above_hull \
  +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.dft_band_gap=dft_band_gap \
  +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.space_group=space_group \
  ~trainer.logger \
  data_module.properties=["chemical_system","energy_above_hull","dft_band_gap","space_group"] \
  trainer.accumulate_grad_batches=4 \
  trainer.max_epochs=200 \
  hydra.run.dir=$OUTPUT_DIR
```

### For Memory-Constrained Systems
```bash
mattergen-finetune \
  adapter.pretrained_name=mattergen_base \
  data_module=mp_20 \
  +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.chemical_system=chemical_system \
  +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.energy_above_hull=energy_above_hull \
  +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.dft_band_gap=dft_band_gap \
  +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.space_group=space_group \
  ~trainer.logger \
  data_module.properties=["chemical_system","energy_above_hull","dft_band_gap","space_group"] \
  data_module.batch_size=32 \
  trainer.accumulate_grad_batches=4 \
  trainer.max_epochs=200 \
  hydra.run.dir=$OUTPUT_DIR
```

## Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| Out of memory | Reduce `data_module.batch_size=32` or increase `trainer.accumulate_grad_batches=8` |
| Training not converging | Lower learning rate: `lightning_module.optimizer_partial.lr=1e-6` |
| Missing properties | Check dataset with: `python -c "import pandas as pd; print(pd.read_csv('datasets/mp_20/train.csv').columns.tolist())"` |
| Slow training | Use `data_module=mp_20` instead of `alex_mp_20` for testing |

## Monitoring Training
```bash
# Watch training progress
tail -f $OUTPUT_DIR/train.log

# Check validation loss
grep "val_loss" $OUTPUT_DIR/train.log | tail -10
```

## Evaluate Results
```bash
mattergen-evaluate \
  --structures_path=$RESULTS_PATH \
  --relax=True \
  --structure_matcher='disordered' \
  --save_as="$RESULTS_PATH/metrics.json"
```

## Available Properties in Datasets
- `chemical_system`: Chemical composition
- `energy_above_hull`: Thermodynamic stability
- `dft_band_gap`: Electronic band gap (DFT)
- `space_group`: Crystal symmetry
- `dft_mag_density`: Magnetic density
- `ml_bulk_modulus`: Bulk modulus (ML predicted)
- `hhi_score`: Material scarcity index