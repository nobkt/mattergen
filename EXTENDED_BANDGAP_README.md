# Extended Band Gap Crystal Generation

This document describes the extended band gap generation capabilities that include not only band gap energy but also band gap center value and direct/indirect transition type.

## 拡張バンドギャップ結晶生成

このドキュメントでは、バンドギャップエネルギーだけでなく、バンドギャップセンター値および直接/間接遷移タイプも含む拡張バンドギャップ生成機能について説明します。

## New Properties

### 1. `dft_band_gap_center`
- **Type**: Float (continuous)
- **Description**: Band gap center value in eV, representing the center position between valence band maximum (VBM) and conduction band minimum (CBM)
- **Usage**: Allows conditioning on where the band gap is positioned relative to the Fermi level
- **Example values**: -0.5, 0.0, 0.5 (in eV)

### 2. `dft_band_gap_is_direct`
- **Type**: Integer (categorical)
- **Description**: Band gap transition type 
- **Values**: 
  - `0`: Indirect transition (VBM and CBM at different k-points)
  - `1`: Direct transition (VBM and CBM at the same k-point)
- **Usage**: Allows generating materials with specific electronic transition characteristics

## Usage Examples

### Extended Generation Script

Use the new `generate_extended_bandgap_crystals.py` script for comprehensive band gap conditioning:

```bash
# Generate direct band gap semiconductors
python generate_extended_bandgap_crystals.py \
    --band_gap 1.0 \
    --band_gap_center 0.5 \
    --is_direct 1 \
    --output_dir results_direct_bg \
    --batch_size 16

# Generate indirect band gap semiconductors
python generate_extended_bandgap_crystals.py \
    --band_gap 1.2 \
    --band_gap_center -0.2 \
    --is_direct 0 \
    --output_dir results_indirect_bg \
    --batch_size 16
```

### Direct MatterGen Usage

You can also use the standard MatterGen generation command with the new properties:

```bash
# Multiple property conditioning
python -m mattergen.scripts.generate results_extended_bg \
    --pretrained-name=dft_band_gap \
    --batch_size=16 \
    --num_batches=2 \
    --properties_to_condition_on="{'dft_band_gap': 1.0, 'dft_band_gap_center': 0.0, 'dft_band_gap_is_direct': 1}" \
    --diffusion_guidance_factor=2.0
```

### Fine-tuning with Extended Properties

To fine-tune MatterGen on the extended band gap properties:

```bash
# Single property fine-tuning (band gap center)
export PROPERTY=dft_band_gap_center
export MODEL_NAME=mattergen_base
mattergen-finetune adapter.pretrained_name=$MODEL_NAME \
    data_module=mp_20 \
    +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.$PROPERTY=$PROPERTY \
    ~trainer.logger \
    data_module.properties=["$PROPERTY"]

# Multi-property fine-tuning (all band gap properties)
export PROPERTY1=dft_band_gap
export PROPERTY2=dft_band_gap_center  
export PROPERTY3=dft_band_gap_is_direct
export MODEL_NAME=mattergen_base
mattergen-finetune adapter.pretrained_name=$MODEL_NAME \
    data_module=mp_20 \
    +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.$PROPERTY1=$PROPERTY1 \
    +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.$PROPERTY2=$PROPERTY2 \
    +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.$PROPERTY3=$PROPERTY3 \
    ~trainer.logger \
    data_module.properties=["$PROPERTY1","$PROPERTY2","$PROPERTY3"]
```

## Implementation Details

### Property Embedding Classes

1. **`dft_band_gap_center`**: Uses `PropertyEmbedding` with `NoiseLevelEncoding` for continuous values, similar to `dft_band_gap`
2. **`dft_band_gap_is_direct`**: Uses `PropertyEmbedding` with custom `BandGapTransitionTypeEmbeddingVector` for binary categorical data

### Configuration Files

- `mattergen/conf/lightning_module/diffusion_module/model/property_embeddings/dft_band_gap_center.yaml`
- `mattergen/conf/lightning_module/diffusion_module/model/property_embeddings/dft_band_gap_is_direct.yaml`

### Data Format Expectations

When providing data for these properties:

```python
# In your dataset or conditioning
{
    "dft_band_gap": 1.0,           # Band gap energy in eV (0.8-1.5)
    "dft_band_gap_center": 0.2,   # Center position in eV (float)
    "dft_band_gap_is_direct": 1   # Direct (1) or indirect (0) transition
}
```

## Applications

### Direct Band Gap Materials
Useful for applications requiring direct transitions:
- Solar cells and photovoltaics
- LEDs and laser diodes
- Optical sensors

### Indirect Band Gap Materials  
Useful for applications where indirect transitions are preferred:
- Power electronics
- High-efficiency transistors
- Thermal management materials

### Band Gap Center Control
Allows fine-tuning of electronic properties:
- Work function engineering
- Fermi level alignment
- Interface design

## Backward Compatibility

All existing functionality remains unchanged. The new properties are optional and can be used independently or in combination with existing properties.