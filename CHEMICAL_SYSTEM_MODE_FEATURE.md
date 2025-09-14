# Chemical System Conditioning Mode Feature

## Overview

This feature adds support for a new chemical system conditioning mode that allows generating crystals that "contain" specified elements rather than "only containing" those elements.

## Problem Statement

The original issue (in Japanese) requested the ability to generate crystals like "O原子を含む結晶を生成する" (generate crystals containing O atoms) rather than only generating crystals with compositions that contain only the specified elements.

## Solution

Added a new `chemical_system_mode` parameter with two modes:

1. **"exact"** (default): Only allow elements that are in the specified chemical system (original behavior)
2. **"contains"**: Allow any elements but bias towards the specified elements to increase their probability

## Implementation Details

### Key Changes

1. **mattergen/denoiser.py**:
   - Added `chemical_system_mode` parameter to `mask_disallowed_elements()` function
   - Added `mask_disallowed_elements_with_mode()` factory function
   - Updated `GemNetTDenoiser` class to support the new parameter
   - Modified masking logic to support both "exact" and "contains" modes

2. **mattergen/scripts/generate.py**:
   - Added `chemical_system_mode` parameter to the main generation function
   - Updated config overrides to use appropriate mask function based on mode

3. **mattergen/generator.py**:
   - Added `chemical_system_mode` parameter to `CrystalGenerator` class

### Behavior Differences

#### "exact" mode (default, original behavior):
- Only generates elements that are in the specified chemical system
- Uses masking to completely prevent generation of unspecified elements
- Example: chemical_system=["Li", "O"] → generates Li2O, LiO2, etc.

#### "contains" mode (new behavior):
- Allows generation of any elements
- Biases logits towards specified elements to increase their probability
- Example: chemical_system=["Li", "O"] → generates Li2O, Li2O3F, LiMgO2, etc.

## Usage Examples

### Command Line Interface

```bash
# Generate crystals that ONLY contain Li and O (original behavior)
mattergen-generate --output_path outputs_exact \
                   --pretrained_name chemical_system \
                   --properties_to_condition_on '{"chemical_system": ["Li", "O"]}' \
                   --chemical_system_mode exact

# Generate crystals that CONTAIN Li and O (new behavior)
mattergen-generate --output_path outputs_contains \
                   --pretrained_name chemical_system \
                   --properties_to_condition_on '{"chemical_system": ["Li", "O"]}' \
                   --chemical_system_mode contains
```

### Python API

```python
from mattergen.scripts.generate import main

# Generate crystals that contain O atoms (but may have other elements)
structures = main(
    output_path="outputs_contains_o",
    pretrained_name="chemical_system",
    properties_to_condition_on={"chemical_system": ["O"]},
    chemical_system_mode="contains",
    batch_size=32,
    num_batches=1
)
```

## Backward Compatibility

- Default behavior is unchanged (chemical_system_mode="exact")
- All existing code and configurations continue to work as before
- New functionality is opt-in via the `chemical_system_mode` parameter

## Technical Implementation

The "contains" mode works by:
1. Not masking any elements during generation (allows all valid elements)
2. Adding a bias to the logits for specified elements to increase their probability
3. The bias strength is configurable (default: 2.0)

This approach ensures that specified elements are more likely to appear while still allowing other elements to be generated, addressing the requirement for generating crystals that "contain" rather than "only contain" specified elements.