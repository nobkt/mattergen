#!/usr/bin/env python3
"""
Demonstration script for extended band gap properties

This script demonstrates various combinations of band gap properties
that can be used for crystal generation with MatterGen.

このスクリプトは、MatterGenでの結晶生成に使用できるバンドギャップ特性の
さまざまな組み合わせを実証します。
"""

def demo_property_combinations():
    """Show examples of different property combinations"""
    print("Extended Band Gap Properties - Usage Examples")
    print("=" * 60)
    
    examples = [
        {
            "name": "Direct Band Gap Solar Cell Material",
            "properties": {
                "dft_band_gap": 1.1,
                "dft_band_gap_center": 0.0,
                "dft_band_gap_is_direct": 1
            },
            "description": "Suitable for photovoltaic applications requiring direct transitions"
        },
        {
            "name": "Indirect Band Gap Power Electronics", 
            "properties": {
                "dft_band_gap": 1.4,
                "dft_band_gap_center": -0.3,
                "dft_band_gap_is_direct": 0
            },
            "description": "Higher band gap indirect semiconductor for power devices"
        },
        {
            "name": "LED Material with Direct Transition",
            "properties": {
                "dft_band_gap": 0.9,
                "dft_band_gap_center": 0.2,
                "dft_band_gap_is_direct": 1
            },
            "description": "Lower band gap direct transition for near-infrared LEDs"
        },
        {
            "name": "Wide Band Gap Indirect Semiconductor",
            "properties": {
                "dft_band_gap": 1.5,
                "dft_band_gap_center": 0.5,
                "dft_band_gap_is_direct": 0
            },
            "description": "Wide band gap for high-temperature applications"
        },
        {
            "name": "Band Gap Only (Original Functionality)",
            "properties": {
                "dft_band_gap": 1.0
            },
            "description": "Traditional band gap conditioning without extended properties"
        },
        {
            "name": "Band Gap + Center (No Transition Type)",
            "properties": {
                "dft_band_gap": 1.2,
                "dft_band_gap_center": -0.1
            },
            "description": "Partial extended conditioning - band gap and center only"
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n{i}. {example['name']}")
        print(f"   Description: {example['description']}")
        print("   Properties:", example['properties'])
        
        # Generate command
        props_str = str(example['properties']).replace("'", '"')
        cmd = f"""   Command:
   python -m mattergen.scripts.generate results_example_{i} \\
       --pretrained-name=dft_band_gap \\
       --batch_size=16 \\
       --properties_to_condition_on='{props_str}' \\
       --diffusion_guidance_factor=2.0"""
        print(cmd)
        
        # Extended script command if all properties are present
        if all(key in example['properties'] for key in ['dft_band_gap', 'dft_band_gap_center', 'dft_band_gap_is_direct']):
            ext_cmd = f"""   Extended script:
   python generate_extended_bandgap_crystals.py \\
       --band_gap {example['properties']['dft_band_gap']} \\
       --band_gap_center {example['properties']['dft_band_gap_center']} \\
       --is_direct {example['properties']['dft_band_gap_is_direct']} \\
       --output_dir results_example_{i}_ext \\
       --batch_size 16"""
            print(ext_cmd)

def demo_fine_tuning_commands():
    """Show fine-tuning examples with extended properties"""
    print("\n\nFine-tuning Examples")
    print("=" * 60)
    
    examples = [
        {
            "name": "Single Property: Band Gap Center",
            "command": """export PROPERTY=dft_band_gap_center
export MODEL_NAME=mattergen_base
mattergen-finetune adapter.pretrained_name=$MODEL_NAME \\
    data_module=mp_20 \\
    +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.$PROPERTY=$PROPERTY \\
    ~trainer.logger \\
    data_module.properties=["$PROPERTY"]"""
        },
        {
            "name": "Single Property: Direct/Indirect Transition",
            "command": """export PROPERTY=dft_band_gap_is_direct
export MODEL_NAME=mattergen_base
mattergen-finetune adapter.pretrained_name=$MODEL_NAME \\
    data_module=mp_20 \\
    +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.$PROPERTY=$PROPERTY \\
    ~trainer.logger \\
    data_module.properties=["$PROPERTY"]"""
        },
        {
            "name": "All Band Gap Properties Combined",
            "command": """export PROPERTY1=dft_band_gap
export PROPERTY2=dft_band_gap_center  
export PROPERTY3=dft_band_gap_is_direct
export MODEL_NAME=mattergen_base
mattergen-finetune adapter.pretrained_name=$MODEL_NAME \\
    data_module=mp_20 \\
    +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.$PROPERTY1=$PROPERTY1 \\
    +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.$PROPERTY2=$PROPERTY2 \\
    +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.$PROPERTY3=$PROPERTY3 \\
    ~trainer.logger \\
    data_module.properties=["$PROPERTY1","$PROPERTY2","$PROPERTY3"]"""
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n{i}. {example['name']}")
        print(example['command'])

def demo_data_format():
    """Show expected data format for the new properties"""
    print("\n\nData Format Examples")
    print("=" * 60)
    
    print("""
For training data or conditioning, provide properties in this format:

Python dictionary:
{
    "dft_band_gap": 1.0,           # Required: Band gap energy in eV (0.8-1.5)
    "dft_band_gap_center": 0.2,   # Optional: Center position in eV (float)
    "dft_band_gap_is_direct": 1   # Optional: Direct (1) or indirect (0)
}

CSV format (for datasets):
structure_id,dft_band_gap,dft_band_gap_center,dft_band_gap_is_direct
mp-1,1.0,0.2,1
mp-2,1.2,-0.1,0
mp-3,0.9,0.0,1

JSON format:
[
    {
        "structure_id": "mp-1",
        "dft_band_gap": 1.0,
        "dft_band_gap_center": 0.2,
        "dft_band_gap_is_direct": 1
    },
    {
        "structure_id": "mp-2", 
        "dft_band_gap": 1.2,
        "dft_band_gap_center": -0.1,
        "dft_band_gap_is_direct": 0
    }
]

Notes:
- dft_band_gap: Float in range [0.8, 1.5] eV
- dft_band_gap_center: Float (no strict range, typically [-2, 2] eV)
- dft_band_gap_is_direct: Integer 0 (indirect) or 1 (direct)
""")

if __name__ == "__main__":
    demo_property_combinations()
    demo_fine_tuning_commands()
    demo_data_format()
    
    print("\n" + "=" * 60)
    print("For more information, see EXTENDED_BANDGAP_README.md")
    print("=" * 60)