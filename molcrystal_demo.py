#!/usr/bin/env python3
"""
Standalone demo showing how the molcrystal fine-tuning scripts work.

This demo creates sample CIF files and shows the expected workflow
without requiring the full MatterGen environment.
"""

import tempfile
from pathlib import Path

# Sample molecular crystal CIF files (realistic examples)
SAMPLE_CIFS = {
    "mol1.cif": """data_benzene
_symmetry_space_group_name_H-M 'P b c a'
_cell_length_a 7.39
_cell_length_b 9.42
_cell_length_c 6.81
_cell_angle_alpha 90.0
_cell_angle_beta 90.0
_cell_angle_gamma 90.0
loop_
_atom_site_label
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
C1 0.1234 0.2345 0.3456 1.0
C2 0.2345 0.3456 0.4567 1.0
C3 0.3456 0.4567 0.5678 1.0
H1 0.1111 0.2222 0.3333 1.0
H2 0.2222 0.3333 0.4444 1.0
H3 0.3333 0.4444 0.5555 1.0""",

    "mol2.cif": """data_naphthalene
_symmetry_space_group_name_H-M 'P 21/a'
_cell_length_a 8.235
_cell_length_b 5.953
_cell_length_c 8.658
_cell_angle_alpha 90.0
_cell_angle_beta 122.9
_cell_angle_gamma 90.0
loop_
_atom_site_label
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
C1 0.0823 0.0192 0.3298 1.0
C2 0.1234 0.1567 0.2345 1.0
C3 0.2567 0.3234 0.2789 1.0
H1 0.0234 -0.0987 0.2987 1.0
H2 0.0789 0.1234 0.1567 1.0
H3 0.2789 0.4123 0.2234 1.0""",

    "mol3.cif": """data_pyridine
_symmetry_space_group_name_H-M 'P n a 21'
_cell_length_a 8.776
_cell_length_b 5.504
_cell_length_c 9.721
_cell_angle_alpha 90.0
_cell_angle_beta 90.0
_cell_angle_gamma 90.0
loop_
_atom_site_label
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
N1 0.2345 0.3456 0.4567 1.0
C1 0.1234 0.2345 0.3456 1.0
C2 0.0456 0.1234 0.3789 1.0
H1 0.1123 0.2456 0.2567 1.0
H2 -0.0234 0.0567 0.3123 1.0"""
}

def create_demo_structure(base_dir: Path):
    """Create demo directory structure with sample files."""
    
    # Create molcrystal directory with CIF files
    molcrystal_dir = base_dir / "molcrystal"
    molcrystal_dir.mkdir(parents=True)
    
    for filename, content in SAMPLE_CIFS.items():
        (molcrystal_dir / filename).write_text(content)
    
    print(f"📁 Created molcrystal directory: {molcrystal_dir}")
    print(f"   Files: {[f.name for f in molcrystal_dir.glob('*.cif')]}")
    
    return molcrystal_dir

def show_workflow_demo():
    """Demonstrate the complete workflow."""
    
    print("="*70)
    print("🧪 MOLECULAR CRYSTAL FINE-TUNING WORKFLOW DEMO")
    print("="*70)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Step 1: Create sample data
        print("\n📋 STEP 1: Prepare molecular crystal CIF files")
        print("-" * 50)
        molcrystal_dir = create_demo_structure(temp_path)
        
        # Show what the directory structure looks like
        print(f"\n📂 Directory structure:")
        print(f"   {molcrystal_dir.name}/")
        for cif_file in sorted(molcrystal_dir.glob("*.cif")):
            with open(cif_file, 'r') as f:
                first_line = f.readline().strip()
            print(f"   ├── {cif_file.name} ({first_line})")
        
        # Step 2: Show the command that would be run
        print(f"\n🔧 STEP 2: Convert CIF files to MatterGen dataset")
        print("-" * 50)
        print("Command that would be executed:")
        print(f"python molcrystal_finetune.py \\")
        print(f"    --molcrystal_dir {molcrystal_dir} \\")
        print(f"    --output_dir {temp_path}/output \\")
        print(f"    --dataset_name my_molcrystal \\")
        print(f"    --train_split 0.8 \\")
        print(f"    --val_split 0.1")
        
        # Step 3: Show expected output structure
        print(f"\n📊 STEP 3: Expected output structure")
        print("-" * 50)
        output_structure = """output/
├── csv/
│   ├── train.csv              # 80% of data (2 structures)
│   ├── val.csv                # 10% of data (1 structure)  
│   └── test.csv               # 10% of data (1 structure)
├── cache/
│   └── my_molcrystal/
│       ├── train/             # Cached training dataset
│       ├── val/               # Cached validation dataset
│       └── test/              # Cached test dataset
├── configs/
│   ├── my_molcrystal.yaml     # Data module configuration
│   └── molcrystal_finetune.yaml  # Fine-tuning configuration
└── model/                     # (Created after fine-tuning)
    └── [timestamp]/
        ├── checkpoints/
        ├── config.yaml
        └── logs/"""
        
        print(output_structure)
        
        # Step 4: Show fine-tuning process
        print(f"\n🏋️ STEP 4: Fine-tuning process")
        print("-" * 50)
        print("The fine-tuning process would:")
        print("• Load pre-trained MatterGen model (Materials Project)")
        print("• Create adapter layers for your molecular crystal data")
        print("• Train only the adapter layers (efficient fine-tuning)")
        print("• Generate model checkpoints for unconditional sampling")
        
        # Step 5: Show generation
        print(f"\n🎯 STEP 5: Generate new molecular crystals")
        print("-" * 50)
        print("After fine-tuning, generate new structures:")
        print("python -m mattergen.scripts.generate \\")
        print("    --model_path ./output/model \\")
        print("    --output_path ./generated_crystals \\")
        print("    --num_batches 5 \\")
        print("    --batch_size 32")
        
        # Step 6: Show benefits
        print(f"\n✨ BENEFITS OF THIS APPROACH")
        print("-" * 50)
        print("• No property conditioning required (unconditional generation)")
        print("• Efficient adapter-based fine-tuning (faster than training from scratch)")
        print("• Leverages pre-trained knowledge from Materials Project")
        print("• Generates structures in the molecular crystal domain")
        print("• Maintains crystal structure validity and physical constraints")
        
        print(f"\n🎉 DEMO COMPLETED!")
        print("="*70)
        print("This demonstrates how to fine-tune MatterGen for molecular crystals.")
        print("The actual scripts handle all the data processing, configuration")
        print("generation, and fine-tuning automatically.")
        print("="*70)

def print_usage_instructions():
    """Print actual usage instructions."""
    print("\n📖 ACTUAL USAGE INSTRUCTIONS")
    print("="*70)
    
    print("\n1️⃣ Prepare your data:")
    print("   • Create a 'molcrystal' directory")
    print("   • Add your CIF files: mol1.cif, mol2.cif, ...")
    
    print("\n2️⃣ Run the complete pipeline:")
    print("   python mattergen/scripts/molcrystal_finetune.py \\")
    print("       --molcrystal_dir ./molcrystal \\")
    print("       --output_dir ./output")
    
    print("\n3️⃣ Or run steps separately:")
    print("   # Dataset creation only:")
    print("   python mattergen/scripts/cif_to_dataset.py \\")
    print("       --cif_dir ./molcrystal \\")
    print("       --output_dir ./dataset")
    print("")
    print("   # Fine-tuning only:")
    print("   python mattergen/scripts/molcrystal_finetune.py \\")
    print("       --skip_dataset_creation \\")
    print("       --output_dir ./output")
    
    print("\n4️⃣ Generate new molecular crystals:")
    print("   python -m mattergen.scripts.generate \\")
    print("       --model_path ./output/model \\")
    print("       --output_path ./generated_crystals")
    
    print("\n📚 For detailed documentation, see MOLCRYSTAL_README.md")

if __name__ == "__main__":
    show_workflow_demo()
    print_usage_instructions()