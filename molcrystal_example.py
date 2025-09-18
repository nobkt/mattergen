#!/usr/bin/env python3
"""
Example script demonstrating how to use the molecular crystal fine-tuning pipeline.

This script creates sample molecular crystal CIF files and demonstrates
the complete workflow from CIF files to fine-tuned model.
"""

import os
import tempfile
from pathlib import Path

# Sample molecular crystal CIF files
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
C4 0.4567 0.5678 0.6789 1.0
C5 0.5678 0.6789 0.7890 1.0
C6 0.6789 0.7890 0.8901 1.0
H1 0.1111 0.2222 0.3333 1.0
H2 0.2222 0.3333 0.4444 1.0
H3 0.3333 0.4444 0.5555 1.0
H4 0.4444 0.5555 0.6666 1.0
H5 0.5555 0.6666 0.7777 1.0
H6 0.6666 0.7777 0.8888 1.0""",

    "mol2.cif": """data_anthracene
_symmetry_space_group_name_H-M 'P 21/a'
_cell_length_a 8.562
_cell_length_b 6.036
_cell_length_c 11.184
_cell_angle_alpha 90.0
_cell_angle_beta 124.7
_cell_angle_gamma 90.0
loop_
_atom_site_label
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
C1 0.0865 0.0275 0.3675 1.0
C2 0.1234 0.1789 0.2987 1.0
C3 0.2456 0.3567 0.3456 1.0
C4 0.3789 0.4123 0.4987 1.0
C5 0.4567 0.2876 0.5789 1.0
C6 0.3987 0.1234 0.5234 1.0
C7 0.2789 0.0567 0.4123 1.0
H1 0.0234 -0.0987 0.3234 1.0
H2 0.0789 0.2345 0.2123 1.0
H3 0.2789 0.4567 0.2987 1.0
H4 0.4567 0.5234 0.5345 1.0
H5 0.5345 0.3234 0.6567 1.0
H6 0.4345 0.0234 0.5789 1.0""",

    "mol3.cif": """data_naphthalene
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
C4 0.3456 0.3789 0.4123 1.0
C5 0.3234 0.2456 0.5234 1.0
C6 0.2123 0.0789 0.4789 1.0
C7 0.1567 0.0234 0.3456 1.0
C8 0.2789 0.1567 0.2987 1.0
C9 0.3987 0.2789 0.3567 1.0
C10 0.4234 0.3456 0.4987 1.0
H1 0.0234 -0.0987 0.2987 1.0
H2 0.0789 0.1234 0.1567 1.0
H3 0.2789 0.4123 0.2234 1.0
H4 0.4123 0.4789 0.4456 1.0
H5 0.3789 0.2789 0.6123 1.0
H6 0.1789 0.0234 0.5234 1.0
H7 0.2345 0.1234 0.2345 1.0
H8 0.4567 0.3234 0.3123 1.0""",

    "mol4.cif": """data_pyridine
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
C3 0.0789 0.0987 0.5123 1.0
C4 0.1567 0.1789 0.5987 1.0
C5 0.2234 0.2789 0.5678 1.0
H1 0.1123 0.2456 0.2567 1.0
H2 -0.0234 0.0567 0.3123 1.0
H3 0.0234 0.0123 0.5345 1.0
H4 0.1789 0.1678 0.6789 1.0
H5 0.2789 0.3234 0.6234 1.0""",

    "mol5.cif": """data_thiophene
_symmetry_space_group_name_H-M 'P 21/c'
_cell_length_a 7.298
_cell_length_b 5.528
_cell_length_c 8.042
_cell_angle_alpha 90.0
_cell_angle_beta 110.3
_cell_angle_gamma 90.0
loop_
_atom_site_label
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
S1 0.2345 0.1234 0.3456 1.0
C1 0.3456 0.2789 0.5234 1.0
C2 0.4789 0.3567 0.6789 1.0
C3 0.5234 0.2456 0.6234 1.0
C4 0.4123 0.1234 0.4567 1.0
H1 0.3123 0.3234 0.5567 1.0
H2 0.5456 0.4234 0.7789 1.0
H3 0.6123 0.2789 0.7234 1.0
H4 0.4456 0.0567 0.4123 1.0"""
}

def create_sample_molcrystal_directory(base_dir: Path) -> Path:
    """Create a sample molcrystal directory with example CIF files."""
    molcrystal_dir = base_dir / "molcrystal"
    molcrystal_dir.mkdir(parents=True, exist_ok=True)
    
    for filename, content in SAMPLE_CIFS.items():
        (molcrystal_dir / filename).write_text(content)
    
    print(f"Created sample molcrystal directory: {molcrystal_dir}")
    print(f"Files created: {[f.name for f in molcrystal_dir.glob('*.cif')]}")
    
    return molcrystal_dir

def run_example_pipeline(base_dir: Path):
    """Run the complete example pipeline."""
    import subprocess
    import sys
    
    # Create sample data
    molcrystal_dir = create_sample_molcrystal_directory(base_dir)
    output_dir = base_dir / "output"
    
    print("\n" + "="*60)
    print("RUNNING EXAMPLE PIPELINE")
    print("="*60)
    
    # Run the dataset creation only (skip fine-tuning for demo)
    cmd = [
        sys.executable,
        "/home/runner/work/mattergen/mattergen/mattergen/scripts/molcrystal_finetune.py",
        "--molcrystal_dir", str(molcrystal_dir),
        "--output_dir", str(output_dir),
        "--dataset_name", "example_molcrystal",
        "--train_split", "0.6",
        "--val_split", "0.2",
        "--skip_finetuning"  # Skip fine-tuning for this demo
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    print("(Note: Skipping fine-tuning for this demo)\n")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("STDOUT:")
        print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        print("\n✓ Pipeline completed successfully!")
        
        # Show the output structure
        print("\nOutput directory structure:")
        for root, dirs, files in os.walk(output_dir):
            level = root.replace(str(output_dir), '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files:
                print(f"{subindent}{file}")
                
    except subprocess.CalledProcessError as e:
        print(f"❌ Pipeline failed with exit code {e.returncode}")
        print("STDOUT:")
        print(e.stdout)
        print("STDERR:")
        print(e.stderr)

def print_next_steps(output_dir: Path):
    """Print instructions for next steps."""
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    
    print("\n1. To run fine-tuning on this dataset:")
    print(f"   python molcrystal_finetune.py \\")
    print(f"       --skip_dataset_creation \\")
    print(f"       --output_dir {output_dir}")
    
    print("\n2. To generate new molecular crystals after fine-tuning:")
    print(f"   python -m mattergen.scripts.generate \\")
    print(f"       --model_path {output_dir}/model \\")
    print(f"       --output_path ./generated_crystals \\")
    print(f"       --num_batches 3 \\")
    print(f"       --batch_size 16")
    
    print("\n3. Files created in this example:")
    print(f"   - Dataset CSV files: {output_dir}/csv/")
    print(f"   - Cached datasets: {output_dir}/cache/")
    print(f"   - Configuration files: {output_dir}/configs/")

def main():
    """Main function to run the example."""
    print("MOLECULAR CRYSTAL FINE-TUNING EXAMPLE")
    print("="*60)
    print("This example demonstrates the complete workflow for")
    print("fine-tuning MatterGen on molecular crystal datasets.")
    print("="*60)
    
    # Create temporary directory for the example
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        print(f"\nWorking directory: {temp_path}")
        
        try:
            run_example_pipeline(temp_path)
            print_next_steps(temp_path / "output")
            
            print("\n" + "="*60)
            print("✅ EXAMPLE COMPLETED SUCCESSFULLY!")
            print("="*60)
            print("The dataset has been created and is ready for fine-tuning.")
            print("See the MOLCRYSTAL_README.md for detailed documentation.")
            
        except Exception as e:
            print(f"\n❌ Example failed: {e}")
            print("Please check the error messages above.")

if __name__ == "__main__":
    main()