#!/usr/bin/env python3
"""
Test script for extract_bandgap_properties.py

This script tests the band gap properties extraction functionality
to ensure it works correctly with various input scenarios.
"""

import os
import json
import csv
import tempfile
import subprocess
import sys
from pathlib import Path

def create_test_csv(csv_path: str, num_materials: int = 5):
    """Create a test CSV file with sample band gap data."""
    
    sample_cif = '''# generated using pymatgen
data_TestMaterial
_symmetry_space_group_name_H-M   'P 1'
_cell_length_a   5.0
_cell_length_b   5.0
_cell_length_c   5.0
_cell_angle_alpha   90.00
_cell_angle_beta   90.00
_cell_angle_gamma   90.00
_symmetry_Int_Tables_number   1
_chemical_formula_structural   TestMaterial
_chemical_formula_sum   'Test1'
_cell_volume   125.0
_cell_formula_units_Z   1
loop_
 _symmetry_equiv_pos_site_id
 _symmetry_equiv_pos_as_xyz
  1  'x, y, z'
loop_
 _atom_site_type_symbol
 _atom_site_label
 _atom_site_symmetry_multiplicity
 _atom_site_fract_x
 _atom_site_fract_y
 _atom_site_fract_z
 _atom_site_occupancy
  Test  Test0  1  0.00000000  0.00000000  0.00000000  1'''
    
    data = [
        {'material_id': 'test-001', 'dft_band_gap': 0.85, 'formation_energy_per_atom': -1.2, 'cif': sample_cif},
        {'material_id': 'test-002', 'dft_band_gap': 1.0, 'formation_energy_per_atom': -0.8, 'cif': sample_cif},
        {'material_id': 'test-003', 'dft_band_gap': 1.15, 'formation_energy_per_atom': -1.5, 'cif': sample_cif},
        {'material_id': 'test-004', 'dft_band_gap': 1.3, 'formation_energy_per_atom': -0.9, 'cif': sample_cif},
        {'material_id': 'test-005', 'dft_band_gap': 1.42, 'formation_energy_per_atom': -1.1, 'cif': sample_cif},
    ]
    
    # Select the requested number of materials
    data = data[:num_materials]
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['material_id', 'dft_band_gap', 'formation_energy_per_atom', 'cif']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    
    print(f"Created test CSV with {len(data)} materials at: {csv_path}")
    return data

def test_extraction_script(script_path: str, test_csv: str, output_dir: str):
    """Test the extraction script."""
    
    print(f"Testing extraction script: {script_path}")
    
    # Run the extraction script
    cmd = [
        sys.executable, script_path,
        '--input_csv', test_csv,
        '--output_dir', output_dir,
        '--seed', '42',
        '--verbose'
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error running extraction script:")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        return False
    
    print("Extraction script completed successfully!")
    return True

def validate_output_files(output_dir: str, expected_materials: int):
    """Validate that all expected output files are created correctly."""
    
    print("Validating output files...")
    
    # Check for expected files
    expected_files = [
        'complete_bandgap_properties.csv',
        'bandgap_center_dataset.csv', 
        'bandgap_direct_dataset.csv',
        'extraction_statistics.json',
        'fine_tune_bandgap.sh'
    ]
    
    for filename in expected_files:
        filepath = os.path.join(output_dir, filename)
        if not os.path.exists(filepath):
            print(f"ERROR: Expected file not found: {filepath}")
            return False
        print(f"✓ Found: {filename}")
    
    # Validate statistics file
    stats_path = os.path.join(output_dir, 'extraction_statistics.json')
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    
    if stats['total_materials'] != expected_materials:
        print(f"ERROR: Expected {expected_materials} materials, got {stats['total_materials']}")
        return False
    
    print(f"✓ Statistics valid: {stats['total_materials']} materials processed")
    
    # Validate CSV files have correct structure
    for csv_file in ['complete_bandgap_properties.csv', 'bandgap_center_dataset.csv', 'bandgap_direct_dataset.csv']:
        csv_path = os.path.join(output_dir, csv_file)
        with open(csv_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            rows = list(reader)
            
            if len(rows) != expected_materials:
                print(f"ERROR: {csv_file} has {len(rows)} rows, expected {expected_materials}")
                return False
            
            # Check for required columns
            if csv_file == 'complete_bandgap_properties.csv':
                required_cols = ['material_id', 'dft_band_gap', 'dft_band_gap_center', 'dft_band_gap_is_direct']
            elif csv_file == 'bandgap_center_dataset.csv':
                required_cols = ['material_id', 'dft_band_gap_center']
            else:  # bandgap_direct_dataset.csv
                required_cols = ['material_id', 'dft_band_gap_is_direct']
            
            for col in required_cols:
                if col not in reader.fieldnames:
                    print(f"ERROR: {csv_file} missing column: {col}")
                    return False
            
            print(f"✓ {csv_file} structure valid")
    
    print("All output files validated successfully!")
    return True

def test_edge_cases(script_dir: str):
    """Test edge cases and error handling."""
    
    print("Testing edge cases...")
    
    script_path = os.path.join(script_dir, 'extract_bandgap_properties.py')
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test 1: Missing file
        result = subprocess.run([
            sys.executable, script_path,
            '--input_csv', 'nonexistent.csv',
            '--output_dir', temp_dir
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("ERROR: Script should have failed for missing input file")
            return False
        print("✓ Correctly handled missing input file")
        
        # Test 2: Invalid CSV structure
        invalid_csv = os.path.join(temp_dir, 'invalid.csv')
        with open(invalid_csv, 'w') as f:
            f.write("wrong,columns\n1,2\n")
        
        result = subprocess.run([
            sys.executable, script_path,
            '--input_csv', invalid_csv,
            '--output_dir', temp_dir
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("ERROR: Script should have failed for invalid CSV structure")
            return False
        print("✓ Correctly handled invalid CSV structure")
    
    print("Edge case testing completed!")
    return True

def main():
    """Main test function."""
    
    print("Starting band gap properties extraction tests...")
    print("=" * 60)
    
    # Get the script directory
    script_dir = Path(__file__).parent.absolute()
    script_path = os.path.join(script_dir, 'extract_bandgap_properties.py')
    
    if not os.path.exists(script_path):
        print(f"ERROR: Script not found at {script_path}")
        return False
    
    success = True
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test data
        test_csv = os.path.join(temp_dir, 'test_data.csv')
        test_data = create_test_csv(test_csv, num_materials=5)
        
        # Test extraction
        output_dir = os.path.join(temp_dir, 'output')
        if not test_extraction_script(script_path, test_csv, output_dir):
            success = False
        
        # Validate outputs
        if success and not validate_output_files(output_dir, len(test_data)):
            success = False
    
    # Test edge cases
    if success and not test_edge_cases(script_dir):
        success = False
    
    print("=" * 60)
    if success:
        print("✅ All tests passed successfully!")
        print("The band gap properties extraction script is working correctly.")
    else:
        print("❌ Some tests failed!")
        print("Please check the script implementation.")
    
    return success

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)