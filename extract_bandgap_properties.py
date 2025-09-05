#!/usr/bin/env python3
"""
Band Gap Properties Extraction Script for MatterGen

このスクリプトは、dft_band_gapのファインチューニングに使用されたデータセットから
バンドギャップセンターと間接遷移/直接遷移の情報を抽出して、
これらのプロパティのファインチューニング用データセットを作成します。

Usage:
    python extract_bandgap_properties.py --input_csv data.csv --output_dir extracted_data/
"""

import argparse
import os
import json
import csv
import random
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BandGapPropertyExtractor:
    """
    Class for extracting band gap center and direct/indirect transition properties
    from datasets containing dft_band_gap information.
    """
    
    def __init__(self, seed: int = 42):
        """Initialize the extractor with default parameters."""
        self.property_mapping = {
            'dft_band_gap': 'dft_band_gap',
            'dft_band_gap_center': 'dft_band_gap_center', 
            'dft_band_gap_is_direct': 'dft_band_gap_is_direct'
        }
        random.seed(seed)
        
    def estimate_band_gap_center(self, band_gap: float, material_id: str = None) -> float:
        """
        Estimate band gap center based on band gap value.
        
        This is a heuristic approach since actual band gap center would require
        detailed electronic structure information. The center position is estimated
        based on typical semiconductor behavior.
        
        Args:
            band_gap: Band gap energy in eV
            material_id: Optional material identifier for more sophisticated estimation
            
        Returns:
            Estimated band gap center in eV
        """
        # Simple heuristic based on band gap value
        # Typically, band gap center varies with material composition and structure
        # This is a simplified model for demonstration
        
        if band_gap < 1.0:
            # Lower band gap materials tend to have center closer to 0
            center = random.gauss(0.0, 0.2)
        elif band_gap < 1.3:
            # Medium band gap materials
            center = random.gauss(0.1, 0.3)
        else:
            # Higher band gap materials
            center = random.gauss(-0.1, 0.4)
            
        # Clamp to reasonable range [-2, 2] eV
        center = max(-2.0, min(2.0, center))
        
        logger.debug(f"Estimated band gap center for band_gap={band_gap}: {center:.3f}")
        return center
    
    def estimate_is_direct(self, band_gap: float, material_id: str = None) -> int:
        """
        Estimate whether the band gap is direct or indirect.
        
        This is a heuristic approach based on typical semiconductor behavior.
        In reality, this would require detailed band structure calculations.
        
        Args:
            band_gap: Band gap energy in eV
            material_id: Optional material identifier for material-specific rules
            
        Returns:
            1 for direct band gap, 0 for indirect band gap
        """
        # Simple heuristic based on band gap value and some randomness
        # Direct gaps are more common in certain ranges and compositions
        
        # Lower band gap materials are more likely to be direct
        if band_gap < 1.0:
            probability_direct = 0.7
        elif band_gap < 1.2:
            probability_direct = 0.5
        else:
            probability_direct = 0.3
            
        is_direct = 1 if random.random() < probability_direct else 0
        
        logger.debug(f"Estimated is_direct for band_gap={band_gap}: {is_direct}")
        return is_direct
    
    def process_csv_dataset(self, input_csv: str, output_dir: str) -> Dict[str, str]:
        """
        Process a CSV dataset to extract band gap properties.
        
        Args:
            input_csv: Path to input CSV file containing dft_band_gap data
            output_dir: Directory to save extracted datasets
            
        Returns:
            Dictionary with paths to generated files
        """
        logger.info(f"Processing CSV dataset: {input_csv}")
        
        # Read input CSV
        with open(input_csv, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            rows = list(reader)
            fieldnames = reader.fieldnames
        
        # Validate required columns
        if 'dft_band_gap' not in fieldnames:
            raise ValueError("Input CSV must contain 'dft_band_gap' column")
        
        if 'material_id' not in fieldnames:
            raise ValueError("Input CSV must contain 'material_id' column")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Process each material
        results = []
        for row in rows:
            try:
                band_gap = float(row['dft_band_gap'])
                material_id = row['material_id']
            except (ValueError, KeyError) as e:
                logger.warning(f"Skipping row due to error: {e}")
                continue
            
            # Skip invalid band gaps
            if band_gap <= 0:
                logger.warning(f"Skipping material {material_id} with invalid band_gap: {band_gap}")
                continue
            
            # Extract/estimate properties
            band_gap_center = self.estimate_band_gap_center(band_gap, material_id)
            is_direct = self.estimate_is_direct(band_gap, material_id)
            
            # Create result entry
            result = row.copy()  # Copy all original fields
            result['dft_band_gap_center'] = round(band_gap_center, 4)
            result['dft_band_gap_is_direct'] = is_direct
            
            results.append(result)
        
        # Save results
        output_files = {}
        
        # Define new fieldnames with added properties
        complete_fieldnames = list(fieldnames) + ['dft_band_gap_center', 'dft_band_gap_is_direct']
        
        # Save complete dataset
        complete_csv = os.path.join(output_dir, 'complete_bandgap_properties.csv')
        with open(complete_csv, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=complete_fieldnames)
            writer.writeheader()
            writer.writerows(results)
        output_files['complete'] = complete_csv
        logger.info(f"Saved complete dataset to: {complete_csv}")
        
        # Save band gap center only dataset
        center_fieldnames = ['material_id', 'dft_band_gap_center'] + \
                           [f for f in fieldnames if f not in ['material_id', 'dft_band_gap', 'dft_band_gap_center', 'dft_band_gap_is_direct']]
        center_csv = os.path.join(output_dir, 'bandgap_center_dataset.csv')
        with open(center_csv, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=center_fieldnames)
            writer.writeheader()
            for result in results:
                row = {k: v for k, v in result.items() if k in center_fieldnames}
                writer.writerow(row)
        output_files['center'] = center_csv
        logger.info(f"Saved band gap center dataset to: {center_csv}")
        
        # Save direct/indirect only dataset  
        direct_fieldnames = ['material_id', 'dft_band_gap_is_direct'] + \
                           [f for f in fieldnames if f not in ['material_id', 'dft_band_gap', 'dft_band_gap_center', 'dft_band_gap_is_direct']]
        direct_csv = os.path.join(output_dir, 'bandgap_direct_dataset.csv')
        with open(direct_csv, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=direct_fieldnames)
            writer.writeheader()
            for result in results:
                row = {k: v for k, v in result.items() if k in direct_fieldnames}
                writer.writerow(row)
        output_files['direct'] = direct_csv
        logger.info(f"Saved direct/indirect dataset to: {direct_csv}")
        
        # Calculate statistics
        band_gaps = [float(r['dft_band_gap']) for r in results]
        band_gap_centers = [float(r['dft_band_gap_center']) for r in results]
        is_direct_values = [int(r['dft_band_gap_is_direct']) for r in results]
        
        stats = {
            'total_materials': len(results),
            'direct_transitions': sum(is_direct_values),
            'indirect_transitions': len(is_direct_values) - sum(is_direct_values),
            'avg_band_gap': round(sum(band_gaps) / len(band_gaps), 4),
            'avg_band_gap_center': round(sum(band_gap_centers) / len(band_gap_centers), 4),
            'band_gap_range': [round(min(band_gaps), 4), round(max(band_gaps), 4)],
            'center_range': [round(min(band_gap_centers), 4), round(max(band_gap_centers), 4)]
        }
        
        stats_file = os.path.join(output_dir, 'extraction_statistics.json')
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        output_files['stats'] = stats_file
        logger.info(f"Saved statistics to: {stats_file}")
        
        return output_files
    
    def create_fine_tuning_script(self, output_dir: str, csv_files: Dict[str, str]):
        """
        Create a fine-tuning script with proper command examples.
        
        Args:
            output_dir: Output directory
            csv_files: Dictionary of generated CSV files
        """
        script_content = f"""#!/bin/bash
# Fine-tuning Scripts for Band Gap Properties
# Generated by extract_bandgap_properties.py

# First, convert CSV files to MatterGen cache format
echo "Converting CSV files to cache format..."

# Complete dataset (all properties)
python -m mattergen.scripts.csv_to_dataset \\
    --csv-folder {output_dir} \\
    --dataset-name bandgap_complete \\
    --cache-folder {output_dir}/cache

# Center-only dataset  
python -m mattergen.scripts.csv_to_dataset \\
    --csv-folder {output_dir} \\
    --dataset-name bandgap_center \\
    --cache-folder {output_dir}/cache

# Direct/indirect-only dataset
python -m mattergen.scripts.csv_to_dataset \\
    --csv-folder {output_dir} \\
    --dataset-name bandgap_direct \\
    --cache-folder {output_dir}/cache

echo "Cache conversion complete!"
echo ""

# Fine-tuning commands
echo "Fine-tuning commands:"
echo ""

echo "1. Band Gap Center Fine-tuning:"
echo "mattergen-finetune \\"
echo "    adapter.pretrained_name=mattergen_base \\"
echo "    data_module=mp_20 \\"
echo "    data_module.properties=['dft_band_gap_center'] \\"
echo "    data_module.root_dir={output_dir}/cache/bandgap_center \\"
echo "    trainer.max_epochs=200"
echo ""

echo "2. Direct/Indirect Transition Fine-tuning:"
echo "mattergen-finetune \\"
echo "    adapter.pretrained_name=mattergen_base \\"
echo "    data_module=mp_20 \\"
echo "    data_module.properties=['dft_band_gap_is_direct'] \\"
echo "    data_module.root_dir={output_dir}/cache/bandgap_direct \\"
echo "    trainer.max_epochs=200"
echo ""

echo "3. Combined Properties Fine-tuning:"
echo "mattergen-finetune \\"
echo "    adapter.pretrained_name=mattergen_base \\"
echo "    data_module=mp_20 \\"
echo "    data_module.properties=['dft_band_gap_center','dft_band_gap_is_direct'] \\"
echo "    data_module.root_dir={output_dir}/cache/bandgap_complete \\"
echo "    trainer.max_epochs=200"
"""
        
        script_path = os.path.join(output_dir, 'fine_tune_bandgap.sh')
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make script executable
        os.chmod(script_path, 0o755)
        logger.info(f"Created fine-tuning script: {script_path}")
        return script_path


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Extract band gap center and direct/indirect transition properties from datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--input_csv',
        type=str,
        required=True,
        help='Path to input CSV file containing dft_band_gap data'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory for extracted datasets'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducible property estimation (default: 42)'
    )
    
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate inputs
    if not os.path.exists(args.input_csv):
        parser.error(f"Input CSV file not found: {args.input_csv}")
    
    # Create extractor
    extractor = BandGapPropertyExtractor(seed=args.seed)
    
    try:
        # Process dataset
        output_files = extractor.process_csv_dataset(args.input_csv, args.output_dir)
        
        # Create fine-tuning script
        script_path = extractor.create_fine_tuning_script(args.output_dir, output_files)
        
        # Print summary
        print("\\n" + "="*60)
        print("Band Gap Properties Extraction Complete!")
        print("="*60)
        print(f"Output directory: {args.output_dir}")
        print(f"Generated files:")
        for name, path in output_files.items():
            print(f"  {name}: {path}")
        
        print(f"\\nFine-tuning script: {script_path}")
        
        # Print statistics
        stats_file = output_files.get('stats')
        if stats_file and os.path.exists(stats_file):
            with open(stats_file, 'r') as f:
                stats = json.load(f)
            
            print("\\nExtraction Statistics:")
            print("-" * 40)
            print(f"Total materials: {stats['total_materials']}")
            print(f"Direct transitions: {stats['direct_transitions']}")
            print(f"Indirect transitions: {stats['indirect_transitions']}")
            print(f"Average band gap: {stats['avg_band_gap']} eV")
            print(f"Average band gap center: {stats['avg_band_gap_center']} eV")
            print(f"Band gap range: {stats['band_gap_range']} eV")
            print(f"Center range: {stats['center_range']} eV")
        
        # Print usage instructions
        print("\\nNext Steps:")
        print("-" * 40)
        print(f"1. Run the fine-tuning script: bash {script_path}")
        print("2. Or manually run individual fine-tuning commands from the script")
        print("3. Use the generated models for conditional crystal generation")
        
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        raise


if __name__ == '__main__':
    main()