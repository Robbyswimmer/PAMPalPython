#!/usr/bin/env python3
"""
R Data Conversion Script for PAMpal Python

This script converts R data files (.rda) from the original PAMpal R package
to Python-compatible formats (pickle and JSON) for use with PAMpal Python.

Usage:
    python convert_r_data.py [--input-dir INPUT_DIR] [--output-dir OUTPUT_DIR]

The script converts:
- testClick.rda: Click detector test data
- testWhistle.rda: Whistle detector test data  
- testCeps.rda: Cepstrum detector test data
- testGPL.rda: GPL detector test data
- exStudy.rda: Complete example AcousticStudy object
"""

import os
import sys
import pickle
import json
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

try:
    import pyreadr
except ImportError:
    print("Error: pyreadr not installed. Install with: pip install pyreadr")
    sys.exit(1)

# Add parent directory to path for importing PAMpal modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pampal.acoustic_event import AcousticEvent
from pampal.acoustic_study import AcousticStudy
from pampal.settings import PAMpalSettings


class RDataConverter:
    """Converter for R data files to Python formats."""
    
    def __init__(self, input_dir, output_dir):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.conversion_log = []
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def log_conversion(self, filename, status, details=""):
        """Log conversion results."""
        entry = {
            'filename': filename,
            'status': status,
            'details': details,
            'timestamp': datetime.now().isoformat()
        }
        self.conversion_log.append(entry)
        print(f"[{status}] {filename}: {details}")
    
    def convert_all_files(self):
        """Convert all R data files in the input directory."""
        r_files = list(self.input_dir.glob("*.rda"))
        
        if not r_files:
            print(f"No .rda files found in {self.input_dir}")
            return False
            
        print(f"Found {len(r_files)} R data files to convert...")
        
        # Convert each file
        for r_file in r_files:
            try:
                self.convert_single_file(r_file)
            except Exception as e:
                self.log_conversion(r_file.name, "ERROR", str(e))
        
        # Save conversion log
        self.save_conversion_log()
        
        # Print summary
        self.print_summary()
        
        return True
    
    def convert_single_file(self, r_file):
        """Convert a single R data file."""
        print(f"\nConverting {r_file.name}...")
        
        try:
            # Load R data file
            r_data = pyreadr.read_r(str(r_file))
            
            # Get the main object (usually there's one per file)
            if len(r_data) == 1:
                obj_name = list(r_data.keys())[0]
                obj_data = r_data[obj_name]
            else:
                # Multiple objects - handle each
                obj_name = r_file.stem
                obj_data = r_data
            
            # Convert based on filename
            if r_file.name.startswith('test'):
                converted_data = self.convert_test_data(obj_data, r_file.stem)
            elif r_file.name == 'exStudy.rda':
                converted_data = self.convert_study_data(obj_data, r_file.stem)
            else:
                # Generic conversion
                converted_data = self.convert_generic_data(obj_data, r_file.stem)
            
            # Save as pickle
            pickle_path = self.output_dir / f"{r_file.stem}.pkl"
            with open(pickle_path, 'wb') as f:
                pickle.dump(converted_data, f)
            
            # Try to save as JSON (if serializable)
            try:
                json_path = self.output_dir / f"{r_file.stem}.json"
                with open(json_path, 'w') as f:
                    json.dump(converted_data, f, indent=2, default=self.json_serializer)
                
                self.log_conversion(r_file.name, "SUCCESS", f"Saved as .pkl and .json")
            except (TypeError, ValueError):
                self.log_conversion(r_file.name, "PARTIAL", f"Saved as .pkl only (not JSON serializable)")
                
        except Exception as e:
            self.log_conversion(r_file.name, "ERROR", str(e))
            raise
    
    def convert_test_data(self, obj_data, filename):
        """Convert test data objects (testClick, testWhistle, etc.)."""
        converted = {
            'source_file': filename,
            'conversion_timestamp': datetime.now().isoformat(),
            'data_type': 'test_detector_data'
        }
        
        if isinstance(obj_data, dict):
            # Multiple objects in file
            for key, value in obj_data.items():
                converted[key] = self.convert_r_object_to_python(value)
        else:
            # Single object
            converted['data'] = self.convert_r_object_to_python(obj_data)
        
        return converted
    
    def convert_study_data(self, obj_data, filename):
        """Convert AcousticStudy object."""
        converted = {
            'source_file': filename,
            'conversion_timestamp': datetime.now().isoformat(),
            'data_type': 'acoustic_study',
            'description': 'Converted from R PAMpal AcousticStudy object'
        }
        
        # This is complex - we'll need to handle S4 object structure
        # For now, do basic conversion and let user handle the structure
        if isinstance(obj_data, dict):
            for key, value in obj_data.items():
                converted[key] = self.convert_r_object_to_python(value)
        else:
            converted['raw_data'] = self.convert_r_object_to_python(obj_data)
        
        return converted
    
    def convert_generic_data(self, obj_data, filename):
        """Generic conversion for unknown data types."""
        converted = {
            'source_file': filename,
            'conversion_timestamp': datetime.now().isoformat(),
            'data_type': 'generic'
        }
        
        if isinstance(obj_data, dict):
            for key, value in obj_data.items():
                converted[key] = self.convert_r_object_to_python(value)
        else:
            converted['data'] = self.convert_r_object_to_python(obj_data)
        
        return converted
    
    def convert_r_object_to_python(self, obj):
        """Convert R objects to Python equivalents."""
        if obj is None:
            return None
        elif isinstance(obj, (int, float, str, bool)):
            return obj
        elif isinstance(obj, np.ndarray):
            return obj.tolist() if obj.size < 10000 else obj  # Keep large arrays as numpy
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')  # Convert to list of dicts
        elif isinstance(obj, pd.Series):
            return obj.tolist()
        elif isinstance(obj, list):
            return [self.convert_r_object_to_python(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: self.convert_r_object_to_python(value) for key, value in obj.items()}
        else:
            # Try to convert to string representation
            try:
                return str(obj)
            except:
                return f"<Unconvertible R object: {type(obj)}>"
    
    def json_serializer(self, obj):
        """Custom JSON serializer for numpy objects."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        else:
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    def save_conversion_log(self):
        """Save conversion log to file."""
        log_path = self.output_dir / "conversion_log.json"
        with open(log_path, 'w') as f:
            json.dump(self.conversion_log, f, indent=2)
        
        print(f"\nConversion log saved to: {log_path}")
    
    def print_summary(self):
        """Print conversion summary."""
        success_count = sum(1 for entry in self.conversion_log if entry['status'] == 'SUCCESS')
        partial_count = sum(1 for entry in self.conversion_log if entry['status'] == 'PARTIAL')
        error_count = sum(1 for entry in self.conversion_log if entry['status'] == 'ERROR')
        
        print(f"\n{'='*50}")
        print("CONVERSION SUMMARY")
        print(f"{'='*50}")
        print(f"Successful: {success_count}")
        print(f"Partial:    {partial_count}")
        print(f"Errors:     {error_count}")
        print(f"Total:      {len(self.conversion_log)}")
        
        if error_count > 0:
            print(f"\nErrors occurred. Check conversion_log.json for details.")


def main():
    """Main function to run the conversion."""
    parser = argparse.ArgumentParser(description='Convert R data files to Python format')
    parser.add_argument('--input-dir', 
                       default='/Users/robbymoseley/CascadeProjects/PAMpal-main/data',
                       help='Directory containing R data files')
    parser.add_argument('--output-dir',
                       default='./data/real_examples',
                       help='Directory to save converted Python files')
    
    args = parser.parse_args()
    
    print("PAMpal R Data Converter")
    print("=" * 50)
    print(f"Input directory:  {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    
    # Check if input directory exists
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory {args.input_dir} does not exist")
        sys.exit(1)
    
    # Create converter and run conversion
    converter = RDataConverter(args.input_dir, args.output_dir)
    success = converter.convert_all_files()
    
    if success:
        print(f"\nConversion completed! Files saved to: {args.output_dir}")
    else:
        print(f"\nConversion failed!")
        sys.exit(1)


if __name__ == '__main__':
    main()