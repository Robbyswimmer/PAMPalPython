#!/usr/bin/env python3
"""
R Data Inspector Script for PAMpal Python

This script inspects R data files to understand their structure before conversion.
"""

import os
import sys
import pyreadr
from pathlib import Path

def inspect_r_file(file_path):
    """Inspect a single R data file."""
    print(f"\n{'='*60}")
    print(f"Inspecting: {file_path.name}")
    print(f"{'='*60}")
    
    try:
        # Try to read the file
        result = pyreadr.read_r(str(file_path))
        
        print(f"Number of objects in file: {len(result)}")
        
        for obj_name, obj_data in result.items():
            print(f"\nObject: '{obj_name}'")
            print(f"Type: {type(obj_data)}")
            
            if hasattr(obj_data, 'shape'):
                print(f"Shape: {obj_data.shape}")
            
            if hasattr(obj_data, 'columns'):
                print(f"Columns: {list(obj_data.columns)}")
                print(f"Data types:\n{obj_data.dtypes}")
                
                # Show first few rows
                print(f"\nFirst 5 rows:")
                print(obj_data.head())
            
            elif hasattr(obj_data, '__len__'):
                print(f"Length: {len(obj_data)}")
                
                # If it's a list or similar, show first few items
                if len(obj_data) > 0:
                    print(f"First few items: {obj_data[:min(5, len(obj_data))]}")
            
            # Try to show some basic info about the object
            if hasattr(obj_data, 'info'):
                print("\nDataFrame info:")
                obj_data.info()
                
    except Exception as e:
        print(f"Error reading file: {e}")

def main():
    """Main function to inspect all R files."""
    input_dir = Path('/Users/robbymoseley/CascadeProjects/PAMpal-main/data')
    
    r_files = list(input_dir.glob("*.rda"))
    
    if not r_files:
        print(f"No .rda files found in {input_dir}")
        return
    
    print(f"Found {len(r_files)} R data files to inspect...")
    
    for r_file in r_files:
        inspect_r_file(r_file)

if __name__ == '__main__':
    main()