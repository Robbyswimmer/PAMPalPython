#!/usr/bin/env python3
"""
R Script Runner for PAMpal Data Extraction

This module provides an interface to execute R scripts for extracting complex
S4 objects from R data files that cannot be read by pyreadr.
"""

import os
import sys
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RScriptRunner:
    """Interface for running R scripts to extract data from R files."""
    
    def __init__(self, r_executable: str = "Rscript"):
        """
        Initialize R script runner.
        
        Args:
            r_executable: Path to Rscript executable
        """
        self.r_executable = r_executable
        self.check_r_availability()
    
    def check_r_availability(self) -> bool:
        """Check if R is available on the system."""
        try:
            result = subprocess.run(
                [self.r_executable, "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                logger.info(f"R is available: {result.stdout.split()[0]}")
                return True
            else:
                logger.error(f"R check failed: {result.stderr}")
                return False
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.error(f"R not found or not responding: {e}")
            return False
    
    def run_r_script(self, script_path: str, args: List[str], 
                     timeout: int = 300) -> Tuple[int, str, str]:
        """
        Run an R script with arguments.
        
        Args:
            script_path: Path to the R script
            args: List of arguments to pass to the script
            timeout: Timeout in seconds
            
        Returns:
            Tuple of (return_code, stdout, stderr)
        """
        cmd = [self.r_executable, script_path] + args
        
        logger.info(f"Running R command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=os.path.dirname(script_path)
            )
            
            logger.info(f"R script completed with return code: {result.returncode}")
            
            if result.stdout:
                logger.info("R stdout:")
                for line in result.stdout.split('\n'):
                    if line.strip():
                        logger.info(f"  {line}")
            
            if result.stderr:
                logger.warning("R stderr:")
                for line in result.stderr.split('\n'):
                    if line.strip():
                        logger.warning(f"  {line}")
            
            return result.returncode, result.stdout, result.stderr
            
        except subprocess.TimeoutExpired:
            logger.error(f"R script timed out after {timeout} seconds")
            return -1, "", "Script timed out"
        except Exception as e:
            logger.error(f"Error running R script: {e}")
            return -1, "", str(e)


class S4ObjectExtractor:
    """Extractor for S4 objects from R data files using R scripts."""
    
    def __init__(self, r_executable: str = "Rscript"):
        """Initialize the extractor."""
        self.runner = RScriptRunner(r_executable)
        self.script_dir = Path(__file__).parent
        self.extraction_script = self.script_dir / "extract_s4_objects.R"
        self.raw_extraction_script = self.script_dir / "extract_s4_raw.R"
        
        if not self.extraction_script.exists():
            raise FileNotFoundError(f"R extraction script not found: {self.extraction_script}")
        if not self.raw_extraction_script.exists():
            raise FileNotFoundError(f"Raw R extraction script not found: {self.raw_extraction_script}")
    
    def extract_from_rda(self, rda_file: str, output_dir: str) -> Dict:
        """
        Extract S4 objects from an .rda file using R.
        
        Args:
            rda_file: Path to the .rda file
            output_dir: Directory to save extracted data
            
        Returns:
            Dictionary with extraction results
        """
        rda_path = Path(rda_file)
        output_path = Path(output_dir)
        
        if not rda_path.exists():
            raise FileNotFoundError(f"R data file not found: {rda_file}")
        
        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Extracting S4 objects from {rda_file}")
        logger.info(f"Output directory: {output_dir}")
        
        # Try raw extraction first (no package dependencies)
        logger.info("Trying raw S4 extraction (no package loading)...")
        return_code, stdout, stderr = self.runner.run_r_script(
            str(self.raw_extraction_script),
            [str(rda_path.absolute()), str(output_path.absolute())]
        )
        
        # If raw extraction fails, try the full extraction
        if return_code != 0:
            logger.warning("Raw extraction failed, trying full extraction...")
            return_code, stdout, stderr = self.runner.run_r_script(
                str(self.extraction_script),
                [str(rda_path.absolute()), str(output_path.absolute())]
            )
        
        # Parse results
        result = {
            'success': return_code == 0,
            'return_code': return_code,
            'stdout': stdout,
            'stderr': stderr,
            'input_file': str(rda_path),
            'output_dir': str(output_path),
            'extracted_files': []
        }
        
        if return_code == 0:
            # Scan output directory for extracted files
            result['extracted_files'] = self._scan_extracted_files(output_path)
            logger.info(f"Successfully extracted {len(result['extracted_files'])} files")
        else:
            logger.error(f"R extraction failed with code {return_code}")
            if stderr:
                logger.error(f"Error details: {stderr}")
        
        return result
    
    def _scan_extracted_files(self, output_dir: Path) -> List[Dict]:
        """Scan output directory for extracted files."""
        extracted_files = []
        
        for file_path in output_dir.rglob("*"):
            if file_path.is_file():
                file_info = {
                    'path': str(file_path),
                    'relative_path': str(file_path.relative_to(output_dir)),
                    'size': file_path.stat().st_size,
                    'extension': file_path.suffix
                }
                extracted_files.append(file_info)
        
        return extracted_files


class RDataConverter:
    """Enhanced R data converter with multiple extraction methods."""
    
    def __init__(self, fallback_to_synthetic: bool = False):
        """
        Initialize the converter.
        
        Args:
            fallback_to_synthetic: Whether to create synthetic data if real extraction fails
        """
        self.fallback_to_synthetic = fallback_to_synthetic
        self.extractor = None
        
        # Try to initialize R-based extractor
        try:
            self.extractor = S4ObjectExtractor()
            logger.info("R-based extraction available")
        except Exception as e:
            logger.warning(f"R-based extraction not available: {e}")
    
    def convert_with_fallback(self, rda_file: str, output_dir: str) -> Dict:
        """
        Convert R data file with multiple fallback methods.
        
        Args:
            rda_file: Path to the .rda file
            output_dir: Directory to save converted data
            
        Returns:
            Dictionary with conversion results
        """
        methods_tried = []
        final_result = None
        
        # Method 1: Try pyreadr first (fastest)
        try:
            import pyreadr
            logger.info("Trying pyreadr conversion...")
            
            r_data = pyreadr.read_r(rda_file)
            
            # Simple conversion worked
            result = {
                'method': 'pyreadr',
                'success': True,
                'data': r_data,
                'message': 'Successfully converted with pyreadr'
            }
            methods_tried.append(result)
            final_result = result
            logger.info("pyreadr conversion succeeded")
            
        except Exception as e:
            result = {
                'method': 'pyreadr',
                'success': False,
                'error': str(e),
                'message': f'pyreadr failed: {e}'
            }
            methods_tried.append(result)
            logger.warning(f"pyreadr conversion failed: {e}")
        
        # Method 2: Try R script extraction if pyreadr failed
        if not final_result and self.extractor:
            try:
                logger.info("Trying R script extraction...")
                
                extraction_result = self.extractor.extract_from_rda(rda_file, output_dir)
                
                if extraction_result['success']:
                    result = {
                        'method': 'r_script',
                        'success': True,
                        'extraction_result': extraction_result,
                        'message': 'Successfully extracted with R script'
                    }
                    final_result = result
                    logger.info("R script extraction succeeded")
                else:
                    result = {
                        'method': 'r_script',
                        'success': False,
                        'extraction_result': extraction_result,
                        'message': 'R script extraction failed'
                    }
                
                methods_tried.append(result)
                
            except Exception as e:
                result = {
                    'method': 'r_script',
                    'success': False,
                    'error': str(e),
                    'message': f'R script extraction failed: {e}'
                }
                methods_tried.append(result)
                logger.error(f"R script extraction failed: {e}")
        
        # Method 3: Fallback to synthetic data if enabled and all else failed
        if not final_result and self.fallback_to_synthetic:
            try:
                logger.info("Creating synthetic data fallback...")
                
                # Import and use synthetic data generator
                sys.path.append(str(Path(__file__).parent))
                from create_synthetic_data import create_example_study
                
                synthetic_data = create_example_study()
                
                result = {
                    'method': 'synthetic',
                    'success': True,
                    'data': synthetic_data,
                    'message': 'Created synthetic data as fallback'
                }
                methods_tried.append(result)
                final_result = result
                logger.info("Synthetic data fallback succeeded")
                
            except Exception as e:
                result = {
                    'method': 'synthetic',
                    'success': False,
                    'error': str(e),
                    'message': f'Synthetic data creation failed: {e}'
                }
                methods_tried.append(result)
                logger.error(f"Synthetic data creation failed: {e}")
        
        # Compile final result
        conversion_result = {
            'input_file': rda_file,
            'output_dir': output_dir,
            'methods_tried': methods_tried,
            'final_result': final_result,
            'success': final_result is not None and final_result['success']
        }
        
        return conversion_result


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract S4 objects from R data files')
    parser.add_argument('input_file', help='Input .rda file')
    parser.add_argument('output_dir', help='Output directory for extracted data')
    parser.add_argument('--r-executable', default='Rscript', help='Path to Rscript executable')
    parser.add_argument('--fallback-synthetic', action='store_true', 
                       help='Create synthetic data if extraction fails')
    
    args = parser.parse_args()
    
    # Create converter and run extraction
    converter = RDataConverter(fallback_to_synthetic=args.fallback_synthetic)
    result = converter.convert_with_fallback(args.input_file, args.output_dir)
    
    # Print results
    print("\nConversion Results:")
    print("=" * 50)
    print(f"Input file: {result['input_file']}")
    print(f"Output directory: {result['output_dir']}")
    print(f"Success: {result['success']}")
    print()
    
    print("Methods tried:")
    for i, method in enumerate(result['methods_tried'], 1):
        print(f"{i}. {method['method']}: {'SUCCESS' if method['success'] else 'FAILED'}")
        print(f"   {method['message']}")
    
    if result['success']:
        print(f"\nFinal method used: {result['final_result']['method']}")
        return 0
    else:
        print("\nAll conversion methods failed!")
        return 1


if __name__ == '__main__':
    sys.exit(main())