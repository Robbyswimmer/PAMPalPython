#!/usr/bin/env python3
"""
Real Data Loader for Extracted R Data

This module loads the real PAMpal data that was extracted from R data files
and converts it to the Python format used by PAMpal Python.
"""

import os
import sys
import pickle
import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# Add parent directory to path for importing PAMpal modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pampal.acoustic_event import AcousticEvent
from pampal.acoustic_study import AcousticStudy
from pampal.settings import PAMpalSettings


class RealDataLoader:
    """Loader for real PAMpal data extracted from R files."""
    
    def __init__(self, extracted_data_dir: str):
        """
        Initialize the real data loader.
        
        Args:
            extracted_data_dir: Directory containing extracted R data files
        """
        self.data_dir = Path(extracted_data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Extracted data directory not found: {extracted_data_dir}")
    
    def load_extracted_study(self, study_name: str = "exStudy") -> Dict[str, Any]:
        """
        Load the extracted study data and convert to Python format.
        
        Args:
            study_name: Name of the study object (default: exStudy)
            
        Returns:
            Dictionary containing the converted study data
        """
        # Look for the extracted RData file
        rdata_file = self.data_dir / f"{study_name}_raw_extracted.RData"
        
        if not rdata_file.exists():
            raise FileNotFoundError(f"Extracted study file not found: {rdata_file}")
        
        # Load the RData file using pyreadr (the extracted data should be simple enough)
        try:
            import pyreadr
            r_data = pyreadr.read_r(str(rdata_file))
            
            # Get the result object
            if 'result' in r_data:
                extracted_result = r_data['result']
            else:
                # Take the first object
                extracted_result = list(r_data.values())[0]
            
            print(f"Loaded extracted data with keys: {list(extracted_result.columns) if hasattr(extracted_result, 'columns') else 'Not a DataFrame'}")
            
            # Convert to our Python format
            return self._convert_extracted_study(extracted_result)
            
        except Exception as e:
            print(f"Error loading extracted RData: {e}")
            # Try alternative approach - manual R script to convert to JSON
            return self._load_via_r_script(rdata_file)
    
    def _convert_extracted_study(self, extracted_data: Any) -> Dict[str, Any]:
        """Convert extracted R data to Python format."""
        
        if hasattr(extracted_data, 'to_dict'):
            # If it's a pandas DataFrame
            data_dict = extracted_data.to_dict('records')[0] if len(extracted_data) > 0 else {}
        elif hasattr(extracted_data, '__dict__'):
            # If it's an object with attributes
            data_dict = extracted_data.__dict__
        else:
            # Assume it's already a dict-like structure
            data_dict = dict(extracted_data) if extracted_data else {}
        
        print(f"Converting extracted data with top-level keys: {list(data_dict.keys())}")
        
        # Extract the main study data
        if 'extracted_data' in data_dict:
            study_data = data_dict['extracted_data']
        else:
            study_data = data_dict
        
        # Create the converted study structure
        converted_study = {
            'creation_timestamp': datetime.now().isoformat(),
            'created_by': 'PAMpal Python real data loader',
            'format_version': '1.0',
            'source': 'Real R data extraction',
            'original_class': data_dict.get('original_class', ['AcousticStudy']),
            'study': self._create_acoustic_study(study_data),
            'events': self._extract_events(study_data),
            'gps_data': self._extract_gps_data(study_data),
            'source': 'Real R exStudy.rda extraction',
            'description': 'Real AcousticStudy object from R PAMpal package',
            'metadata': {
                'num_events': 0,  # Will be updated below
                'total_detections': 0,  # Will be updated below
                'extraction_timestamp': data_dict.get('extraction_timestamp', datetime.now().isoformat()),
                'r_version': data_dict.get('r_version', 'Unknown'),
                'detection_types': []  # Will be updated below
            }
        }
        
        # Update metadata based on extracted events
        if converted_study['events']:
            converted_study['metadata']['num_events'] = len(converted_study['events'])
            
            total_detections = 0
            detection_types = set()
            
            for event_id, event_data in converted_study['events'].items():
                if 'detectors' in event_data:
                    for detector_type, detector_data in event_data['detectors'].items():
                        if hasattr(detector_data, '__len__'):
                            total_detections += len(detector_data)
                        detection_types.add(detector_type)
            
            converted_study['metadata']['total_detections'] = total_detections
            converted_study['metadata']['detection_types'] = list(detection_types)
        
        return converted_study
    
    def _create_acoustic_study(self, study_data: Dict) -> AcousticStudy:
        """Create AcousticStudy object from extracted data."""
        try:
            # Create a basic AcousticStudy
            study = AcousticStudy()
            
            # Set basic attributes
            if 'id' in study_data:
                study.id = str(study_data['id'])
            else:
                study.id = "RealStudy_Extracted"
            
            # Add other attributes as available
            study.ancillary = study_data.get('ancillary', {})
            
            return study
            
        except Exception as e:
            print(f"Warning: Could not create AcousticStudy object: {e}")
            # Return a mock object
            class MockAcousticStudy:
                def __init__(self):
                    self.id = study_data.get('id', 'RealStudy_Extracted')
                    self.ancillary = study_data.get('ancillary', {})
            
            return MockAcousticStudy()
    
    def _extract_events(self, study_data: Dict) -> Dict[str, Dict]:
        """Extract events from study data."""
        events = {}
        
        if 'events' in study_data:
            events_data = study_data['events']
            
            if isinstance(events_data, dict):
                for event_id, event_info in events_data.items():
                    if isinstance(event_info, dict):
                        events[event_id] = self._convert_event(event_info)
                    else:
                        print(f"Warning: Event {event_id} data is not a dictionary")
        
        return events
    
    def _convert_event(self, event_data: Dict) -> Dict:
        """Convert a single event's data."""
        converted_event = {
            'id': event_data.get('id', 'UnknownEvent'),
            'detectors': {},
            'localizations': event_data.get('localizations', {}),
            'settings': event_data.get('settings', {}),
            'species': event_data.get('species', {}),
            'files': event_data.get('files', {}),
            'ancillary': event_data.get('ancillary', {})
        }
        
        # Extract detector data
        if 'detectors' in event_data:
            detectors_data = event_data['detectors']
            
            if isinstance(detectors_data, dict):
                for detector_name, detector_info in detectors_data.items():
                    converted_event['detectors'][detector_name] = self._convert_detector_data(detector_info)
        
        return converted_event
    
    def _convert_detector_data(self, detector_data: Any) -> pd.DataFrame:
        """Convert detector data to DataFrame."""
        try:
            if isinstance(detector_data, pd.DataFrame):
                return detector_data
            elif isinstance(detector_data, dict):
                # Try to convert dict to DataFrame
                if all(isinstance(v, (list, np.ndarray)) for v in detector_data.values()):
                    return pd.DataFrame(detector_data)
                else:
                    # Single row DataFrame
                    return pd.DataFrame([detector_data])
            elif isinstance(detector_data, list):
                return pd.DataFrame(detector_data)
            else:
                print(f"Warning: Unknown detector data type: {type(detector_data)}")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"Warning: Could not convert detector data: {e}")
            return pd.DataFrame()
    
    def _extract_gps_data(self, study_data: Dict) -> pd.DataFrame:
        """Extract GPS data from study."""
        try:
            if 'gps' in study_data:
                gps_data = study_data['gps']
                
                if isinstance(gps_data, pd.DataFrame):
                    return gps_data
                elif isinstance(gps_data, dict):
                    return pd.DataFrame(gps_data)
                else:
                    print(f"Warning: GPS data type not recognized: {type(gps_data)}")
            
            # Return empty GPS DataFrame with expected columns
            return pd.DataFrame(columns=['UTC', 'latitude', 'longitude', 'depth'])
            
        except Exception as e:
            print(f"Warning: Could not extract GPS data: {e}")
            return pd.DataFrame(columns=['UTC', 'latitude', 'longitude', 'depth'])
    
    def _load_via_r_script(self, rdata_file: Path) -> Dict[str, Any]:
        """Load data via R script conversion to JSON."""
        print("Attempting to load via R script conversion...")
        
        # Create a simple R script to convert RData to JSON
        r_script = f'''
        load("{rdata_file}")
        library(jsonlite)
        
        # Convert result to JSON
        json_output <- toJSON(result, pretty=TRUE, auto_unbox=TRUE, na="null")
        cat(json_output)
        '''
        
        try:
            import subprocess
            import tempfile
            
            # Write R script to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.R', delete=False) as f:
                f.write(r_script)
                script_path = f.name
            
            try:
                # Run R script
                result = subprocess.run(
                    ['Rscript', script_path],
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                if result.returncode == 0:
                    # Parse JSON output
                    json_data = json.loads(result.stdout)
                    return self._convert_extracted_study(json_data)
                else:
                    print(f"R script failed: {result.stderr}")
                    
            finally:
                # Clean up temporary file
                os.unlink(script_path)
                
        except Exception as e:
            print(f"R script conversion failed: {e}")
        
        # Return empty structure as fallback
        return {
            'creation_timestamp': datetime.now().isoformat(),
            'created_by': 'PAMpal Python real data loader (fallback)',
            'format_version': '1.0',
            'source': 'Real R data extraction (partial)',
            'study': None,
            'events': {},
            'gps_data': pd.DataFrame(),
            'metadata': {
                'num_events': 0,
                'total_detections': 0,
                'detection_types': [],
                'status': 'Failed to load - using empty structure'
            }
        }
    
    def save_as_python_format(self, study_data: Dict[str, Any], output_file: str) -> None:
        """Save the converted study data in Python format."""
        output_path = Path(output_file)
        
        # Save as pickle
        pickle_path = output_path.with_suffix('.pkl')
        with open(pickle_path, 'wb') as f:
            pickle.dump(study_data, f)
        print(f"Saved real data as pickle: {pickle_path}")
        
        # Try to save as JSON (if serializable)
        try:
            json_path = output_path.with_suffix('.json')
            
            # Create JSON-serializable version
            json_data = self._make_json_serializable(study_data)
            
            with open(json_path, 'w') as f:
                json.dump(json_data, f, indent=2)
            print(f"Saved real data as JSON: {json_path}")
            
        except Exception as e:
            print(f"Could not save as JSON: {e}")
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert object to JSON-serializable format."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif isinstance(obj, pd.Series):
            return obj.tolist()
        elif isinstance(obj, (datetime, pd.Timestamp)):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            return self._make_json_serializable(obj.__dict__)
        else:
            return str(obj)


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Load and convert real extracted PAMpal data')
    parser.add_argument('extracted_dir', help='Directory containing extracted R data')
    parser.add_argument('output_file', help='Output file for converted Python data')
    parser.add_argument('--study-name', default='exStudy', help='Name of study object to load')
    
    args = parser.parse_args()
    
    print("PAMpal Real Data Loader")
    print("=" * 30)
    print(f"Extracted data directory: {args.extracted_dir}")
    print(f"Output file: {args.output_file}")
    print(f"Study name: {args.study_name}")
    print()
    
    try:
        # Create loader and load data
        loader = RealDataLoader(args.extracted_dir)
        study_data = loader.load_extracted_study(args.study_name)
        
        print("Successfully loaded real study data!")
        
        # Access study ID safely
        study_obj = study_data.get('study')
        if hasattr(study_obj, 'id'):
            study_id = study_obj.id
        else:
            study_id = 'Unknown'
        
        print(f"Study ID: {study_id}")
        print(f"Number of events: {study_data['metadata']['num_events']}")
        print(f"Total detections: {study_data['metadata']['total_detections']}")
        print(f"Detection types: {', '.join(study_data['metadata']['detection_types'])}")
        
        # Save in Python format
        loader.save_as_python_format(study_data, args.output_file)
        
        print(f"\nReal data conversion completed successfully!")
        print(f"Output saved to: {args.output_file}")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())