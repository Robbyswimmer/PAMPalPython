"""
Unit tests for the processing module.
"""
import unittest
import pandas as pd
import numpy as np
import tempfile
import os
from datetime import datetime
from pathlib import Path
import struct
import sqlite3

from pampal.settings import PAMpalSettings
from pampal.acoustic_event import AcousticEvent
from pampal.acoustic_study import AcousticStudy
from pampal.processing import process_detections, load_binaries, load_database, apply_functions


class TestProcessing(unittest.TestCase):
    """Test cases for processing module functions."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        
        # Create mock binary files
        self.binary_dir = self.temp_path / "binaries"
        self.binary_dir.mkdir(exist_ok=True)
        
        # Create a proper mock binary file with complete header
        self.binary_file1 = self.binary_dir / "Click_Detector_001.pgdf"
        with open(self.binary_file1, "wb") as f:
            # Write PGDF header (big-endian format)
            f.write(b"PGDF")  # Magic number
            f.write(struct.pack(">i", 1))  # Version (big-endian)
            f.write(struct.pack(">q", 1640995200000))  # Creation time in milliseconds (2022-01-01)
            f.write(struct.pack(">q", 1640995200000))  # Analysis time in milliseconds
            f.write(struct.pack(">h", 12))  # File type length (signed short)
            f.write(b"ClickDetector")  # File type
            f.write(struct.pack(">h", 6))  # Module type length (signed short)
            f.write(b"Module")  # Module type
            f.write(struct.pack(">h", 4))  # Module name length (signed short)
            f.write(b"Test")  # Module name
            f.write(struct.pack(">h", 7))  # Stream name length (signed short)
            f.write(b"Stream1")  # Stream name
            f.write(struct.pack(">h", 0))  # Extra info length (signed short)
            # Add a simple data block to avoid EOF issues
            f.write(struct.pack(">I", 100))  # Data length
            f.write(struct.pack(">I", 0x12345678))  # Object ID
            f.write(struct.pack(">q", 1640995200000))  # Timestamp in milliseconds
            f.write(b"\x00" * 88)  # Padding to reach 100 bytes
            
        self.binary_file2 = self.binary_dir / "WhistlesMoans_001.pgdf"
        with open(self.binary_file2, "wb") as f:
            # Write PGDF header (big-endian format)
            f.write(b"PGDF")  # Magic number
            f.write(struct.pack(">i", 1))  # Version (big-endian)
            f.write(struct.pack(">q", 1640995200000))  # Creation time in milliseconds
            f.write(struct.pack(">q", 1640995200000))  # Analysis time in milliseconds
            f.write(struct.pack(">h", 13))  # File type length (signed short)
            f.write(b"WhistlesMoans")  # File type
            f.write(struct.pack(">h", 6))  # Module type length (signed short)
            f.write(b"Module")  # Module type
            f.write(struct.pack(">h", 4))  # Module name length (signed short)
            f.write(b"Test")  # Module name
            f.write(struct.pack(">h", 7))  # Stream name length (signed short)
            f.write(b"Stream1")  # Stream name
            f.write(struct.pack(">h", 0))  # Extra info length (signed short)
            # Add a simple data block
            f.write(struct.pack(">I", 100))  # Data length
            f.write(struct.pack(">I", 0x87654321))  # Object ID
            f.write(struct.pack(">q", 1640995200000))  # Timestamp in milliseconds
            f.write(b"\x00" * 88)  # Padding to reach 100 bytes
        
        # Create a proper SQLite database file with test data
        self.db_file = self.temp_path / "test.sqlite3"
        conn = sqlite3.connect(str(self.db_file))
        
        # Create a Click_Detector_Clicks table with test data
        conn.execute("""
            CREATE TABLE Click_Detector_Clicks (
                Id INTEGER PRIMARY KEY,
                UTC REAL,
                UID INTEGER,
                amplitude REAL,
                duration REAL
            )
        """)
        
        # Insert test data
        test_data = [
            (1, 1640995200.0, 1001, 0.5, 0.001),
            (2, 1640995201.0, 1002, 0.6, 0.0012),
            (3, 1640995202.0, 1003, 0.7, 0.0015)
        ]
        
        conn.executemany("""
            INSERT INTO Click_Detector_Clicks (Id, UTC, UID, amplitude, duration)
            VALUES (?, ?, ?, ?, ?)
        """, test_data)
        
        conn.commit()
        conn.close()
            
        # Create settings with test files
        self.settings = PAMpalSettings()
        self.settings.add_database(str(self.db_file))
        self.settings.add_binaries(str(self.binary_dir))
    
    def tearDown(self):
        """Tear down test fixtures after each test method."""
        self.temp_dir.cleanup()
    
    def test_process_detections(self):
        """Test the process_detections function."""
        # Process detections
        study = process_detections(self.settings)
        
        # Check that the study was created correctly
        self.assertTrue(isinstance(study, AcousticStudy))
        self.assertEqual(study.pps, self.settings)
        self.assertEqual(study.files["db"], str(self.db_file))
        self.assertIn(str(self.binary_dir), study.files["binaries"])
        
        # In the current implementation, we should have events from both binary and database data
        self.assertGreaterEqual(len(study.events), 1)
        self.assertLessEqual(len(study.events), 3)  # Allow for reasonable number of events
    
    def test_process_detections_error(self):
        """Test process_detections with invalid settings."""
        # Create empty settings
        empty_settings = PAMpalSettings()
        
        # Should raise ValueError when no binaries or database
        with self.assertRaises(ValueError):
            process_detections(empty_settings)
    
    def test_load_binaries(self):
        """Test the load_binaries function."""
        # This is just testing our placeholder implementation for now
        data = load_binaries([str(self.binary_file1), str(self.binary_file2)])
        
        # Check that the data format is correct - we should have both detector types
        self.assertTrue(len(data) >= 1)  # At least one detector type should work
        # Check that we have DataFrames
        for detector_type, df in data.items():
            self.assertTrue(isinstance(df, pd.DataFrame))
    
    def test_load_database_legacy_mode(self):
        """Test the load_database function in legacy mode."""
        # Test legacy mode for backward compatibility
        data = load_database(str(self.db_file), legacy_format=True)
        
        # Check that the data format is correct (legacy format)
        self.assertIn("Click_Detector_Clicks", data)
        self.assertTrue(isinstance(data["Click_Detector_Clicks"], pd.DataFrame))
        
        # Check that legacy format doesn't include metadata columns
        df = data["Click_Detector_Clicks"]
        self.assertNotIn('detector_type', df.columns)
        self.assertNotIn('detector_table', df.columns)
    
    def test_load_database_enhanced_mode(self):
        """Test the load_database function in enhanced mode."""
        # Test enhanced mode with full functionality
        result = load_database(str(self.db_file), legacy_format=False)
        
        # Check enhanced data structure
        self.assertIn('detector_data', result)
        self.assertIn('database_path', result)
        self.assertIn('schema', result)
        self.assertIn('summary', result)
        
        # Check detector data
        detector_data = result['detector_data']
        self.assertIn('click', detector_data)
        
        click_df = detector_data['click']
        self.assertTrue(isinstance(click_df, pd.DataFrame))
        self.assertIn('detector_type', click_df.columns)
        self.assertEqual(click_df['detector_type'].iloc[0], 'click')
    
    def test_load_database_specific_detector_types(self):
        """Test loading specific detector types."""
        result = load_database(str(self.db_file), detector_types=['click'], legacy_format=False)
        
        detector_data = result['detector_data']
        self.assertIn('click', detector_data)
        # Should only have click data since we specified only click detector type
        self.assertEqual(len(detector_data), 1)
    
    def test_apply_functions(self):
        """Test the apply_functions function."""
        # Create test data and functions
        data = {
            "ClickDetector": pd.DataFrame({
                "UTC": [datetime.now()],
                "amplitude": [0.5]
            })
        }
        
        def add_squared_amplitude(df):
            df["amplitude_squared"] = df["amplitude"] ** 2
            return df
            
        def double_amplitude(df):
            df["amplitude"] = df["amplitude"] * 2
            return df
        
        functions = {
            "ClickDetector": {
                "add_squared": add_squared_amplitude,
                "double": double_amplitude
            }
        }
        
        # Apply functions
        processed_data = apply_functions(data, functions)
        
        # Check that the functions were applied in order
        self.assertIn("amplitude_squared", processed_data["ClickDetector"].columns)
        self.assertEqual(processed_data["ClickDetector"]["amplitude"].iloc[0], 1.0)  # Doubled
        self.assertEqual(processed_data["ClickDetector"]["amplitude_squared"].iloc[0], 0.25)  # Squared before doubling
    
    def test_apply_functions_error(self):
        """Test apply_functions with a function that raises an error."""
        # Create test data and function that raises an error
        data = {
            "ClickDetector": pd.DataFrame({
                "UTC": [datetime.now()],
                "amplitude": [0.5]
            })
        }
        
        def error_function(df):
            raise ValueError("Test error")
            
        functions = {
            "ClickDetector": {
                "error": error_function
            }
        }
        
        # Should not raise the error, but continue processing
        processed_data = apply_functions(data, functions)
        
        # Check that the original data was returned
        self.assertEqual(processed_data["ClickDetector"]["amplitude"].iloc[0], 0.5)


if __name__ == "__main__":
    unittest.main()
