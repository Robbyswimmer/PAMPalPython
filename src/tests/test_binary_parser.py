"""
Unit tests for the PAMpal binary parser module.

These tests verify the functionality of the binary parser components
including header parsing, data block reading, and different detector type parsing.
"""
import unittest
import os
import tempfile
import struct
import datetime
import pandas as pd
import numpy as np
from io import BytesIO

from pampal.binary_parser import (
    read_pgdf_header,
    read_string,
    read_binary_data,
    read_data_block, 
    read_binary_file,
    read_click_detector_data,
    read_whistles_moans_data,
    read_gpl_detector_data,
    load_binary_files,
    PamBinaryError
)

class TestBinaryParser(unittest.TestCase):
    """Test the binary parser functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create test binary files
        self.click_file = os.path.join(self.temp_dir.name, "Click_Detector_Test.pgdf")
        self.whistle_file = os.path.join(self.temp_dir.name, "WhistlesMoans_Test.pgdf")
        self.gpl_file = os.path.join(self.temp_dir.name, "GPL_Detector_Test.pgdf")
        self.invalid_file = os.path.join(self.temp_dir.name, "Invalid_File.pgdf")
        
        # Create basic test files
        self._create_test_click_file()
        self._create_test_whistle_file()
        self._create_test_gpl_file()
        self._create_invalid_file()
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()
    
    def _create_test_click_file(self):
        """Create a test click detector binary file."""
        with open(self.click_file, 'wb') as f:
            # Write header
            f.write(b'PGDF')                          # Magic number
            f.write(struct.pack('>i', 3))             # Version
            create_time = int(datetime.datetime(2025, 6, 27, 10, 0, 0).timestamp() * 1000)
            analysis_time = int(datetime.datetime(2025, 6, 27, 10, 1, 0).timestamp() * 1000)
            f.write(struct.pack('>q', create_time))    # Create time
            f.write(struct.pack('>q', analysis_time))  # Analysis time
            
            # Write string fields
            self._write_string(f, "Click_Detector_Clicks")  # File type
            self._write_string(f, "ClickDetector")          # Module type
            self._write_string(f, "Click Detector 1")       # Module name
            self._write_string(f, "Clicks")                 # Stream name
            
            data_start_pos = f.tell()  # Remember position where data blocks start
            
            # Write a few data blocks
            for i in range(3):
                # Basic object header
                timestamp = int(datetime.datetime(2025, 6, 27, 10, 0, i).timestamp() * 1000)
                f.write(struct.pack('>q', timestamp))     # Timestamp
                f.write(struct.pack('>i', 1))             # Channel bitmap
                f.write(struct.pack('>i', i))             # Sequence number
                f.write(struct.pack('>q', 1000 + i))      # UID
                
                # Number of data items
                f.write(struct.pack('>h', 3))             # 3 data items
                
                # Data items
                self._write_string(f, "amplitude")
                f.write(struct.pack('B', 4))              # FLOAT
                f.write(struct.pack('>f', 0.5 + i * 0.1)) # Value
                
                self._write_string(f, "duration")
                f.write(struct.pack('B', 4))              # FLOAT
                f.write(struct.pack('>f', 0.001 + i * 0.0005)) # Value
                
                self._write_string(f, "waveform")
                f.write(struct.pack('B', 10))             # ARRAY
                f.write(struct.pack('B', 4))              # FLOAT elements
                f.write(struct.pack('>i', 5))             # 5 elements
                for j in range(5):
                    f.write(struct.pack('>f', 0.1 * j))   # Values
    
    def _create_test_whistle_file(self):
        """Create a test whistle & moan detector binary file."""
        with open(self.whistle_file, 'wb') as f:
            # Write header
            f.write(b'PGDF')                          # Magic number
            f.write(struct.pack('>i', 3))             # Version
            create_time = int(datetime.datetime(2025, 6, 27, 11, 0, 0).timestamp() * 1000)
            analysis_time = int(datetime.datetime(2025, 6, 27, 11, 1, 0).timestamp() * 1000)
            f.write(struct.pack('>q', create_time))    # Create time
            f.write(struct.pack('>q', analysis_time))  # Analysis time
            
            # Write string fields
            self._write_string(f, "WhistlesMoans_Detector")  # File type
            self._write_string(f, "WhistleMoanDetector")     # Module type
            self._write_string(f, "Whistle Detector 1")      # Module name
            self._write_string(f, "Whistles")                # Stream name
            
            data_start_pos = f.tell()  # Remember position where data blocks start
            
            # Write a few data blocks
            for i in range(2):
                # Basic object header
                timestamp = int(datetime.datetime(2025, 6, 27, 11, 0, i).timestamp() * 1000)
                f.write(struct.pack('>q', timestamp))     # Timestamp
                f.write(struct.pack('>i', 1))             # Channel bitmap
                f.write(struct.pack('>i', i))             # Sequence number
                f.write(struct.pack('>q', 2000 + i))      # UID
                
                # Number of data items
                f.write(struct.pack('>h', 2))             # 2 data items
                
                # Data items
                self._write_string(f, "duration")
                f.write(struct.pack('B', 4))              # FLOAT
                f.write(struct.pack('>f', 1.0 + i * 0.5))  # Value
                
                self._write_string(f, "contour")
                f.write(struct.pack('B', 10))             # ARRAY
                f.write(struct.pack('B', 4))              # FLOAT elements
                f.write(struct.pack('>i', 4))             # 4 elements
                for j in range(4):
                    f.write(struct.pack('>f', 1000.0 + j * 100))  # Frequency values
    
    def _create_test_gpl_file(self):
        """Create a test GPL detector binary file."""
        with open(self.gpl_file, 'wb') as f:
            # Write header
            f.write(b'PGDF')                          # Magic number
            f.write(struct.pack('>i', 3))             # Version
            create_time = int(datetime.datetime(2025, 6, 27, 12, 0, 0).timestamp() * 1000)
            analysis_time = int(datetime.datetime(2025, 6, 27, 12, 1, 0).timestamp() * 1000)
            f.write(struct.pack('>q', create_time))    # Create time
            f.write(struct.pack('>q', analysis_time))  # Analysis time
            
            # Write string fields
            self._write_string(f, "GPL_Detector")       # File type
            self._write_string(f, "GPLDetector")        # Module type
            self._write_string(f, "GPL Detector 1")     # Module name
            self._write_string(f, "GPL")                # Stream name
            
            data_start_pos = f.tell()  # Remember position where data blocks start
            
            # Write a data block
            # Basic object header
            timestamp = int(datetime.datetime(2025, 6, 27, 12, 0, 0).timestamp() * 1000)
            f.write(struct.pack('>q', timestamp))      # Timestamp
            f.write(struct.pack('>i', 1))             # Channel bitmap
            f.write(struct.pack('>i', 0))             # Sequence number
            f.write(struct.pack('>q', 3000))          # UID
            
            # Number of data items
            f.write(struct.pack('>h', 2))             # 2 data items
            
            # Data items
            self._write_string(f, "value")
            f.write(struct.pack('B', 4))              # FLOAT
            f.write(struct.pack('>f', 0.75))          # Value
            
            self._write_string(f, "name")
            f.write(struct.pack('B', 8))              # STRING
            self._write_string(f, "TestGPL")          # Value
    
    def _create_invalid_file(self):
        """Create an invalid binary file."""
        with open(self.invalid_file, 'wb') as f:
            f.write(b'INVALID')  # Invalid magic number
            f.write(b'This is not a valid PGDF file')
    
    def _write_string(self, file, string):
        """Write a string to a binary file."""
        bytes_str = string.encode('utf-8')
        file.write(struct.pack('>h', len(bytes_str)))
        file.write(bytes_str)
    
    def test_read_pgdf_header_valid(self):
        """Test reading a valid PGDF header."""
        header = read_pgdf_header(self.click_file)
        
        self.assertEqual(header['version'], 3)
        self.assertEqual(header['file_type'], "Click_Detector_Clicks")
        self.assertEqual(header['module_type'], "ClickDetector")
        self.assertEqual(header['module_name'], "Click Detector 1")
        self.assertEqual(header['stream_name'], "Clicks")
        self.assertIsInstance(header['create_time'], datetime.datetime)
        self.assertEqual(header['create_time'].year, 2025)
        self.assertEqual(header['create_time'].month, 6)
        self.assertEqual(header['create_time'].day, 27)
        self.assertEqual(header['create_time'].hour, 10)
        self.assertEqual(header['create_time'].minute, 0)
        self.assertEqual(header['create_time'].second, 0)
        self.assertIsInstance(header['data_start'], int)
    
    def test_read_pgdf_header_invalid(self):
        """Test reading an invalid PGDF header."""
        with self.assertRaises(PamBinaryError):
            read_pgdf_header(self.invalid_file)
    
    def test_read_pgdf_header_nonexistent(self):
        """Test reading a nonexistent file."""
        with self.assertRaises(FileNotFoundError):
            read_pgdf_header("/nonexistent/file.pgdf")
    
    def test_read_string_from_bytes(self):
        """Test reading a string from a bytes object."""
        # Create a BytesIO object with a string
        buffer = BytesIO()
        self._write_string(buffer, "Hello World")
        buffer.seek(0)
        
        # Read the string
        result = read_string(buffer)
        self.assertEqual(result, "Hello World")
    
    def test_read_binary_data_types(self):
        """Test reading different binary data types."""
        # Test SHORT
        buffer = BytesIO(struct.pack('>h', 12345))
        self.assertEqual(read_binary_data(buffer, 1), 12345)
        
        # Test INT
        buffer = BytesIO(struct.pack('>i', 123456789))
        self.assertEqual(read_binary_data(buffer, 2), 123456789)
        
        # Test LONG
        buffer = BytesIO(struct.pack('>q', 1234567890123456789))
        self.assertEqual(read_binary_data(buffer, 3), 1234567890123456789)
        
        # Test FLOAT
        buffer = BytesIO(struct.pack('>f', 123.456))
        self.assertAlmostEqual(read_binary_data(buffer, 4), 123.456, delta=0.001)
        
        # Test DOUBLE
        buffer = BytesIO(struct.pack('>d', 123.4567890123))
        self.assertAlmostEqual(read_binary_data(buffer, 5), 123.4567890123, delta=0.0000001)
        
        # Test CHAR
        buffer = BytesIO(struct.pack('c', b'A'))
        self.assertEqual(read_binary_data(buffer, 6), 'A')
        
        # Test BOOLEAN
        buffer = BytesIO(struct.pack('?', True))
        self.assertEqual(read_binary_data(buffer, 7), True)
        
        # Test STRING
        buffer = BytesIO()
        self._write_string(buffer, "Test String")
        buffer.seek(0)
        self.assertEqual(read_binary_data(buffer, 8), "Test String")
        
        # Test ARRAY
        buffer = BytesIO()
        buffer.write(struct.pack('B', 4))  # Element type (FLOAT)
        buffer.write(struct.pack('>i', 3))  # Array length
        buffer.write(struct.pack('>f', 1.0))
        buffer.write(struct.pack('>f', 2.0))
        buffer.write(struct.pack('>f', 3.0))
        buffer.seek(0)
        self.assertEqual(read_binary_data(buffer, 10), [1.0, 2.0, 3.0])
    
    def test_read_data_block(self):
        """Test reading a data block from a binary file."""
        with open(self.click_file, 'rb') as f:
            # Skip header
            header = read_pgdf_header(self.click_file)
            f.seek(header['data_start'])
            
            # Read data block
            data_block = read_data_block(f)
            
            # Verify structure
            self.assertIn('UTC', data_block)
            self.assertIn('milliseconds', data_block)
            self.assertIn('channels', data_block)
            self.assertIn('sequence', data_block)
            self.assertIn('UID', data_block)
            self.assertIn('amplitude', data_block)
            self.assertIn('duration', data_block)
            self.assertIn('waveform', data_block)
            
            # Verify types
            self.assertIsInstance(data_block['UTC'], datetime.datetime)
            self.assertIsInstance(data_block['milliseconds'], int)
            self.assertIsInstance(data_block['channels'], int)
            self.assertIsInstance(data_block['sequence'], int)
            self.assertIsInstance(data_block['UID'], int)
            self.assertIsInstance(data_block['amplitude'], float)
            self.assertIsInstance(data_block['duration'], float)
            self.assertIsInstance(data_block['waveform'], list)
            
            # Verify value
            self.assertEqual(data_block['sequence'], 0)
            self.assertEqual(data_block['UID'], 1000)
            self.assertAlmostEqual(data_block['amplitude'], 0.5, delta=0.001)
            self.assertAlmostEqual(data_block['duration'], 0.001, delta=0.0001)
            self.assertEqual(len(data_block['waveform']), 5)
    
    def test_read_click_detector_data(self):
        """Test reading click detector data."""
        df = read_click_detector_data(self.click_file)
        
        # Verify DataFrame structure
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn('UTC', df.columns)
        self.assertIn('UID', df.columns)
        self.assertIn('amplitude', df.columns)
        self.assertIn('duration', df.columns)
        self.assertIn('waveform', df.columns)
        
        # Verify data content
        self.assertEqual(len(df), 3)  # We created 3 data blocks
        self.assertTrue(all(df['UID'].isin([1000, 1001, 1002])))
        
        # Verify non-click file raises error
        with self.assertRaises(PamBinaryError):
            read_click_detector_data(self.whistle_file)
    
    def test_read_whistles_moans_data(self):
        """Test reading whistle and moan detector data."""
        df = read_whistles_moans_data(self.whistle_file)
        
        # Verify DataFrame structure
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn('UTC', df.columns)
        self.assertIn('UID', df.columns)
        self.assertIn('duration', df.columns)
        self.assertIn('contour', df.columns)
        
        # Additional fields computed from contour
        self.assertIn('min_freq', df.columns)
        self.assertIn('max_freq', df.columns)
        self.assertIn('mean_freq', df.columns)
        self.assertIn('contour_points', df.columns)
        
        # Verify data content
        self.assertEqual(len(df), 2)  # We created 2 data blocks
        self.assertTrue(all(df['UID'].isin([2000, 2001])))
        
        # Verify contour processing
        self.assertEqual(df['contour_points'].iloc[0], 4)
        self.assertTrue(1000 <= df['min_freq'].iloc[0] <= df['max_freq'].iloc[0])
        
        # Verify non-whistle file raises error
        with self.assertRaises(PamBinaryError):
            read_whistles_moans_data(self.click_file)
    
    def test_read_gpl_detector_data(self):
        """Test reading GPL detector data."""
        df = read_gpl_detector_data(self.gpl_file)
        
        # Verify DataFrame structure
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn('UTC', df.columns)
        self.assertIn('UID', df.columns)
        self.assertIn('value', df.columns)
        self.assertIn('name', df.columns)
        
        # Verify data content
        self.assertEqual(len(df), 1)  # We created 1 data block
        self.assertEqual(df['UID'].iloc[0], 3000)
        self.assertAlmostEqual(df['value'].iloc[0], 0.75, delta=0.001)
        self.assertEqual(df['name'].iloc[0], "TestGPL")
        
        # Verify non-GPL file raises error
        with self.assertRaises(PamBinaryError):
            read_gpl_detector_data(self.click_file)
    
    def test_read_binary_file(self):
        """Test the generic binary file reader."""
        # Test click file
        df_click = read_binary_file(self.click_file)
        self.assertIsInstance(df_click, pd.DataFrame)
        self.assertEqual(len(df_click), 3)
        
        # Test whistle file
        df_whistle = read_binary_file(self.whistle_file)
        self.assertIsInstance(df_whistle, pd.DataFrame)
        self.assertEqual(len(df_whistle), 2)
        
        # Test GPL file
        df_gpl = read_binary_file(self.gpl_file)
        self.assertIsInstance(df_gpl, pd.DataFrame)
        self.assertEqual(len(df_gpl), 1)
        
        # Test invalid file
        with self.assertRaises(PamBinaryError):
            read_binary_file(self.invalid_file)
    
    def test_load_binary_files(self):
        """Test loading multiple binary files."""
        file_list = [self.click_file, self.whistle_file, self.gpl_file]
        result = load_binary_files(file_list, verbose=False)
        
        # Verify result structure
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 3)  # 3 different detector types
        
        # Check for expected detector types
        detector_types = list(result.keys())
        self.assertTrue(any('Click' in dt for dt in detector_types))
        self.assertTrue(any('WhistlesMoans' in dt for dt in detector_types))
        self.assertTrue(any('GPL' in dt for dt in detector_types))
        
        # Check that each value is a DataFrame with the BinaryFile column added
        for df in result.values():
            self.assertIsInstance(df, pd.DataFrame)
            self.assertIn('BinaryFile', df.columns)


if __name__ == '__main__':
    unittest.main()
