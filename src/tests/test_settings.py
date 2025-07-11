"""
Unit tests for the PAMpalSettings class.
"""
import unittest
import os
import tempfile
from pathlib import Path

# Import the module to test
from pampal.settings import PAMpalSettings


class TestPAMpalSettings(unittest.TestCase):
    """Test cases for PAMpalSettings class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.settings = PAMpalSettings()
        
        # Create temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        
        # Create a mock binary file
        self.binary_dir = self.temp_path / "binaries"
        self.binary_dir.mkdir(exist_ok=True)
        self.binary_file = self.binary_dir / "test.pgdf"
        with open(self.binary_file, "wb") as f:
            f.write(b"PGDF")  # Write magic number
            f.write(b"\x00\x00\x00\x01")  # Version
            f.write(b"\x00\x00\x00\x00\x00\x00\x00\x01")  # Creation time
            f.write(b"\x00\x00\x00\x00\x00\x00\x00\x02")  # Analysis time
            f.write(b"\x00\x0AClickDetector")  # File type (length 10 followed by string)
            f.write(b"\x00\x06Module")  # Module type
            f.write(b"\x00\x04Test")  # Module name
            f.write(b"\x00\x06Stream1")  # Stream name
        
        # Create a mock database file
        self.db_file = self.temp_path / "test.sqlite3"
        with open(self.db_file, "wb") as f:
            f.write(b"SQLite format 3\x00")  # SQLite header
            
        # Create a mock settings file
        self.settings_file = self.temp_path / "settings.xml"
        with open(self.settings_file, "w") as f:
            f.write("<pamguard>\n<settings>\n<test>value</test>\n</settings>\n</pamguard>")
    
    def tearDown(self):
        """Tear down test fixtures after each test method."""
        self.temp_dir.cleanup()
    
    def test_initialization(self):
        """Test initialization of PAMpalSettings."""
        settings = PAMpalSettings()
        
        # Check default values
        self.assertEqual(settings.db, "")
        self.assertEqual(settings.binaries["folder"], [])
        self.assertEqual(settings.binaries["list"], [])
        self.assertEqual(list(settings.functions.keys()), ["ClickDetector", "WhistlesMoans", "Cepstrum"])
        self.assertEqual(list(settings.calibration.keys()), ["ClickDetector"])
    
    def test_add_database(self):
        """Test adding a database file."""
        # Test with valid file
        self.settings.add_database(str(self.db_file))
        self.assertEqual(self.settings.db, str(self.db_file))
        
        # Test with non-existent file
        with self.assertRaises(FileNotFoundError):
            self.settings.add_database("/nonexistent/path.db")
    
    def test_add_binaries(self):
        """Test adding binary files from a directory."""
        # Test with valid directory
        self.settings.add_binaries(str(self.binary_dir))
        self.assertEqual(len(self.settings.binaries["folder"]), 1)
        self.assertEqual(self.settings.binaries["folder"][0], str(self.binary_dir))
        self.assertEqual(len(self.settings.binaries["list"]), 1)
        self.assertEqual(self.settings.binaries["list"][0], str(self.binary_file))
        
        # Test with non-existent directory
        with self.assertRaises(FileNotFoundError):
            self.settings.add_binaries("/nonexistent/directory")
    
    def test_add_function(self):
        """Test adding processing functions."""
        # Define a test function
        def test_func(data):
            return data
            
        # Add function to supported module
        self.settings.add_function("ClickDetector", "test_func", test_func)
        self.assertIn("test_func", self.settings.functions["ClickDetector"])
        
        # Test with unsupported module
        with self.assertRaises(ValueError):
            self.settings.add_function("UnsupportedModule", "test_func", test_func)
    
    def test_add_calibration(self):
        """Test adding calibration functions."""
        # Define a test function
        def test_cal(data):
            return data
            
        # Add calibration function
        self.settings.add_calibration("ClickDetector", "test_cal", test_cal)
        self.assertIn("test_cal", self.settings.calibration["ClickDetector"])
        
        # Add calibration for new module
        self.settings.add_calibration("NewModule", "test_cal", test_cal)
        self.assertIn("NewModule", self.settings.calibration)
        self.assertIn("test_cal", self.settings.calibration["NewModule"])
    
    def test_add_settings(self):
        """Test adding settings from an XML file."""
        # Test with valid file
        self.settings.add_settings(str(self.settings_file))
        self.assertEqual(self.settings.settings["file"], str(self.settings_file))
        
        # Test with non-existent file
        with self.assertRaises(FileNotFoundError):
            self.settings.add_settings("/nonexistent/settings.xml")
    
    def test_string_representation(self):
        """Test the string representation of PAMpalSettings."""
        # Add sample data
        self.settings.add_database(str(self.db_file))
        self.settings.add_binaries(str(self.binary_dir))
        
        # Get string representation
        settings_str = str(self.settings)
        
        # Check that key info is included
        self.assertIn("database", settings_str)
        self.assertIn("binary folder", settings_str)
        self.assertIn("function", settings_str)


if __name__ == "__main__":
    unittest.main()
