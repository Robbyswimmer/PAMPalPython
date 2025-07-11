"""
Unit tests for PAMpal calibration functionality.

This module tests the calibration system including CalibrationFunction, 
CalibrationLoader, CalibrationManager, and integration with PAMpalSettings.
"""

import unittest
import numpy as np
import pandas as pd
import tempfile
import os

# Import the modules to test
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pampal.calibration import (
    CalibrationFunction, CalibrationLoader, CalibrationManager, 
    CalibrationError, load_calibration_file, apply_calibration_to_spectrum
)
from pampal.settings import PAMpalSettings


class TestCalibrationFunction(unittest.TestCase):
    """Test the CalibrationFunction class."""
    
    def setUp(self):
        """Set up test data for calibration functions."""
        # Create simple test calibration data
        self.frequencies = np.array([1000, 5000, 10000, 20000, 50000])
        self.sensitivities = np.array([90.0, 85.0, 80.0, 75.0, 70.0])
        
    def test_calibration_function_creation(self):
        """Test basic calibration function creation."""
        cal_func = CalibrationFunction(
            frequencies=self.frequencies,
            sensitivities=self.sensitivities,
            unit_type=3,
            name="test_calibration"
        )
        
        self.assertEqual(cal_func.name, "test_calibration")
        self.assertEqual(cal_func.unit_type, 3)
        self.assertEqual(cal_func.scale_factor, 1.0)
        np.testing.assert_array_equal(cal_func.frequencies, self.frequencies)
        np.testing.assert_array_equal(cal_func.sensitivities, self.sensitivities)
        
    def test_calibration_function_units(self):
        """Test different unit types for calibration functions."""
        # Test uPa/FullScale (unit_type=3)
        cal_func_3 = CalibrationFunction(
            frequencies=self.frequencies,
            sensitivities=self.sensitivities,
            unit_type=3
        )
        self.assertEqual(cal_func_3.scale_factor, 1.0)
        
        # Test dB re V/uPa (unit_type=1)
        cal_func_1 = CalibrationFunction(
            frequencies=self.frequencies,
            sensitivities=self.sensitivities,
            unit_type=1,
            voltage_range=5.0
        )
        self.assertEqual(cal_func_1.scale_factor, 5.0)
        
        # Test uPa/Counts (unit_type=2)
        cal_func_2 = CalibrationFunction(
            frequencies=self.frequencies,
            sensitivities=self.sensitivities,
            unit_type=2,
            bit_rate=16
        )
        self.assertEqual(cal_func_2.scale_factor, 2**15)  # 2^(16-1)
        
    def test_calibration_function_errors(self):
        """Test error handling in calibration function creation."""
        # Test mismatched array lengths
        with self.assertRaises(CalibrationError):
            CalibrationFunction(
                frequencies=self.frequencies,
                sensitivities=self.sensitivities[:-1],  # One less element
                unit_type=3
            )
        
        # Test insufficient data points
        with self.assertRaises(CalibrationError):
            CalibrationFunction(
                frequencies=np.array([1000]),
                sensitivities=np.array([90.0]),
                unit_type=3
            )
        
        # Test missing voltage range for unit_type=1
        with self.assertRaises(CalibrationError):
            CalibrationFunction(
                frequencies=self.frequencies,
                sensitivities=self.sensitivities,
                unit_type=1  # Missing voltage_range
            )
        
        # Test missing bit rate for unit_type=2
        with self.assertRaises(CalibrationError):
            CalibrationFunction(
                frequencies=self.frequencies,
                sensitivities=self.sensitivities,
                unit_type=2  # Missing bit_rate
            )
            
    def test_calibration_function_call(self):
        """Test calling calibration function with frequencies."""
        cal_func = CalibrationFunction(
            frequencies=self.frequencies,
            sensitivities=self.sensitivities,
            unit_type=3
        )
        
        # Test single frequency
        result = cal_func(5000)
        self.assertIsInstance(result, float)
        
        # Test array of frequencies
        test_freqs = np.array([2000, 7500, 15000])
        results = cal_func(test_freqs)
        self.assertEqual(len(results), len(test_freqs))
        
        # Test extrapolation beyond data range
        low_freq_result = cal_func(500)  # Below minimum
        high_freq_result = cal_func(100000)  # Above maximum
        self.assertIsInstance(low_freq_result, float)
        self.assertIsInstance(high_freq_result, float)
        
    def test_frequency_and_sensitivity_ranges(self):
        """Test getting frequency and sensitivity ranges."""
        cal_func = CalibrationFunction(
            frequencies=self.frequencies,
            sensitivities=self.sensitivities,
            unit_type=3
        )
        
        freq_min, freq_max = cal_func.get_frequency_range()
        self.assertEqual(freq_min, 1000.0)
        self.assertEqual(freq_max, 50000.0)
        
        sens_min, sens_max = cal_func.get_sensitivity_range()
        self.assertEqual(sens_min, 70.0)
        self.assertEqual(sens_max, 90.0)


class TestCalibrationLoader(unittest.TestCase):
    """Test the CalibrationLoader class."""
    
    def setUp(self):
        """Set up test calibration data files."""
        # Create test CSV content
        self.csv_content = """Frequency,Sensitivity
1000,90.0
5000,85.0
10000,80.0
20000,75.0
50000,70.0"""
        
        # Create test CSV with different column names
        self.csv_content_alt = """Freq,dB
1000,90.0
5000,85.0
10000,80.0
20000,75.0
50000,70.0"""
        
        # Create test CSV without headers
        self.csv_content_no_header = """1000,90.0
5000,85.0
10000,80.0
20000,75.0
50000,70.0"""
        
    def test_load_from_csv_standard(self):
        """Test loading calibration from standard CSV format."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(self.csv_content)
            temp_file = f.name
        
        try:
            frequencies, sensitivities = CalibrationLoader.load_from_csv(temp_file)
            
            expected_freqs = np.array([1000, 5000, 10000, 20000, 50000])
            expected_sens = np.array([90.0, 85.0, 80.0, 75.0, 70.0])
            
            np.testing.assert_array_equal(frequencies, expected_freqs)
            np.testing.assert_array_equal(sensitivities, expected_sens)
            
        finally:
            os.unlink(temp_file)
    
    def test_load_from_csv_alt_columns(self):
        """Test loading calibration with alternative column names."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(self.csv_content_alt)
            temp_file = f.name
        
        try:
            frequencies, sensitivities = CalibrationLoader.load_from_csv(temp_file)
            
            expected_freqs = np.array([1000, 5000, 10000, 20000, 50000])
            expected_sens = np.array([90.0, 85.0, 80.0, 75.0, 70.0])
            
            np.testing.assert_array_equal(frequencies, expected_freqs)
            np.testing.assert_array_equal(sensitivities, expected_sens)
            
        finally:
            os.unlink(temp_file)
    
    def test_load_from_csv_no_header(self):
        """Test loading calibration from CSV without headers."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(self.csv_content_no_header)
            temp_file = f.name
        
        try:
            frequencies, sensitivities = CalibrationLoader.load_from_csv(temp_file)
            
            expected_freqs = np.array([1000, 5000, 10000, 20000, 50000])
            expected_sens = np.array([90.0, 85.0, 80.0, 75.0, 70.0])
            
            np.testing.assert_array_equal(frequencies, expected_freqs)
            np.testing.assert_array_equal(sensitivities, expected_sens)
            
        finally:
            os.unlink(temp_file)
    
    def test_load_from_csv_errors(self):
        """Test error handling in CSV loading."""
        # Test non-existent file
        with self.assertRaises(CalibrationError):
            CalibrationLoader.load_from_csv("nonexistent_file.csv")
    
    def test_load_from_dataframe(self):
        """Test loading calibration from pandas DataFrame."""
        df = pd.DataFrame({
            'Frequency': [1000, 5000, 10000, 20000, 50000],
            'Sensitivity': [90.0, 85.0, 80.0, 75.0, 70.0]
        })
        
        frequencies, sensitivities = CalibrationLoader.load_from_dataframe(df)
        
        expected_freqs = np.array([1000, 5000, 10000, 20000, 50000])
        expected_sens = np.array([90.0, 85.0, 80.0, 75.0, 70.0])
        
        np.testing.assert_array_equal(frequencies, expected_freqs)
        np.testing.assert_array_equal(sensitivities, expected_sens)


class TestCalibrationManager(unittest.TestCase):
    """Test the CalibrationManager class."""
    
    def setUp(self):
        """Set up test calibration manager."""
        self.manager = CalibrationManager()
        
        # Create test calibration functions
        frequencies = np.array([1000, 5000, 10000, 20000, 50000])
        sensitivities = np.array([90.0, 85.0, 80.0, 75.0, 70.0])
        
        self.cal_func_1 = CalibrationFunction(
            frequencies=frequencies,
            sensitivities=sensitivities,
            unit_type=3,
            name="test_cal_1"
        )
        
        self.cal_func_2 = CalibrationFunction(
            frequencies=frequencies,
            sensitivities=sensitivities - 5,  # Different sensitivities
            unit_type=3,
            name="test_cal_2"
        )
    
    def test_add_and_get_calibration(self):
        """Test adding and retrieving calibration functions."""
        # Add calibration
        self.manager.add_calibration("ClickDetector", "cal1", self.cal_func_1)
        
        # Retrieve calibration
        retrieved = self.manager.get_calibration("ClickDetector", "cal1")
        self.assertEqual(retrieved, self.cal_func_1)
        
        # Test getting first calibration when name not specified
        first_cal = self.manager.get_calibration("ClickDetector")
        self.assertEqual(first_cal, self.cal_func_1)
    
    def test_has_calibration(self):
        """Test checking if calibration exists."""
        # Initially no calibrations
        self.assertFalse(self.manager.has_calibration("ClickDetector"))
        self.assertFalse(self.manager.has_calibration("ClickDetector", "cal1"))
        
        # Add calibration
        self.manager.add_calibration("ClickDetector", "cal1", self.cal_func_1)
        
        # Now should exist
        self.assertTrue(self.manager.has_calibration("ClickDetector"))
        self.assertTrue(self.manager.has_calibration("ClickDetector", "cal1"))
        self.assertFalse(self.manager.has_calibration("ClickDetector", "cal2"))
    
    def test_remove_calibration(self):
        """Test removing calibration functions."""
        # Add calibrations
        self.manager.add_calibration("ClickDetector", "cal1", self.cal_func_1)
        self.manager.add_calibration("ClickDetector", "cal2", self.cal_func_2)
        
        # Remove specific calibration
        removed = self.manager.remove_calibration("ClickDetector", "cal1")
        self.assertTrue(removed)
        self.assertFalse(self.manager.has_calibration("ClickDetector", "cal1"))
        self.assertTrue(self.manager.has_calibration("ClickDetector", "cal2"))
        
        # Remove all calibrations for detector type
        removed = self.manager.remove_calibration("ClickDetector")
        self.assertTrue(removed)
        self.assertFalse(self.manager.has_calibration("ClickDetector"))
    
    def test_list_calibrations(self):
        """Test listing all calibrations."""
        # Initially empty
        calibrations = self.manager.list_calibrations()
        self.assertEqual(calibrations, {})
        
        # Add calibrations
        self.manager.add_calibration("ClickDetector", "cal1", self.cal_func_1)
        self.manager.add_calibration("ClickDetector", "cal2", self.cal_func_2)
        self.manager.add_calibration("WhistlesMoans", "cal1", self.cal_func_1)
        
        calibrations = self.manager.list_calibrations()
        expected = {
            "ClickDetector": ["cal1", "cal2"],
            "WhistlesMoans": ["cal1"]
        }
        self.assertEqual(calibrations, expected)


class TestPAMpalSettingsCalibration(unittest.TestCase):
    """Test calibration integration with PAMpalSettings."""
    
    def setUp(self):
        """Set up test PAMpalSettings."""
        self.settings = PAMpalSettings()
        
        # Create test calibration CSV
        self.csv_content = """Frequency,Sensitivity
1000,90.0
5000,85.0
10000,80.0
20000,75.0
50000,70.0"""
    
    def test_add_calibration_file(self):
        """Test adding calibration from file through PAMpalSettings."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(self.csv_content)
            temp_file = f.name
        
        try:
            # Add calibration file
            self.settings.add_calibration_file(
                file_path=temp_file,
                module="ClickDetector",
                unit_type=3
            )
            
            # Check that calibration was added
            self.assertTrue(self.settings.has_calibration("ClickDetector"))
            
            # Get calibration function
            cal_func = self.settings.get_calibration("ClickDetector")
            self.assertIsInstance(cal_func, CalibrationFunction)
            
            # Test calibration function
            result = cal_func(5000)
            self.assertIsInstance(result, float)
            
        finally:
            os.unlink(temp_file)
    
    def test_calibration_management_methods(self):
        """Test calibration management methods in PAMpalSettings."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(self.csv_content)
            temp_file = f.name
        
        try:
            # Add calibration
            self.settings.add_calibration_file(
                file_path=temp_file,
                module="ClickDetector",
                name="hydrophone_cal"
            )
            
            # Test has_calibration
            self.assertTrue(self.settings.has_calibration("ClickDetector"))
            self.assertTrue(self.settings.has_calibration("ClickDetector", "hydrophone_cal"))
            
            # Test list_calibrations
            calibrations = self.settings.list_calibrations()
            self.assertIn("ClickDetector", calibrations)
            self.assertIn("hydrophone_cal", calibrations["ClickDetector"])
            
            # Test remove_calibration
            removed = self.settings.remove_calibration("ClickDetector", "hydrophone_cal")
            self.assertTrue(removed)
            self.assertFalse(self.settings.has_calibration("ClickDetector", "hydrophone_cal"))
            
        finally:
            os.unlink(temp_file)
    
    def test_apply_to_all_modules(self):
        """Test applying calibration to all detector modules."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(self.csv_content)
            temp_file = f.name
        
        try:
            # Add calibration to all modules
            self.settings.add_calibration_file(
                file_path=temp_file,
                module="ClickDetector",
                apply_to_all=True
            )
            
            # Check that calibration was added to all modules
            for module in self.settings.functions.keys():
                self.assertTrue(self.settings.has_calibration(module))
                
        finally:
            os.unlink(temp_file)


class TestCalibrationApplication(unittest.TestCase):
    """Test calibration application to spectra and signals."""
    
    def setUp(self):
        """Set up test data for calibration application."""
        frequencies = np.array([1000, 5000, 10000, 20000, 50000])
        sensitivities = np.array([90.0, 85.0, 80.0, 75.0, 70.0])
        
        self.cal_func = CalibrationFunction(
            frequencies=frequencies,
            sensitivities=sensitivities,
            unit_type=3,
            name="test_calibration"
        )
        
        # Create test spectrum
        self.test_frequencies = np.linspace(1000, 50000, 100)
        self.test_spectrum = np.random.randn(100) * 10 + 80  # Random spectrum around 80 dB
    
    def test_apply_calibration_to_spectrum(self):
        """Test applying calibration to power spectrum."""
        calibrated_spectrum = apply_calibration_to_spectrum(
            self.test_frequencies, 
            self.test_spectrum, 
            self.cal_func
        )
        
        # Check that calibrated spectrum has same shape
        self.assertEqual(calibrated_spectrum.shape, self.test_spectrum.shape)
        
        # Check that calibration was applied (values should be different)
        self.assertFalse(np.array_equal(calibrated_spectrum, self.test_spectrum))
        
        # Check that all values are finite
        self.assertTrue(np.all(np.isfinite(calibrated_spectrum)))


class TestCalibrationUtilities(unittest.TestCase):
    """Test calibration utility functions."""
    
    def test_load_calibration_file(self):
        """Test the convenience function for loading calibration files."""
        csv_content = """Frequency,Sensitivity
1000,90.0
5000,85.0
10000,80.0
20000,75.0
50000,70.0"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            temp_file = f.name
        
        try:
            cal_func = load_calibration_file(temp_file, unit_type=3)
            
            self.assertIsInstance(cal_func, CalibrationFunction)
            self.assertEqual(cal_func.unit_type, 3)
            
            # Test function works
            result = cal_func(5000)
            self.assertIsInstance(result, float)
            
        finally:
            os.unlink(temp_file)


if __name__ == '__main__':
    unittest.main()