"""
Calibration module for PAMpal Python package.

This module provides comprehensive calibration functionality for acoustic data processing,
including loading calibration data from CSV files, creating calibration functions,
and applying frequency-dependent calibration to acoustic measurements.

The calibration system supports multiple unit types and provides seamless integration
with the PAMpal processing pipeline while maintaining compatibility with the R implementation.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from scipy.interpolate import interp1d
import warnings


class CalibrationError(Exception):
    """Custom exception for calibration-related errors."""
    pass


class CalibrationFunction:
    """
    A calibration function that applies frequency-dependent corrections to acoustic data.
    
    This class encapsulates a calibration function created from frequency-sensitivity data,
    providing interpolation and scaling capabilities for acoustic measurements.
    
    Attributes:
        frequencies: Array of frequency values (Hz)
        sensitivities: Array of sensitivity values (dB)
        unit_type: Type of calibration units (1=dB re V/uPa, 2=uPa/Counts, 3=uPa/FullScale)
        scale_factor: Scaling factor applied to calibration values
        interpolator: Scipy interpolation function for frequency interpolation
    """
    
    def __init__(self, frequencies: np.ndarray, sensitivities: np.ndarray, 
                 unit_type: int = 3, voltage_range: Optional[float] = None, 
                 bit_rate: Optional[int] = None, name: str = "calibration"):
        """
        Initialize a CalibrationFunction.
        
        Args:
            frequencies: Array of frequency values in Hz
            sensitivities: Array of sensitivity values in dB
            unit_type: Unit type (1=dB re V/uPa, 2=uPa/Counts, 3=uPa/FullScale)
            voltage_range: Voltage range for unit_type=1 (required for dB re V/uPa)
            bit_rate: Bit rate for unit_type=2 (required for uPa/Counts)
            name: Name identifier for this calibration function
            
        Raises:
            CalibrationError: If required parameters are missing or invalid
        """
        self.name = name
        self.unit_type = unit_type
        self.frequencies = np.array(frequencies)
        self.sensitivities = np.array(sensitivities)
        
        # Validate inputs
        if len(self.frequencies) != len(self.sensitivities):
            raise CalibrationError("Frequencies and sensitivities must have the same length")
        
        if len(self.frequencies) < 2:
            raise CalibrationError("At least 2 calibration points are required")
        
        # Calculate scale factor based on unit type
        self.scale_factor = self._calculate_scale_factor(unit_type, voltage_range, bit_rate)
        
        # Create interpolation function
        self.interpolator = self._create_interpolator()
        
    def _calculate_scale_factor(self, unit_type: int, voltage_range: Optional[float], 
                              bit_rate: Optional[int]) -> float:
        """Calculate scale factor based on unit type and parameters."""
        if unit_type == 1:  # dB re V/uPa
            if voltage_range is None:
                raise CalibrationError("Voltage range is required for dB re V/uPa units")
            return voltage_range
        elif unit_type == 2:  # uPa/Counts
            if bit_rate is None:
                raise CalibrationError("Bit rate is required for uPa/Counts units")
            return 2**(bit_rate - 1)
        elif unit_type == 3:  # uPa/FullScale
            return 1.0
        else:
            raise CalibrationError(f"Unsupported unit type: {unit_type}")
    
    def _create_interpolator(self) -> interp1d:
        """Create scipy interpolation function for frequency interpolation."""
        # Sort by frequency to ensure proper interpolation
        sort_idx = np.argsort(self.frequencies)
        sorted_freq = self.frequencies[sort_idx]
        sorted_sens = self.sensitivities[sort_idx]
        
        # Handle sign conversion (R implementation feature)
        if np.median(sorted_sens) < 0:
            sorted_sens = -sorted_sens
            warnings.warn("Negative sensitivity values detected, converting to positive")
        
        # Create interpolation function with extrapolation
        interpolator = interp1d(
            sorted_freq, sorted_sens, 
            kind='linear', 
            bounds_error=False,
            fill_value='extrapolate'
        )
        
        return interpolator
    
    def __call__(self, frequencies: Union[float, np.ndarray], **kwargs) -> Union[float, np.ndarray]:
        """
        Apply calibration to frequency or array of frequencies.
        
        Args:
            frequencies: Frequency value(s) in Hz to calibrate
            **kwargs: Additional keyword arguments (for compatibility)
            
        Returns:
            Calibration value(s) in dB
        """
        # Ensure frequencies is numpy array
        freq_array = np.atleast_1d(frequencies)
        
        # Get interpolated sensitivity values
        interpolated_sensitivities = self.interpolator(freq_array)
        
        # Apply scale factor
        calibrated_values = interpolated_sensitivities + 20 * np.log10(self.scale_factor)
        
        # Return scalar if input was scalar
        if np.isscalar(frequencies):
            return float(calibrated_values[0])
        else:
            return calibrated_values
    
    def get_frequency_range(self) -> Tuple[float, float]:
        """Get the frequency range covered by this calibration function."""
        return float(np.min(self.frequencies)), float(np.max(self.frequencies))
    
    def get_sensitivity_range(self) -> Tuple[float, float]:
        """Get the sensitivity range covered by this calibration function."""
        return float(np.min(self.sensitivities)), float(np.max(self.sensitivities))
    
    def __str__(self) -> str:
        """String representation of the calibration function."""
        freq_min, freq_max = self.get_frequency_range()
        sens_min, sens_max = self.get_sensitivity_range()
        unit_names = {1: "dB re V/uPa", 2: "uPa/Counts", 3: "uPa/FullScale"}
        
        return (f"CalibrationFunction '{self.name}'\n"
                f"  Frequency range: {freq_min:.1f} - {freq_max:.1f} Hz\n"
                f"  Sensitivity range: {sens_min:.1f} - {sens_max:.1f} dB\n"
                f"  Unit type: {unit_names.get(self.unit_type, 'Unknown')}\n"
                f"  Scale factor: {self.scale_factor}")


class CalibrationLoader:
    """
    Utility class for loading calibration data from various sources.
    
    This class provides methods for loading calibration data from CSV files,
    DataFrames, or other sources, with validation and error handling.
    """
    
    @staticmethod
    def load_from_csv(file_path: str, frequency_col: str = None, 
                      sensitivity_col: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load calibration data from a CSV file.
        
        Args:
            file_path: Path to the CSV file
            frequency_col: Name of frequency column (auto-detected if None)
            sensitivity_col: Name of sensitivity column (auto-detected if None)
            
        Returns:
            Tuple of (frequencies, sensitivities) as numpy arrays
            
        Raises:
            CalibrationError: If file cannot be loaded or parsed
        """
        if not os.path.exists(file_path):
            raise CalibrationError(f"Calibration file not found: {file_path}")
        
        try:
            # Try to load CSV file
            df = pd.read_csv(file_path)
            
            # Handle case where file has no header
            if df.columns[0].startswith('Unnamed') or df.columns[0].isdigit():
                # Try reloading without header
                df = pd.read_csv(file_path, header=None)
                df.columns = ['Frequency', 'Sensitivity']
            
            # Auto-detect column names if not specified
            if frequency_col is None:
                freq_candidates = ['Frequency', 'Freq', 'frequency', 'freq', 'f', 'F']
                frequency_col = None
                for col in freq_candidates:
                    if col in df.columns:
                        frequency_col = col
                        break
                
                if frequency_col is None:
                    frequency_col = df.columns[0]
                    warnings.warn(f"Frequency column not found, using first column: {frequency_col}")
            
            if sensitivity_col is None:
                sens_candidates = ['Sensitivity', 'Sens', 'sensitivity', 'sens', 'dB', 'db']
                sensitivity_col = None
                for col in sens_candidates:
                    if col in df.columns:
                        sensitivity_col = col
                        break
                
                if sensitivity_col is None:
                    sensitivity_col = df.columns[1]
                    warnings.warn(f"Sensitivity column not found, using second column: {sensitivity_col}")
            
            # Extract data
            frequencies = df[frequency_col].values
            sensitivities = df[sensitivity_col].values
            
            # Validate data
            if not np.all(np.isfinite(frequencies)):
                raise CalibrationError("Invalid frequency values found in calibration file")
            
            if not np.all(np.isfinite(sensitivities)):
                raise CalibrationError("Invalid sensitivity values found in calibration file")
            
            # Convert frequencies to Hz if they appear to be in kHz
            if np.max(frequencies) < 1000:
                frequencies = frequencies * 1000
                warnings.warn("Frequencies appear to be in kHz, converting to Hz")
            
            return frequencies, sensitivities
            
        except Exception as e:
            raise CalibrationError(f"Error loading calibration file {file_path}: {str(e)}")
    
    @staticmethod
    def load_from_dataframe(df: pd.DataFrame, frequency_col: str = None, 
                           sensitivity_col: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load calibration data from a pandas DataFrame.
        
        Args:
            df: DataFrame containing calibration data
            frequency_col: Name of frequency column (auto-detected if None)
            sensitivity_col: Name of sensitivity column (auto-detected if None)
            
        Returns:
            Tuple of (frequencies, sensitivities) as numpy arrays
        """
        if frequency_col is None:
            frequency_col = df.columns[0]
        
        if sensitivity_col is None:
            sensitivity_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
        
        frequencies = df[frequency_col].values
        sensitivities = df[sensitivity_col].values
        
        # Convert frequencies to Hz if they appear to be in kHz
        if np.max(frequencies) < 1000:
            frequencies = frequencies * 1000
            warnings.warn("Frequencies appear to be in kHz, converting to Hz")
        
        return frequencies, sensitivities


class CalibrationManager:
    """
    Manager class for handling multiple calibration functions.
    
    This class provides centralized management of calibration functions for different
    detector types and processing workflows.
    """
    
    def __init__(self):
        """Initialize an empty CalibrationManager."""
        self.calibrations = {}  # Dict[str, Dict[str, CalibrationFunction]]
    
    def add_calibration(self, detector_type: str, name: str, 
                       calibration_function: CalibrationFunction) -> None:
        """
        Add a calibration function for a specific detector type.
        
        Args:
            detector_type: Type of detector (e.g., 'ClickDetector', 'WhistlesMoans')
            name: Name identifier for this calibration function
            calibration_function: CalibrationFunction object
        """
        if detector_type not in self.calibrations:
            self.calibrations[detector_type] = {}
        
        self.calibrations[detector_type][name] = calibration_function
    
    def get_calibration(self, detector_type: str, name: str = None) -> Optional[CalibrationFunction]:
        """
        Get a calibration function for a specific detector type.
        
        Args:
            detector_type: Type of detector
            name: Name of calibration function (returns first if None)
            
        Returns:
            CalibrationFunction object or None if not found
        """
        if detector_type not in self.calibrations:
            return None
        
        if name is None:
            # Return first calibration function if name not specified
            if self.calibrations[detector_type]:
                return list(self.calibrations[detector_type].values())[0]
            return None
        
        return self.calibrations[detector_type].get(name)
    
    def remove_calibration(self, detector_type: str, name: str = None) -> bool:
        """
        Remove a calibration function.
        
        Args:
            detector_type: Type of detector
            name: Name of calibration function (removes all if None)
            
        Returns:
            True if removed, False if not found
        """
        if detector_type not in self.calibrations:
            return False
        
        if name is None:
            # Remove all calibrations for this detector type
            removed = len(self.calibrations[detector_type]) > 0
            self.calibrations[detector_type] = {}
            return removed
        
        if name in self.calibrations[detector_type]:
            del self.calibrations[detector_type][name]
            return True
        
        return False
    
    def list_calibrations(self) -> Dict[str, List[str]]:
        """
        List all available calibration functions.
        
        Returns:
            Dictionary mapping detector types to lists of calibration names
        """
        return {detector_type: list(calibrations.keys()) 
                for detector_type, calibrations in self.calibrations.items()}
    
    def has_calibration(self, detector_type: str, name: str = None) -> bool:
        """
        Check if a calibration function exists.
        
        Args:
            detector_type: Type of detector
            name: Name of calibration function (checks any if None)
            
        Returns:
            True if calibration exists, False otherwise
        """
        if detector_type not in self.calibrations:
            return False
        
        if name is None:
            return len(self.calibrations[detector_type]) > 0
        
        return name in self.calibrations[detector_type]
    
    def __str__(self) -> str:
        """String representation of the CalibrationManager."""
        result = ["CalibrationManager:"]
        
        if not self.calibrations:
            result.append("  No calibrations loaded")
        else:
            for detector_type, calibrations in self.calibrations.items():
                result.append(f"  {detector_type}:")
                for name, cal_func in calibrations.items():
                    freq_min, freq_max = cal_func.get_frequency_range()
                    result.append(f"    {name}: {freq_min:.0f}-{freq_max:.0f} Hz")
        
        return "\n".join(result)


def load_calibration_file(file_path: str, unit_type: int = 3, 
                         voltage_range: Optional[float] = None, 
                         bit_rate: Optional[int] = None,
                         name: str = None) -> CalibrationFunction:
    """
    Convenience function to load a calibration function from a CSV file.
    
    Args:
        file_path: Path to the CSV calibration file
        unit_type: Unit type (1=dB re V/uPa, 2=uPa/Counts, 3=uPa/FullScale)
        voltage_range: Voltage range for unit_type=1
        bit_rate: Bit rate for unit_type=2
        name: Name for the calibration function (defaults to filename)
        
    Returns:
        CalibrationFunction object
    """
    if name is None:
        name = os.path.splitext(os.path.basename(file_path))[0]
    
    # Load data from CSV
    frequencies, sensitivities = CalibrationLoader.load_from_csv(file_path)
    
    # Create and return calibration function
    return CalibrationFunction(
        frequencies=frequencies,
        sensitivities=sensitivities,
        unit_type=unit_type,
        voltage_range=voltage_range,
        bit_rate=bit_rate,
        name=name
    )


def apply_calibration_to_spectrum(frequencies: np.ndarray, spectrum: np.ndarray,
                                 calibration_function: CalibrationFunction) -> np.ndarray:
    """
    Apply calibration to a power spectrum.
    
    Args:
        frequencies: Array of frequency values (Hz)
        spectrum: Array of power spectrum values (dB)
        calibration_function: CalibrationFunction to apply
        
    Returns:
        Calibrated power spectrum (dB)
    """
    calibration_values = calibration_function(frequencies)
    return spectrum + calibration_values


def apply_calibration_to_waveform(waveform: np.ndarray, sample_rate: int,
                                 calibration_function: CalibrationFunction) -> np.ndarray:
    """
    Apply calibration to a time-domain waveform.
    
    This function applies calibration by converting to frequency domain,
    applying calibration, and converting back to time domain.
    
    Args:
        waveform: Time-domain waveform
        sample_rate: Sample rate (Hz)
        calibration_function: CalibrationFunction to apply
        
    Returns:
        Calibrated waveform
    """
    # This is a simplified implementation
    # In practice, calibration is typically applied to spectral measurements
    # rather than time-domain waveforms
    
    # For now, apply a simple scaling based on the calibration function
    # evaluated at the Nyquist frequency
    nyquist_freq = sample_rate / 2
    calibration_value = calibration_function(nyquist_freq)
    
    # Convert dB to linear scale and apply
    linear_scale = 10**(calibration_value / 20)
    return waveform * linear_scale