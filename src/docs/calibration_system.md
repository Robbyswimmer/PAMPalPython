# PAMpal Calibration System Documentation

## Overview

The PAMpal calibration system provides comprehensive support for applying frequency-dependent calibration corrections to acoustic measurements. This system is essential for ensuring scientific accuracy when analyzing marine mammal acoustic data, as it accounts for the frequency response characteristics of hydrophones and recording equipment.

The calibration system maintains compatibility with the R PAMpal implementation while leveraging Python's scientific computing capabilities for enhanced performance and flexibility.

## Architecture

The calibration system consists of four main components:

### 1. Core Calibration Classes
- **CalibrationFunction**: Encapsulates calibration data with frequency interpolation
- **CalibrationLoader**: Handles loading calibration data from various sources  
- **CalibrationManager**: Manages multiple calibration functions for different detector types

### 2. Settings Integration
- **PAMpalSettings**: Enhanced with calibration management methods
- Seamless integration with existing workflow
- Backward compatibility with legacy calibration formats

### 3. Signal Processing Integration
- **Spectral Analysis**: Automatic calibration application to spectrograms and power spectra
- **Parameter Calculation**: Calibrated acoustic parameter extraction
- **Waveform Processing**: Optional time-domain calibration application

### 4. Processing Pipeline Integration
- **Automatic Application**: Calibration applied during data processing
- **Function Compatibility**: Processing functions can optionally accept calibration parameters
- **Error Handling**: Robust error handling and fallback mechanisms

## Key Features

### ✅ **Multiple Unit Types**
- **dB re V/uPa**: Voltage-based calibration with voltage range specification
- **uPa/Counts**: Count-based calibration with bit rate specification  
- **uPa/FullScale**: Full-scale calibration (most common)

### ✅ **Flexible Data Loading**
- **CSV Files**: Standard comma-separated format
- **Pandas DataFrames**: Direct DataFrame input
- **Auto-detection**: Automatic column name detection
- **Unit Conversion**: Automatic kHz to Hz conversion

### ✅ **Advanced Interpolation**
- **Linear Interpolation**: Smooth frequency response curves
- **Extrapolation**: Handling of out-of-range frequencies
- **Validation**: Data validation and error checking

### ✅ **Seamless Integration**
- **PAMpalSettings**: Direct integration with settings workflow
- **Signal Processing**: Automatic application to spectral analysis
- **Processing Pipeline**: Integration with function application system

## Quick Start Guide

### Basic Usage

```python
import pampal
from pampal.calibration import load_calibration_file

# Load calibration from CSV file
calibration_function = load_calibration_file(
    file_path="hydrophone_calibration.csv",
    unit_type=3,  # uPa/FullScale
    name="my_hydrophone"
)

# Create PAMpal settings and add calibration
settings = pampal.PAMpalSettings()
settings.add_calibration_file(
    file_path="hydrophone_calibration.csv",
    module="ClickDetector",
    unit_type=3
)

# Process data with calibration
study = pampal.process_detections(settings)
```

### Advanced Usage

```python
# Load calibration with specific unit type
settings.add_calibration_file(
    file_path="calibration.csv",
    module="ClickDetector",
    unit_type=1,  # dB re V/uPa
    voltage_range=5.0,  # 5V range
    name="custom_cal"
)

# Apply calibration to all detector types
settings.add_calibration_file(
    file_path="calibration.csv",
    module="ClickDetector",
    apply_to_all=True
)

# Manual calibration management
cal_func = settings.get_calibration("ClickDetector")
has_cal = settings.has_calibration("ClickDetector")
calibrations = settings.list_calibrations()
```

## Calibration File Format

### CSV File Structure

The calibration system uses a simple CSV format that matches the R PAMpal specification:

```csv
Frequency,Sensitivity
1000,90.0
5000,85.0
10000,80.0
20000,75.0
50000,70.0
```

### Format Requirements

1. **Two Columns**: Frequency and Sensitivity data
2. **Headers**: Optional but recommended
3. **Frequency Units**: Hz (automatically converts from kHz if detected)
4. **Sensitivity Units**: dB (specific meaning depends on unit_type)
5. **Numeric Data**: All values must be finite numbers

### Column Name Flexibility

The system automatically detects column names:

**Frequency Columns**: `Frequency`, `Freq`, `frequency`, `freq`, `f`, `F`
**Sensitivity Columns**: `Sensitivity`, `Sens`, `sensitivity`, `sens`, `dB`, `db`

### Example Files

See `examples/example_calibration.csv` for a comprehensive example with realistic hydrophone calibration data covering 1 Hz to 200 kHz.

## Unit Types and Scaling

### Unit Type 1: dB re V/uPa
**Description**: Voltage-based calibration requiring voltage range specification
**Parameters**: `voltage_range` (float) - voltage range of recording system
**Use Case**: When calibration is provided in volts per microPascal
**Example**:
```python
settings.add_calibration_file(
    file_path="cal.csv",
    unit_type=1,
    voltage_range=5.0  # 5V range
)
```

### Unit Type 2: uPa/Counts  
**Description**: Count-based calibration requiring bit rate specification
**Parameters**: `bit_rate` (int) - bit depth of recording system
**Use Case**: When calibration is provided in microPascals per count
**Example**:
```python
settings.add_calibration_file(
    file_path="cal.csv", 
    unit_type=2,
    bit_rate=16  # 16-bit recording
)
```

### Unit Type 3: uPa/FullScale
**Description**: Full-scale calibration (most common)
**Parameters**: None required
**Use Case**: Standard hydrophone calibration format
**Example**:
```python
settings.add_calibration_file(
    file_path="cal.csv",
    unit_type=3  # Default, no additional parameters
)
```

## API Reference

### CalibrationFunction Class

```python
class CalibrationFunction:
    def __init__(self, frequencies, sensitivities, unit_type=3, 
                 voltage_range=None, bit_rate=None, name="calibration"):
        """Create a calibration function with frequency interpolation."""
    
    def __call__(self, frequencies):
        """Apply calibration to frequency or array of frequencies."""
    
    def get_frequency_range(self):
        """Get the frequency range covered by this calibration."""
    
    def get_sensitivity_range(self):
        """Get the sensitivity range covered by this calibration."""
```

### CalibrationManager Class

```python
class CalibrationManager:
    def add_calibration(self, detector_type, name, calibration_function):
        """Add a calibration function for a specific detector type."""
    
    def get_calibration(self, detector_type, name=None):
        """Get a calibration function for a specific detector type."""
    
    def remove_calibration(self, detector_type, name=None):
        """Remove a calibration function."""
    
    def list_calibrations(self):
        """List all available calibration functions."""
    
    def has_calibration(self, detector_type, name=None):
        """Check if a calibration function exists."""
```

### PAMpalSettings Calibration Methods

```python
class PAMpalSettings:
    def add_calibration_file(self, file_path, module="ClickDetector", 
                           name=None, unit_type=3, voltage_range=None, 
                           bit_rate=None, apply_to_all=False):
        """Add calibration from a CSV file."""
    
    def add_calibration(self, module, name, function):
        """Add a calibration function to a specific module."""
    
    def remove_calibration(self, module, name=None):
        """Remove calibration function(s) from a module."""
    
    def get_calibration(self, module, name=None):
        """Get a calibration function for a module."""
    
    def list_calibrations(self):
        """List all available calibration functions."""
    
    def has_calibration(self, module, name=None):
        """Check if a calibration function exists."""
```

### Utility Functions

```python
def load_calibration_file(file_path, unit_type=3, voltage_range=None, 
                         bit_rate=None, name=None):
    """Convenience function to load a calibration function from CSV."""

def apply_calibration_to_spectrum(frequencies, spectrum, calibration_function):
    """Apply calibration to a power spectrum."""
```

## Integration with Signal Processing

### Automatic Application

The calibration system automatically integrates with signal processing functions:

```python
# Calibrated spectrogram calculation
Sxx, freqs, times = calculate_spectrogram(
    waveform, sample_rate, 
    calibration_function=cal_func
)

# Calibrated parameter calculation  
params = calculate_click_parameters(
    waveform, sample_rate,
    calibration_function=cal_func
)
```

### Processing Pipeline Integration

Calibration is automatically applied when using the processing pipeline:

```python
# Calibration applied during processing if available
study = pampal.process_detections(settings)
```

The system automatically:
1. Checks for available calibration functions
2. Applies calibration before other processing functions
3. Passes calibration to functions that support it
4. Handles errors gracefully with fallback to uncalibrated processing

## Best Practices

### 1. Calibration File Management
- **Version Control**: Keep calibration files under version control
- **Documentation**: Document calibration file sources and dates
- **Validation**: Validate calibration data before use
- **Backup**: Maintain backup copies of calibration files

### 2. Unit Type Selection
- **uPa/FullScale**: Use for standard hydrophone calibrations (most common)
- **dB re V/uPa**: Use when voltage range information is available
- **uPa/Counts**: Use for specialized count-based calibrations

### 3. Frequency Range Considerations
- **Coverage**: Ensure calibration covers your analysis frequency range
- **Extrapolation**: Be cautious of extrapolation beyond calibration range
- **Resolution**: Higher frequency resolution provides better accuracy

### 4. Integration Workflow
- **Early Application**: Apply calibration early in processing pipeline
- **Consistency**: Use same calibration across related analyses
- **Documentation**: Document calibration application in analysis notes

## Error Handling

The calibration system provides comprehensive error handling:

### File Loading Errors
```python
try:
    settings.add_calibration_file("calibration.csv")
except CalibrationError as e:
    print(f"Calibration error: {e}")
```

### Common Error Types
- **File Not Found**: Calibration file doesn't exist
- **Invalid Format**: CSV format errors or missing columns
- **Invalid Data**: Non-numeric or infinite values
- **Missing Parameters**: Required parameters for unit type not provided
- **Insufficient Data**: Less than 2 calibration points

### Fallback Behavior
- **Processing Continues**: Analysis continues without calibration if errors occur
- **Warning Messages**: Clear warning messages indicate calibration failures
- **Graceful Degradation**: Functions work normally without calibration

## Performance Considerations

### Memory Usage
- **Efficient Storage**: Calibration functions use minimal memory
- **Shared References**: Same calibration can be used across multiple detectors
- **Lazy Loading**: Calibration applied only when needed

### Computational Performance
- **Fast Interpolation**: Linear interpolation for real-time performance
- **Vectorized Operations**: Numpy-based calculations for speed
- **Caching**: Interpolation results cached for repeated use

### Large Dataset Handling
- **Streaming Processing**: Works with large datasets through streaming
- **Batch Application**: Efficient batch processing for multiple files
- **Memory Management**: Automatic memory management for large analyses

## Testing and Validation

### Unit Tests
The calibration system includes comprehensive unit tests:
- **CalibrationFunction**: Creation, interpolation, error handling
- **CalibrationLoader**: File loading, format detection, validation
- **CalibrationManager**: Multiple calibration management
- **PAMpalSettings**: Integration testing
- **Signal Processing**: Calibration application testing

### Running Tests
```bash
cd PAMpal-main/Python
python -m unittest tests.test_calibration
```

### Validation Against R Implementation
The Python implementation is validated against the R PAMpal calibration system:
- **Identical Results**: Same calibration values for identical inputs
- **File Format Compatibility**: Same CSV format support
- **Unit Type Compatibility**: Identical unit type calculations

## Examples and Demos

### Demo Script
Run the calibration demo to see the system in action:
```bash
cd PAMpal-main/Python/examples
python calibration_demo.py
```

The demo demonstrates:
- Loading calibration from CSV files
- Applying calibration to acoustic spectra
- Calibrated parameter calculations
- PAMpalSettings integration
- Visualization of calibration effects

### Example Files
- **example_calibration.csv**: Realistic hydrophone calibration data
- **calibration_demo.py**: Complete demonstration script
- **test_calibration.py**: Comprehensive unit tests

## Troubleshooting

### Common Issues

**Q: Calibration file not loading**
A: Check file path, CSV format, and column names. Ensure data is numeric.

**Q: Strange calibration values**
A: Verify unit type selection and required parameters (voltage_range, bit_rate).

**Q: Calibration not applied**
A: Check that calibration is properly added to PAMpalSettings and detector type matches.

**Q: Processing errors with calibration**
A: Enable verbose output to see calibration application warnings.

### Debug Mode
Enable detailed calibration logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Support
For additional support:
1. Check unit tests for usage examples
2. Review demo script for complete workflow
3. Consult R PAMpal documentation for calibration concepts
4. Check GitHub issues for known problems

## Future Enhancements

### Planned Features
- **Advanced Interpolation**: Spline and GAM interpolation options
- **Multiple Calibrations**: Support for multiple simultaneous calibrations
- **Calibration Validation**: Automated calibration quality checks
- **Database Storage**: Calibration storage in PAMGuard databases
- **Visualization Tools**: Built-in calibration plotting functions

### R Compatibility Roadmap
- **Full Feature Parity**: Complete compatibility with R implementation
- **Extended Functionality**: Python-specific enhancements
- **Cross-Platform Validation**: Automated testing against R results

## Conclusion

The PAMpal calibration system provides a robust, flexible, and scientifically accurate solution for acoustic data calibration. It seamlessly integrates with the PAMpal processing pipeline while maintaining compatibility with existing R workflows. The system's modular design allows for easy extension and customization while ensuring reliable, reproducible results for marine mammal acoustic research.