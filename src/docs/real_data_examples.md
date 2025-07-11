# Real Data Examples Guide

PAMpal Python includes a comprehensive set of real example datasets and analysis workflows that demonstrate the complete capabilities of the marine mammal acoustic analysis suite. This guide explains how to use the converted real data examples and run complete analysis workflows.

## Overview

The real data examples were converted from the original PAMpal R package datasets and include:

- **testClick**: Two-channel delphinid click recording (800 samples at 500kHz)
- **testWhistle**: Synthetic whistle frequency contour (1-3.1 kHz over 1.5s)  
- **testCeps**: Cepstral analysis of echolocation clicks with harmonic structure
- **testGPL**: GPL detector output with transient acoustic events
- **exStudy**: Complete AcousticStudy with 2 events and mixed detection types

## Data Loading

### Quick Access Functions

```python
from pampal.data import (
    load_test_click, load_test_whistle, load_test_cepstrum,
    load_test_gpl, load_example_study, load_all_test_data
)

# Load individual datasets
click_data = load_test_click()
whistle_data = load_test_whistle()
ceps_data = load_test_cepstrum()
gpl_data = load_test_gpl()
study_data = load_example_study()

# Load all datasets at once
all_data = load_all_test_data()
```

### Convenience Functions

For quick access to commonly used data components:

```python
from pampal.data import (
    get_click_waveform, get_whistle_contour, get_cepstrum_data,
    get_gpl_detection_data, get_study_detections
)

# Get click waveform and sample rate
waveform, sample_rate = get_click_waveform()

# Get whistle frequency contour
freq, time = get_whistle_contour()

# Get study detections as a DataFrame
detections_df = get_study_detections()
```

### Dataset Information

```python
from pampal.data import list_available_datasets, get_dataset_info

# List all available datasets
datasets = list_available_datasets()
for name, description in datasets.items():
    print(f"{name}: {description}")

# Get detailed information about a specific dataset
info = get_dataset_info('testClick')
print(f"Sample rate: {info['sr']} Hz")
print(f"Duration: {info['duration_ms']:.1f} ms")
print(f"Channels: {info['channels']}")
```

## Example Analysis Workflows

### 1. Click Analysis Example

The click analysis example demonstrates comprehensive analysis of marine mammal echolocation clicks:

```bash
cd examples/real_data_analysis
python click_analysis_example.py
```

This example creates:
- `click_waveform.png` - Time-domain waveform visualization
- `click_multichannel.png` - Two-channel comparison
- `click_spectrogram.png` - Spectral analysis
- `click_analysis_summary.png` - Comprehensive parameter analysis
- `click_channel_analysis.png` - Inter-channel analysis

**Key Features:**
- Time-domain signal analysis
- Spectral analysis and spectrograms
- Acoustic parameter extraction
- Multi-channel comparison
- Cross-correlation analysis
- Power spectral density comparison

### 2. Comprehensive Workflow Example

The comprehensive workflow demonstrates the complete PAMpal analysis pipeline:

```bash
cd examples/real_data_analysis
python comprehensive_workflow.py
```

This example creates:
- `workflow_click_analysis.png` - Complete click detection analysis
- `workflow_whistle_analysis.png` - Whistle contour characterization
- `workflow_cepstral_analysis.png` - Harmonic structure analysis
- `workflow_study_overview.png` - Study-level detection patterns
- `workflow_publication_summary.png` - Publication-ready summary

**Analysis Pipeline:**
1. **Data Loading**: Load all example datasets
2. **Signal Analysis**: Click, whistle, and cepstral analysis
3. **Study-Level Analysis**: Detection patterns and statistics
4. **Publication Figure Generation**: Create publication-ready visualizations

## Dataset Details

### Click Data (testClick)
- **Type**: Two-channel echolocation click
- **Duration**: 1.6 ms (800 samples)
- **Sample Rate**: 500 kHz
- **Content**: Synthetic delphinid click with realistic characteristics
- **Use Cases**: Time-domain analysis, spectral analysis, parameter extraction

```python
click_data = load_test_click()
wave = np.array(click_data['wave'])  # Shape: (800, 2)
sr = click_data['sr']  # 500000 Hz
```

### Whistle Data (testWhistle)
- **Type**: Frequency contour of a whistle
- **Duration**: 1.5 seconds
- **Frequency Range**: 1-3.1 kHz
- **Content**: Synthetic whistle with frequency modulation
- **Use Cases**: Contour analysis, frequency tracking, whistle characterization

```python
whistle_data = load_test_whistle()
freq = np.array(whistle_data['freq'])  # Frequency values in Hz
time = np.array(whistle_data['time'])  # Time values in seconds
```

### Cepstrum Data (testCeps)
- **Type**: Cepstral analysis of echolocation sequences
- **Content**: Cepstrogram showing harmonic structure
- **Use Cases**: Harmonic analysis, fundamental frequency detection
- **Applications**: Species identification, click rate analysis

```python
ceps_data = load_test_cepstrum()
cepstrum = np.array(ceps_data['cepstrum'])  # Cepstrum matrix
quefrency = np.array(ceps_data['quefrency'])  # Quefrency axis
time = np.array(ceps_data['time'])  # Time axis
```

### GPL Data (testGPL)
- **Type**: Generalized Power Law detector output
- **Content**: Energy matrix and detection points
- **Use Cases**: Transient detection, energy analysis
- **Applications**: Automated detection, signal characterization

```python
gpl_data = load_test_gpl()
energy = np.array(gpl_data['energy'])  # Energy matrix
freq = np.array(gpl_data['freq'])  # Frequency axis
time = np.array(gpl_data['time'])  # Time axis
points = gpl_data['points']  # Detection points DataFrame
```

### Study Data (exStudy)
- **Type**: Complete acoustic study with multiple events
- **Content**: 2 acoustic events with mixed detection types
- **Detections**: 48 total detections (clicks and whistles)
- **Use Cases**: Study-level analysis, detection statistics, temporal patterns

```python
study_data = load_example_study()
events = study_data['events']  # Dictionary of AcousticEvent objects
gps_data = study_data['gps_data']  # GPS tracking data

# Get all detections as a single DataFrame
detections = get_study_detections()
print(f"Total detections: {len(detections)}")
```

## Integration with Visualization System

The real data examples are fully integrated with the PAMpal visualization system:

```python
from pampal.viz import plot_waveform, plot_spectrogram, MultipanelFigure
from pampal.data import get_click_waveform

# Load data and create visualizations
waveform, sr = get_click_waveform()

# Create waveform plot
fig1, ax1 = plot_waveform(waveform[:, 0], sample_rate=sr, 
                         title="Delphinid Click")

# Create spectrogram
fig2, ax2 = plot_spectrogram(waveform[:, 0], sample_rate=sr,
                            title="Click Spectrogram")

# Create multi-panel figure
mp_fig = MultipanelFigure((2, 2), figsize=(12, 8))
# ... add panels as needed
```

## Analysis Data Structure

For complex workflows, use the combined analysis data structure:

```python
from pampal.data import create_sample_analysis_data

analysis_data = create_sample_analysis_data()

# Access organized data
waveforms = analysis_data['waveforms']
contours = analysis_data['contours'] 
spectral_analysis = analysis_data['spectral_analysis']
study_data = analysis_data['study_data']
```

This structure organizes all datasets into categories suitable for comprehensive analysis workflows.

## Testing and Validation

The real data examples include comprehensive integration tests:

```bash
# Run integration tests
pytest tests/test_real_data_integration.py -v

# Run tests directly
python tests/test_real_data_integration.py
```

**Test Coverage:**
- Data loading functionality
- Visualization integration
- Signal processing workflows
- Error handling
- Complete analysis pipelines

## Best Practices

1. **Memory Management**: Load only the datasets you need for your analysis
2. **Error Handling**: Use try-except blocks when loading data
3. **Visualization**: Set appropriate figure styles before creating plots
4. **Analysis Workflow**: Follow the comprehensive workflow example as a template
5. **Documentation**: Document your analysis parameters and methods

## Troubleshooting

### Common Issues

**Data Loading Errors:**
```python
from pampal.data import DataLoadError

try:
    data = load_test_click()
except DataLoadError as e:
    print(f"Failed to load data: {e}")
```

**Missing Dependencies:**
- Ensure all required packages are installed: `pip install -r requirements.txt`
- Activate the virtual environment: `source venv/bin/activate`

**Visualization Issues:**
- Use non-interactive backend for testing: `matplotlib.use('Agg')`
- Reset plot styles: `from pampal.viz import reset_style; reset_style()`

### Performance Optimization

For large-scale analysis:

```python
# Load data efficiently
from pampal.data import load_all_test_data
all_data = load_all_test_data()  # Load once, use multiple times

# Use caching for repeated calculations
from pampal.viz.core import VisualizationBase
vis = VisualizationBase(enable_cache=True)
```

## Further Reading

- [Visualization Guide](visualization_guide.md) - Complete visualization system documentation
- [Signal Processing Guide](signal_processing_guide.md) - Signal processing workflows
- [API Reference](api_reference.md) - Complete API documentation
- [User Guide](user_guide.md) - Getting started guide

## Examples Repository

All example scripts and their outputs are available in the `examples/real_data_analysis/` directory:

```
examples/real_data_analysis/
├── click_analysis_example.py
├── comprehensive_workflow.py
├── README.md
└── (generated output files)
```

Each example includes detailed comments and can be run independently to demonstrate specific analysis capabilities.