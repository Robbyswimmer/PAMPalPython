# PAMpal Python: Marine Mammal Bioacoustics Analysis

PAMpal Python is a comprehensive library for loading, processing, and analyzing passive acoustic monitoring (PAM) data from marine mammal research. Originally developed in R, this Python implementation provides powerful tools for bioacoustic analysis, making advanced marine mammal research accessible through a modern, scientific Python ecosystem.

## 🌊 Overview

PAMpal enables researchers to:

- **Load PAMGuard Data**: Import detections from PAMGuard binary files (.pgdf) and SQLite databases
- **Acoustic Analysis**: Extract detailed acoustic parameters from clicks, whistles, and other marine mammal vocalizations
- **Signal Processing**: Generate spectrograms, perform cepstral analysis, and calculate inter-click intervals (ICI)
- **Visualization**: Create publication-quality plots including waveforms, spectrograms, and multi-panel figures
- **Study Management**: Organize data into events and studies for large-scale analysis
- **Real Data Examples**: Work with real marine mammal datasets from delphinid clicks to whale whistles

### Key Features

🔬 **Comprehensive Signal Processing**
- Acoustic parameter extraction (frequency, bandwidth, duration, amplitude, SNR)
- Spectrogram generation with configurable parameters
- Cepstral analysis for echolocation click structure
- Inter-click interval analysis for species identification

📊 **Advanced Visualization**
- Waveform and spectrogram plotting with customizable styling
- Multi-channel acoustic data visualization
- Interactive plots for Jupyter notebooks
- Publication-ready figures with professional themes

🗄️ **Data Management**
- PAMGuard binary file parsing (PGDF format)
- SQLite database integration with automatic schema discovery
- Event and study-level data organization
- Calibration system for hydrophone corrections

🐋 **Marine Mammal Specific**
- Optimized for echolocation clicks, whistles, and burst-pulse sounds
- Species identification through acoustic parameters
- Behavioral analysis through temporal patterns
- Integration with marine mammal research workflows

## 📦 Installation

### Requirements

- Python 3.7 or higher
- NumPy ≥ 1.20.0
- Pandas ≥ 1.3.0
- SciPy ≥ 1.7.0
- Matplotlib ≥ 3.4.0

### Quick Install

```bash
# Clone the repository
git clone https://github.com/robbyswimmer/PAMpalPython.git
cd PAMpalPython

# Create virtual environment (recommended)
python -m venv pampal-env
source pampal-env/bin/activate  # On Windows: pampal-env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install PAMpal in development mode
pip install -e .
```

### Alternative Installation Methods

**From PyPI** (when available):
```bash
pip install pampal
```

**Using Conda** (when available):
```bash
conda install -c conda-forge pampal
```

### Verify Installation

```python
import pampal
print(f"PAMpal version: {pampal.__version__}")

# Test with example data
from pampal.data import load_test_click
click_data = load_test_click()
print("✅ PAMpal installed successfully!")
```

## 🚀 Quick Start

### 5-Minute Tutorial

Let's analyze a real delphinid click detection:

```python
import pampal
from pampal.data import get_click_waveform
from pampal.viz import plot_waveform, plot_spectrogram
from pampal.signal_processing import calculate_click_parameters
import matplotlib.pyplot as plt

# Load real click data
waveform, sample_rate = get_click_waveform()
print(f"Loaded click: {len(waveform)} samples at {sample_rate/1000} kHz")

# Extract acoustic parameters
params = calculate_click_parameters(waveform[:, 0], sample_rate)
print(f"Peak frequency: {params.peak_frequency:.1f} Hz")
print(f"Bandwidth: {params.bandwidth:.1f} Hz")
print(f"Duration: {params.duration*1000:.2f} ms")

# Create visualizations
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Plot waveform
plot_waveform(waveform[:, 0], sample_rate=sample_rate, ax=ax1)
ax1.set_title("Delphinid Echolocation Click - Waveform")

# Plot spectrogram
plot_spectrogram(waveform[:, 0], sample_rate=sample_rate, ax=ax2)
ax2.set_title("Delphinid Echolocation Click - Spectrogram")

plt.tight_layout()
plt.show()
```

### Working with Study Data

```python
from pampal.data import load_example_study

# Load complete study with multiple events
study = load_example_study()
print(f"Study contains {len(study.events)} acoustic events")

# Analyze first event
event = list(study.events.values())[0]
print(f"Event {event.id} has {len(event.detectors)} detector types")

# Access detection data
if 'ClickDetector' in event.detectors:
    clicks = event.detectors['ClickDetector']
    print(f"Found {len(clicks)} click detections")
```

## 🏗️ Core Components

### PAMpalSettings

Central configuration class for all processing operations:

```python
from pampal import PAMpalSettings

# Create settings object
pps = PAMpalSettings()

# Add data sources
pps.add_database("/path/to/pamguard.sqlite3")
pps.add_binaries("/path/to/binary/files/")

# Add calibration
pps.add_calibration("/path/to/calibration.csv", detector="ClickDetector")

# Add processing functions
pps.add_function("ClickDetector", "standardClickCalcs", params={})
```

### AcousticEvent

Container for detections from a single acoustic event:

```python
from pampal import AcousticEvent

# Events contain detector data, localizations, and metadata
event = AcousticEvent(id="Event_001")

# Access detection data by detector type
clicks = event.detectors.get('ClickDetector')
whistles = event.detectors.get('WhistlesMoans')

# Add species classification
event.species['manual'] = 'Delphinus delphis'
```

### AcousticStudy

Top-level container for entire studies:

```python
from pampal import AcousticStudy

# Studies contain multiple events plus metadata
study = AcousticStudy(id="Study_2024")

# Add events
study.events['Event_001'] = event

# Access study-level data
gps_data = study.gps
effort_data = study.effort
```

## 🎯 Signal Processing API

### Acoustic Parameter Extraction

```python
from pampal.signal_processing import calculate_click_parameters, AcousticParameters

# Calculate comprehensive acoustic parameters
params = calculate_click_parameters(waveform, sample_rate, freq_range=(10000, 150000))

# Access individual parameters
print(f"Peak Frequency: {params.peak_frequency} Hz")
print(f"Centroid Frequency: {params.centroid_frequency} Hz") 
print(f"Bandwidth (-3dB): {params.bandwidth} Hz")
print(f"Q-factor: {params.q_factor}")
print(f"Signal Duration: {params.duration} s")
print(f"RMS Amplitude: {params.amplitude} dB")
print(f"SNR: {params.snr} dB")
```

### Spectrogram Analysis

```python
from pampal.signal_processing import calculate_spectrogram

# Generate high-quality spectrogram
frequencies, times, spectrogram = calculate_spectrogram(
    waveform, 
    sample_rate, 
    nperseg=1024,
    overlap=0.8,
    window='hann'
)

# Power in dB relative to full scale
print(f"Frequency range: {frequencies[0]:.1f} - {frequencies[-1]:.1f} Hz")
print(f"Time resolution: {times[1]-times[0]:.4f} s")
```

### Advanced Analysis

```python
from pampal.signal_processing import (
    calculate_cepstrum, 
    calculate_inter_click_intervals,
    extract_whistle_contour
)

# Cepstral analysis for click structure
cepstrum, quefrency = calculate_cepstrum(waveform, sample_rate)

# ICI analysis for behavioral patterns  
ici_stats = calculate_inter_click_intervals(detection_times)
print(f"Mean ICI: {ici_stats['mean_ici']:.3f} s")
print(f"ICI regularity: {ici_stats['cv']:.3f}")

# Whistle contour extraction
contour_df = extract_whistle_contour(waveform, sample_rate)
```

## 📊 Visualization Suite

### Basic Plotting

```python
from pampal.viz import plot_waveform, plot_spectrogram, set_style

# Apply publication theme
set_style('publication')

# Create waveform plot
fig, ax = plot_waveform(waveform, sample_rate=sample_rate, 
                       title="Delphinid Click", 
                       xlabel="Time (ms)", ylabel="Amplitude")

# Create spectrogram plot
fig, ax = plot_spectrogram(waveform, sample_rate=sample_rate,
                          freq_range=(10000, 150000),
                          db_range=(-60, 0))
```

### Multi-Panel Figures

```python
from pampal.viz import MultipanelFigure

# Create publication-quality multi-panel figure
fig = MultipanelFigure(2, 2, figsize=(12, 10))

# Add waveform
fig.add_waveform(waveform, sample_rate, panel=(0, 0), title="A) Waveform")

# Add spectrogram  
fig.add_spectrogram(waveform, sample_rate, panel=(0, 1), title="B) Spectrogram")

# Add custom plot
ax = fig.get_axis(1, 0)
ax.plot(frequencies, power_spectrum)
ax.set_title("C) Power Spectrum")

# Finalize and save
fig.finalize()
fig.save("marine_mammal_analysis.png", dpi=300)
```

### Interactive Visualization

```python
from pampal.viz.interactive import InteractivePlotter

# Create interactive plots for Jupyter
plotter = InteractivePlotter()
plotter.waveform_spectrogram(waveform, sample_rate)
```

## 🗄️ Database Integration

### Loading PAMGuard Data

```python
from pampal.database import PAMGuardDatabase

# Connect to PAMGuard database
db = PAMGuardDatabase("/path/to/database.sqlite3")

# Discover database schema
schema = db.discover_schema()
print(f"Found detector tables: {list(schema['detector_tables'].keys())}")

# Load detection data
detections = db.load_detections(
    detector_type="Click_Detector_OfflineClicks",
    start_time="2024-01-01 00:00:00",
    end_time="2024-01-02 00:00:00"
)

# Group detections into events
events = db.group_detections_into_events(detections)
```

### Binary File Processing

```python
from pampal.binary_data import get_binary_data

# Load binary data for specific detections
binary_data = get_binary_data(
    binary_folders=["/path/to/binaries/"],
    detection_uids=["12345", "12346", "12347"],
    detector_type="ClickDetector"
)

# Access waveform data
for uid, data in binary_data.items():
    waveform = data['waveform']
    sample_rate = data['sample_rate']
    # Process waveform...
```

## 🐋 Real Data Examples

PAMpal includes comprehensive real datasets converted from marine mammal research:

### Available Datasets

- **testClick**: Two-channel delphinid click (500kHz, 1.6ms duration)
- **testWhistle**: Whistle frequency contour (1-3.1 kHz, 1.5s duration)  
- **testCeps**: Cepstral analysis of echolocation clicks with harmonic structure
- **testGPL**: GPL detector output with transient acoustic events
- **exStudy**: Complete study with 48 detections across 2 acoustic events

### Quick Data Access

```python
from pampal.data import (
    load_all_test_data, get_click_waveform, get_whistle_contour,
    get_study_detections, list_available_datasets
)

# List all available datasets
datasets = list_available_datasets()
print(f"Available datasets: {datasets}")

# Load specific data types
click_waveform, sr = get_click_waveform()
whistle_freq, whistle_time = get_whistle_contour()
study_detections = get_study_detections()

# Load all test data
all_data = load_all_test_data()
```

### Complete Analysis Workflows

Run comprehensive analysis examples:

```bash
# Navigate to examples directory
cd examples/real_data_analysis

# Run complete workflow demonstration
python comprehensive_workflow.py

# Run detailed click analysis
python click_analysis_example.py
```

### Expected Outputs

The examples generate:
- **Signal Analysis**: Waveforms, spectrograms, and parameter extraction
- **Multi-channel Processing**: Cross-correlation and phase analysis
- **Behavioral Analysis**: ICI patterns and detection sequences
- **Publication Figures**: High-quality visualizations ready for papers

## 🔬 Advanced Usage

### Custom Analysis Functions

```python
from pampal import PAMpalSettings

def custom_click_analysis(detection_data, binary_data, calibration=None):
    """Custom function for specialized click analysis."""
    # Your analysis code here
    results = {}
    return results

# Add custom function to settings
pps = PAMpalSettings()
pps.add_function("ClickDetector", "customAnalysis", custom_click_analysis)
```

### Calibration System

```python
from pampal.calibration import CalibrationFunction, load_calibration_file

# Load calibration from file
cal_func = load_calibration_file("/path/to/calibration.csv")

# Apply calibration during processing
calibrated_spectrum = cal_func.apply(frequencies, power_spectrum)

# Create custom calibration function
def my_calibration(frequency):
    # Custom calibration logic
    return sensitivity_correction

cal_func = CalibrationFunction(my_calibration, freq_range=(10000, 150000))
```

### Batch Processing

```python
from pampal.processing import process_detections
import glob

# Process multiple studies
database_files = glob.glob("/data/studies/*/database.sqlite3")

for db_file in database_files:
    pps = PAMpalSettings()
    pps.add_database(db_file)
    pps.add_binaries(db_file.replace("database.sqlite3", "binaries/"))
    
    # Process study
    study = process_detections(pps)
    
    # Save results
    study.save(f"processed_{study.id}.pkl")
```

## 🧪 Development & Testing

### Running Tests

```bash
# Run all tests
cd tests
python run_tests.py

# Run specific test modules
python -m unittest test_signal_processing.py
python -m unittest test_visualization.py
```

### Development Setup

```bash
# Clone repository
git clone https://github.com/TaikiSan21/PAMpal.git
cd PAMpal/Python

# Install in development mode
pip install -e .

# Install development dependencies
pip install pytest pytest-cov black flake8
```

### Test Coverage

PAMpal includes comprehensive test coverage:
- **90+ unit tests** covering all core functionality
- **Integration tests** with real data examples
- **Signal processing validation** with synthetic signals
- **Visualization tests** for plot generation
- **Database tests** with mock PAMGuard data

## 🗺️ Development Roadmap

### ✅ Completed

- **Core Infrastructure**: Complete package structure and class hierarchy
- **Binary File Processing**: Full PGDF parser supporting all detector types
- **Signal Processing**: Comprehensive acoustic analysis capabilities
- **Database Integration**: Complete PAMGuard database support
- **Visualization System**: Publication-quality plotting with interactive capabilities
- **Real Data Examples**: Complete workflow demonstrations
- **Testing Framework**: 100+ tests with comprehensive coverage

### 🚧 In Progress

- **Advanced Signal Processing**: Enhanced filtering and noise reduction
- **Calibration Improvements**: Extended calibration function support
- **Documentation**: Complete API reference and tutorials

### 📋 Planned

- **PyPI Release**: Package distribution through Python Package Index
- **Conda Integration**: Conda-forge package availability
- **External Tool Integration**: BANTER, Raven, and other analysis tools
- **Cloud Processing**: Support for large-scale cloud-based analysis
- **Machine Learning**: Integration with deep learning models for species classification

## 🙏 Acknowledgments

PAMpal Python is based on the original R package developed by Taiki Sakai and the marine mammal research community. This Python implementation extends the capabilities to the broader scientific Python ecosystem while maintaining compatibility with established bioacoustic analysis workflows.

## Mission Statement

PAMpal Python is built with the belief that cutting-edge marine mammal research should be accessible to all scientists, educators, and conservationists—empowering open, transparent, and reproducible bioacoustic analysis.

---

**Ready to dive into marine mammal bioacoustics?** Explore the [real data examples](src/examples/real_data_analysis/) to see PAMpal in action!
