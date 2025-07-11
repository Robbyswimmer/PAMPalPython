# Real Data Analysis Examples

This directory contains comprehensive examples demonstrating PAMpal Python's capabilities using real marine mammal acoustic data converted from the original R package.

## Available Examples

### 1. Click Analysis Example (`click_analysis_example.py`)

Comprehensive analysis of marine mammal echolocation clicks demonstrating:

- **Time-domain Analysis**: Waveform visualization and envelope extraction
- **Spectral Analysis**: Spectrograms and power spectral density
- **Parameter Extraction**: Acoustic parameters like peak frequency, bandwidth, Q-factor
- **Multi-channel Analysis**: Cross-correlation and phase relationships
- **Signal Processing**: Advanced acoustic parameter calculations

**Generated Outputs:**
- `click_waveform.png` - Single channel waveform plot
- `click_multichannel.png` - Two-channel comparison
- `click_spectrogram.png` - Time-frequency analysis
- `click_analysis_summary.png` - Comprehensive parameter analysis
- `click_channel_analysis.png` - Inter-channel analysis

**Usage:**
```bash
python click_analysis_example.py
```

### 2. Comprehensive Workflow Example (`comprehensive_workflow.py`)

End-to-end demonstration of the complete PAMpal Python analysis pipeline using all available datasets:

- **Data Loading**: Load and organize all example datasets
- **Signal Analysis**: Click, whistle, and cepstral analysis
- **Study-level Analysis**: Detection patterns and statistics across multiple events
- **Publication Figures**: Create publication-ready multi-panel figures
- **Workflow Integration**: Demonstrate how all components work together

**Generated Outputs:**
- `workflow_click_analysis.png` - Complete click detection analysis (4-panel figure)
- `workflow_whistle_analysis.png` - Whistle contour characterization (4-panel figure)
- `workflow_cepstral_analysis.png` - Harmonic structure analysis (4-panel figure)
- `workflow_study_overview.png` - Study-level detection patterns (6-panel figure)
- `workflow_publication_summary.png` - Publication-ready summary (6-panel figure)

**Usage:**
```bash
python comprehensive_workflow.py
```

## Dataset Information

The examples use 5 converted datasets from the original PAMpal R package:

| Dataset | Type | Description | Duration | Sample Rate |
|---------|------|-------------|----------|-------------|
| `testClick` | Echolocation Click | Two-channel delphinid click | 1.6 ms | 500 kHz |
| `testWhistle` | Whistle Contour | Frequency-modulated whistle | 1.5 s | - |
| `testCeps` | Cepstrum | Harmonic structure analysis | - | - |
| `testGPL` | GPL Detection | Transient acoustic events | - | - |
| `exStudy` | Complete Study | 2 events, 48 detections | - | - |

## Key Features Demonstrated

### Signal Processing
- Time-domain waveform analysis
- Frequency-domain spectral analysis
- Cepstral analysis for harmonic detection
- Cross-correlation between channels
- Power spectral density calculation

### Visualization
- Publication-quality plotting
- Multi-panel figure layouts
- Scientific color schemes
- Proper axis labeling and formatting
- Statistical annotations

### Data Management
- Efficient data loading and caching
- Organized data structures
- Error handling and validation
- Memory-efficient processing

### Analysis Workflows
- Parameter extraction and statistics
- Detection pattern analysis
- Study-level aggregation
- Temporal analysis
- Species-specific analysis

## Requirements

All examples require the full PAMpal Python installation with dependencies:

```bash
# Ensure virtual environment is activated
source ../../../venv/bin/activate

# Install all dependencies
pip install -r ../../../requirements.txt

# Install PAMpal in development mode
pip install -e ../../../
```

## Running the Examples

### Individual Examples

```bash
# Run click analysis
python click_analysis_example.py

# Run comprehensive workflow
python comprehensive_workflow.py
```

### Batch Processing

To run all examples:

```bash
# Run both examples
python click_analysis_example.py && python comprehensive_workflow.py
```

## Output Files

All examples generate high-resolution PNG files suitable for:

- **Scientific Publications**: 300 DPI resolution, publication themes
- **Presentations**: Clear, high-contrast visualizations
- **Reports**: Comprehensive multi-panel layouts
- **Documentation**: Well-annotated analysis outputs

## Integration Testing

These examples also serve as integration tests for the PAMpal Python system:

```bash
# Run integration tests
cd ../../../
python -m pytest tests/test_real_data_integration.py -v
```

## Customization

The examples are designed to be easily customizable:

1. **Modify Analysis Parameters**: Change frequency ranges, window sizes, etc.
2. **Add New Visualizations**: Extend with additional plot types
3. **Custom Data**: Replace with your own acoustic data
4. **Analysis Workflows**: Adapt the workflow for specific research needs

## Further Reading

- [Real Data Examples Guide](../../docs/real_data_examples.md) - Comprehensive documentation
- [Visualization Guide](../../docs/visualization_guide.md) - Complete visualization system
- [API Reference](../../docs/api_reference.md) - Complete API documentation
- [User Guide](../../docs/user_guide.md) - Getting started guide

## Support

For questions or issues with the examples:

1. Check the documentation in `docs/`
2. Run the integration tests to verify installation
3. Review the example code comments for implementation details
4. Submit issues to the project repository