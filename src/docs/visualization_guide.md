# PAMpal Python Visualization System Guide

## Overview

The PAMpal Python visualization system provides comprehensive tools for analyzing and visualizing passive acoustic monitoring data. It includes everything from basic waveform plots to interactive dashboards and publication-ready figures.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Core Components](#core-components)
3. [Plotting Functions](#plotting-functions)
4. [Interactive Tools](#interactive-tools)
5. [Publication Quality](#publication-quality)
6. [Performance & Optimization](#performance--optimization)
7. [Jupyter Integration](#jupyter-integration)
8. [Examples](#examples)
9. [API Reference](#api-reference)

## Quick Start

### Installation

The visualization system is included with PAMpal Python. Ensure you have the required dependencies:

```bash
pip install matplotlib seaborn plotly ipywidgets scipy
```

### Basic Usage

```python
import pampal
from pampal.viz import plot_waveform, plot_spectrogram, plot_detection_overview

# Load your data
study = pampal.AcousticStudy('path/to/data')

# Basic waveform plot
fig, ax = plot_waveform(waveform, sample_rate=192000)

# Spectrogram with detections
fig, ax = plot_spectrogram_with_detections(waveform, detections, sample_rate)

# Detection analysis overview
fig, axes = plot_detection_overview(detections)
```

## Core Components

### VisualizationBase

The foundation class that provides:
- Consistent theming across all plots
- Color scheme management
- Figure setup utilities
- Axis formatting helpers

```python
from pampal.viz.core import VisualizationBase

# Create a custom visualization
class MyPlotter(VisualizationBase):
    def __init__(self, theme='default'):
        super().__init__(theme)
    
    def my_plot(self, data):
        fig, ax = self._setup_figure()
        # Your plotting code here
        return fig, ax
```

### Color Schemes

Scientific color palettes and detection-specific colors:

```python
from pampal.viz.core import ColorSchemes

colors = ColorSchemes()

# Get detection type colors
detection_colors = colors.detection_colors()
print(detection_colors['click'])  # '#1f77b4'

# Scientific color palette
viridis_colors = colors.scientific_palette('viridis', n_colors=10)
```

### Themes

Multiple visual themes for different contexts:

```python
from pampal.viz import set_style

# Set global theme
set_style('publication')  # For manuscripts
set_style('presentation')  # For talks
set_style('interactive')  # For exploration
```

## Plotting Functions

### Waveform Visualizations

#### Basic Waveform Plot

```python
from pampal.viz import plot_waveform

fig, ax = plot_waveform(
    waveform, 
    sample_rate=192000,
    time_range=(0, 1),  # seconds
    show_envelope=True,
    title="Echolocation Click"
)
```

#### Multi-Waveform Comparison

```python
from pampal.viz import plot_multi_waveform

waveforms = {
    'Click 1': click1_data,
    'Click 2': click2_data,
    'Whistle': whistle_data
}

fig, axes = plot_multi_waveform(
    waveforms, 
    sample_rate,
    stack_vertical=True,
    normalize=True
)
```

#### Waveform with Detection Markers

```python
from pampal.viz import plot_waveform_with_detections

fig, ax = plot_waveform_with_detections(
    waveform,
    detections,  # DataFrame with 'time' column
    sample_rate,
    detection_window=0.01  # seconds around each detection
)
```

### Spectrogram Visualizations

#### Basic Spectrogram

```python
from pampal.viz import plot_spectrogram

fig, ax = plot_spectrogram(
    waveform,
    sample_rate=192000,
    window_size=512,
    overlap=0.75,
    freq_range=(5000, 80000),
    colormap='viridis'
)
```

#### Spectrogram with Detection Overlays

```python
from pampal.viz import plot_spectrogram_with_detections

fig, ax = plot_spectrogram_with_detections(
    waveform,
    detections,  # DataFrame with 'time', 'frequency' columns
    sample_rate,
    detection_colors={'click': 'red', 'whistle': 'blue'}
)
```

#### Average Spectrum Comparison

```python
from pampal.viz import plot_average_spectrum

spectra = {
    'Species A': waveform_a,
    'Species B': waveform_b,
    'Noise': noise_data
}

fig, ax = plot_average_spectrum(
    spectra,
    sample_rate,
    method='welch',
    freq_range=(1000, 100000)
)
```

### Detection Analysis

#### Detection Overview

```python
from pampal.viz import plot_detection_overview

fig, axes = plot_detection_overview(
    detections,  # DataFrame with detection parameters
    detection_type='click'
)
```

#### Click Parameter Analysis

```python
from pampal.viz import plot_click_parameters

parameters = ['peak_freq', 'duration', 'bandwidth', 'amplitude']

fig, axes = plot_click_parameters(
    click_detections,
    parameters=parameters,
    color_by='species'
)
```

#### Inter-Click Interval Analysis

```python
from pampal.viz import plot_ici_analysis

fig, axes = plot_ici_analysis(
    click_detections,
    max_ici=2.0,  # seconds
    time_window=60.0  # rolling window size
)
```

#### Whistle Contour Analysis

```python
from pampal.viz import plot_whistle_contours

contours = {
    'whistle_1': contour1_df,  # DataFrame with 'time', 'frequency'
    'whistle_2': contour2_df
}

fig, ax = plot_whistle_contours(
    contours,
    color_by='amplitude',
    freq_range=(5000, 25000)
)
```

### Study-Level Visualizations

#### Study Overview Dashboard

```python
from pampal.viz import plot_study_overview

study_data = {
    'detections': all_detections,
    'survey_info': {
        'name': 'Monterey Bay Survey',
        'location': 'California Coast',
        'start_date': '2024-01-01',
        'end_date': '2024-01-10'
    },
    'effort_data': {
        'total_hours': 240,
        'active_hours': 200
    }
}

fig, axes = plot_study_overview(study_data)
```

#### Temporal Pattern Analysis

```python
from pampal.viz import plot_temporal_patterns

fig, axes = plot_temporal_patterns(
    detections,
    time_grouping='hour',  # 'hour', 'day', 'week', 'month'
    detection_types=['click', 'whistle']
)
```

#### Spatial Distribution

```python
from pampal.viz import plot_spatial_distribution

# Requires 'latitude', 'longitude' columns in detections
fig, ax = plot_spatial_distribution(
    detections,
    detection_types=['click'],
    map_background=False  # Set True if cartopy available
)
```

#### Species Comparison

```python
from pampal.viz import plot_species_comparison

detections_by_species = {
    'Dolphin': dolphin_detections,
    'Whale': whale_detections,
    'Porpoise': porpoise_detections
}

fig, axes = plot_species_comparison(
    detections_by_species,
    parameters=['peak_freq', 'duration', 'amplitude']
)
```

### Advanced Analysis

#### Cepstrograms

```python
from pampal.viz.advanced import plot_cepstrogram

fig, ax = plot_cepstrogram(
    click_waveform,
    sample_rate,
    quefrency_range=(0, 0.001),
    title="Click Cepstrogram"
)
```

#### Depth Analysis

```python
from pampal.viz.advanced import plot_depth_analysis

depth_data = pd.DataFrame({
    'time': times,
    'depth': depths,
    'echo_delay': delays,
    'confidence': confidences
})

fig, axes = plot_depth_analysis(
    depth_data,
    depth_estimates,
    confidence_intervals
)
```

#### Bearing Analysis

```python
from pampal.viz.advanced import plot_bearing_analysis

bearing_data = pd.DataFrame({
    'time': times,
    'bearing': bearings,
    'confidence': confidences
})

array_geometry = {
    'H1': (0, 0),
    'H2': (100, 0),
    'H3': (50, 86.6),
    'H4': (50, -86.6)
}

fig, axes = plot_bearing_analysis(bearing_data, array_geometry)
```

## Interactive Tools

### Interactive Spectrograms

```python
from pampal.viz.interactive import plot_interactive_spectrogram

# Returns a Plotly figure
fig = plot_interactive_spectrogram(
    waveform,
    sample_rate,
    detections=detections
)

fig.show()
```

### Detection Browser

```python
from pampal.viz.interactive import plot_detection_browser

fig = plot_detection_browser(
    detections,
    parameters=['peak_freq', 'duration', 'amplitude'],
    color_by='detection_type'
)

fig.show()
```

### Study Dashboard

```python
from pampal.viz.interactive import create_detection_dashboard

fig = create_detection_dashboard(study_data)
fig.show()
```

### Web-based Explorer

```python
from pampal.viz.interactive import launch_detection_explorer

# Launches a Dash web application
launch_detection_explorer(
    detections,
    waveforms=waveform_dict,
    port=8050
)
```

## Publication Quality

### Publication Themes

```python
from pampal.viz.publication import PublicationTheme

# Set up for specific journals
theme = PublicationTheme('nature')  # 'science', 'plos'
theme.setup_theme()

# Get journal-specific colors
colors = theme.get_colors()
```

### Multi-Panel Figures

```python
from pampal.viz.publication import MultipanelFigure

# Create complex layout
layout = {
    'nrows': 2, 'ncols': 3,
    'width_ratios': [2, 1, 1],
    'height_ratios': [1, 1]
}

multi_fig = MultipanelFigure(layout, figsize=(12, 8))

# Add subplots and panel labels
ax1 = multi_fig.add_subplot((0, 0))
multi_fig.add_panel_label(ax1, 'A')

# Save in multiple formats
multi_fig.save_figure('figure1', dpi=300, formats=['png', 'pdf'])
```

### Publication-Ready Spectrograms

```python
from pampal.viz.publication import create_publication_spectrogram

fig, ax = create_publication_spectrogram(
    waveform,
    sample_rate,
    detections=detections,
    theme='nature',
    title="Echolocation clicks in *Tursiops truncatus*"
)
```

### Parameter Comparison Figures

```python
from pampal.viz.publication import create_parameter_comparison_figure

species_data = {
    'Species A': detections_a,
    'Species B': detections_b,
    'Species C': detections_c
}

fig, axes = create_parameter_comparison_figure(
    species_data,
    parameters=['peak_freq', 'duration', 'bandwidth'],
    theme='science'
)
```

### Complete Detection Summary

```python
from pampal.viz.publication import create_detection_summary_figure

fig, axes = create_detection_summary_figure(
    detections,
    waveform_examples=example_waveforms,
    sample_rate=192000,
    theme='nature'
)
```

### Export Utilities

```python
from pampal.viz.publication import export_figure_formats

# Export in multiple formats
export_figure_formats(
    fig,
    'my_figure',
    formats=['png', 'pdf', 'svg'],
    dpi=300
)
```

## Performance & Optimization

### Caching

```python
from pampal.viz.optimization import cached_computation

@cached_computation()
def expensive_analysis(data, param1, param2):
    # Computationally expensive function
    return results

# First call computes and caches
result1 = expensive_analysis(data, 10, 'value')

# Second call retrieves from cache
result2 = expensive_analysis(data, 10, 'value')  # Fast!
```

### Memory Management

```python
from pampal.viz.optimization import MemoryManager

# Check memory usage
memory_info = MemoryManager.get_memory_usage()
print(f"Memory usage: {memory_info['rss_mb']:.1f} MB")

# Process large data in chunks
large_dataset = np.random.random(1000000)
chunks = MemoryManager.chunked_processing(
    large_dataset,
    chunk_size=10000,
    func=my_processing_function
)
```

### Data Downsampling

```python
from pampal.viz.optimization import DataDownsampler

# Downsample for visualization
downsampled_waveform = DataDownsampler.downsample_waveform(
    long_waveform,
    target_samples=10000,
    method='decimate'
)

# Downsample spectrogram
downsampled_spec = DataDownsampler.downsample_spectrogram(
    large_spectrogram,
    target_shape=(500, 1000),
    method='interpolate'
)

# Smart detection sampling
sampled_detections = DataDownsampler.adaptive_detection_sampling(
    many_detections,
    max_detections=1000,
    preserve_extremes=True
)
```

### Batch Export

```python
from pampal.viz.optimization import PlotExporter

def create_plot1():
    return plot_waveform(data1, sample_rate)

def create_plot2():
    return plot_spectrogram(data2, sample_rate)

plot_functions = [create_plot1, create_plot2]

PlotExporter.batch_export_plots(
    plot_functions,
    output_dir='plots',
    prefix='figure',
    formats=['png', 'pdf']
)
```

## Jupyter Integration

### Detection Explorer Widget

```python
from pampal.viz.jupyter import DetectionExplorerWidget

widget = DetectionExplorerWidget(
    detections,
    waveforms=waveform_dict,
    sample_rate=192000
)

widget.display()
```

### Study Dashboard Widget

```python
from pampal.viz.jupyter import StudyDashboard

dashboard = StudyDashboard(study_data)
dashboard.display()
```

### Live Plotting

```python
from pampal.viz.jupyter import LivePlotter

def update_plot():
    # Function that creates/updates a plot
    new_data = get_latest_data()
    fig, ax = plot_waveform(new_data, sample_rate)
    return fig, ax

live_plotter = LivePlotter(update_plot, update_interval=2.0)
live_plotter.display()
```

### Interactive Parameter Selection

```python
from pampal.viz.jupyter import create_parameter_selector

selector = create_parameter_selector(detections)
display(selector)
```

### Data Table Display

```python
from pampal.viz.jupyter import display_detection_table

display_detection_table(
    detections,
    columns=['UTC', 'detection_type', 'peak_freq', 'duration'],
    max_rows=100
)
```

## Examples

### Complete Analysis Workflow

```python
import pampal
from pampal.viz import *

# 1. Load data
study = pampal.AcousticStudy('survey_data.db3')
detections = study.get_detections()

# 2. Basic exploration
set_style('interactive')

# Waveform analysis
click_data = study.get_waveform(detection_id='click_001')
fig1, ax1 = plot_waveform(click_data.waveform, click_data.sample_rate)

# Detection overview
fig2, axes2 = plot_detection_overview(detections)

# 3. Study-level analysis
study_data = {
    'detections': detections,
    'survey_info': study.info,
    'effort_data': study.effort
}

fig3, axes3 = plot_study_overview(study_data)

# 4. Interactive exploration
dashboard = create_detection_dashboard(study_data)
dashboard.show()

# 5. Publication figure
set_style('publication')
fig4, axes4 = create_detection_summary_figure(
    detections,
    waveform_examples=study.get_example_waveforms(),
    theme='nature'
)

# 6. Export
export_figure_formats(fig4, 'figure1', formats=['png', 'pdf'])
```

### Custom Visualization

```python
from pampal.viz.core import VisualizationBase

class CustomPlotter(VisualizationBase):
    def __init__(self, theme='default'):
        super().__init__(theme)
    
    def plot_custom_analysis(self, data, parameter):
        fig, ax = self._setup_figure(figsize=(8, 6))
        
        # Custom plotting logic
        processed_data = self._process_data(data, parameter)
        ax.plot(processed_data)
        
        # Use theme colors
        ax.set_facecolor(self.theme.background_color)
        
        # Format axes
        self._format_time_axis(ax)
        
        return fig, ax
    
    def _process_data(self, data, parameter):
        # Custom data processing
        return data * parameter

# Usage
plotter = CustomPlotter('publication')
fig, ax = plotter.plot_custom_analysis(my_data, 1.5)
```

### Batch Processing

```python
from pathlib import Path
from pampal.viz.optimization import PlotExporter

# Process multiple files
data_dir = Path('acoustic_data')
output_dir = Path('plots')

for data_file in data_dir.glob('*.db3'):
    try:
        # Load data
        study = pampal.AcousticStudy(data_file)
        detections = study.get_detections()
        
        if len(detections) > 0:
            # Create plots
            fig1, _ = plot_detection_overview(detections)
            fig2, _ = plot_temporal_patterns(detections)
            
            # Export
            base_name = output_dir / data_file.stem
            PlotExporter.export_high_res_figure(fig1, f"{base_name}_overview")
            PlotExporter.export_high_res_figure(fig2, f"{base_name}_temporal")
            
            plt.close('all')
            
    except Exception as e:
        print(f"Error processing {data_file}: {e}")
```

## API Reference

### Core Classes

- `VisualizationBase`: Base class for all plotters
- `ColorSchemes`: Color palette management
- `PampalTheme`: Theme management

### Plotting Functions

#### Waveforms
- `plot_waveform()`: Basic waveform plotting
- `plot_multi_waveform()`: Multiple waveform comparison
- `plot_waveform_envelope()`: Envelope analysis
- `plot_waveform_with_detections()`: Waveform with detection markers

#### Spectrograms
- `plot_spectrogram()`: Basic spectrogram
- `plot_spectrogram_with_detections()`: Spectrogram with overlays
- `plot_average_spectrum()`: Average power spectrum
- `plot_concatenated_spectrogram()`: Multiple detection spectrograms

#### Detections
- `plot_detection_overview()`: Comprehensive detection analysis
- `plot_click_parameters()`: Click parameter distributions
- `plot_whistle_contours()`: Whistle frequency contours
- `plot_ici_analysis()`: Inter-click interval analysis

#### Study Level
- `plot_study_overview()`: Study dashboard
- `plot_temporal_patterns()`: Temporal analysis
- `plot_spatial_distribution()`: Geographic distribution
- `plot_species_comparison()`: Multi-species comparison

#### Advanced
- `plot_cepstrogram()`: Cepstral analysis
- `plot_wigner_ville()`: Wigner-Ville distribution
- `plot_depth_analysis()`: Depth estimation analysis
- `plot_bearing_analysis()`: Bearing estimation analysis

### Interactive Tools

- `plot_interactive_spectrogram()`: Interactive spectrogram (Plotly)
- `plot_interactive_waveform()`: Interactive waveform (Plotly)
- `plot_detection_browser()`: Parameter browser (Plotly)
- `create_detection_dashboard()`: Study dashboard (Plotly)
- `launch_detection_explorer()`: Web application (Dash)

### Jupyter Widgets

- `DetectionExplorerWidget`: Interactive detection explorer
- `StudyDashboard`: Study-level dashboard
- `LivePlotter`: Live plotting utility

### Publication Tools

- `PublicationTheme`: Journal-specific themes
- `MultipanelFigure`: Complex figure layouts
- `create_publication_spectrogram()`: Publication spectrograms
- `create_parameter_comparison_figure()`: Parameter comparisons
- `create_detection_summary_figure()`: Complete summaries

### Optimization

- `VisualizationCache`: Computation caching
- `MemoryManager`: Memory management utilities
- `DataDownsampler`: Intelligent data downsampling
- `PlotExporter`: Advanced export utilities

## Best Practices

1. **Memory Management**: Use downsampling for large datasets
2. **Caching**: Enable caching for expensive computations
3. **Themes**: Use appropriate themes for context (interactive vs publication)
4. **Color Schemes**: Use scientific color palettes for accessibility
5. **Export**: Export in multiple formats for different uses
6. **Documentation**: Include descriptive titles and labels
7. **Performance**: Profile memory usage for large analyses

## Troubleshooting

### Common Issues

1. **Out of Memory**: Use data downsampling or chunked processing
2. **Slow Performance**: Enable caching for repeated analyses
3. **Poor Plot Quality**: Use publication themes and high DPI
4. **Missing Dependencies**: Install optional packages (plotly, ipywidgets)

### Getting Help

- Check function docstrings: `help(plot_waveform)`
- Review examples in `examples/visualization_demo.py`
- Run unit tests: `python -m pytest tests/test_visualization.py`
- Report issues on GitHub

## Changelog

### Version 0.1.0
- Initial release with complete visualization system
- Support for all major plot types
- Interactive tools with Plotly integration
- Jupyter notebook widgets
- Publication-quality figure tools
- Performance optimization utilities
- Comprehensive test suite