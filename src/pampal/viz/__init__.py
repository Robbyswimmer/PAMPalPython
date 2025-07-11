"""
PAMpal Visualization Module

This module provides comprehensive visualization capabilities for passive acoustic 
monitoring data analysis. It includes tools for plotting waveforms, spectrograms,
detection analyses, and study-level visualizations with both static and interactive
plotting capabilities.

The module is designed to mirror and enhance the R PAMpal visualization functionality
while leveraging modern Python plotting libraries for improved performance and 
interactivity.
"""

from .core import (
    VisualizationBase, PlotManager, ColorSchemes,
    PampalTheme, set_style, reset_style
)

from .waveforms import (
    plot_waveform, plot_multi_waveform, plot_waveform_envelope
)

from .spectrograms import (
    plot_spectrogram, plot_spectrogram_with_detections,
    plot_average_spectrum, plot_concatenated_spectrogram
)

from .detections import (
    plot_detection_overview, plot_click_parameters,
    plot_whistle_contours, plot_ici_analysis
)

from .study import (
    plot_study_overview, plot_temporal_patterns,
    plot_spatial_distribution, plot_species_comparison
)

# Interactive tools (optional import)
try:
    from .interactive import (
        plot_interactive_spectrogram, plot_interactive_waveform,
        plot_detection_browser, create_detection_dashboard,
        launch_detection_explorer, save_interactive_plot
    )
    INTERACTIVE_AVAILABLE = True
except ImportError:
    INTERACTIVE_AVAILABLE = False

# Jupyter tools (optional import)
try:
    from .jupyter import (
        DetectionExplorerWidget, StudyDashboard, LivePlotter,
        create_parameter_selector, display_detection_table
    )
    JUPYTER_AVAILABLE = True
except ImportError:
    JUPYTER_AVAILABLE = False

# Publication and optimization tools
from .publication import (
    PublicationTheme, MultipanelFigure, create_publication_spectrogram,
    create_parameter_comparison_figure, create_detection_summary_figure,
    export_figure_formats, create_journal_template
)

from .optimization import (
    VisualizationCache, MemoryManager, DataDownsampler, PlotExporter,
    cached_computation, clear_cache, get_cache_stats
)

__all__ = [
    # Core infrastructure
    'VisualizationBase', 'PlotManager', 'ColorSchemes', 'PampalTheme',
    'set_style', 'reset_style',
    
    # Waveform plotting
    'plot_waveform', 'plot_multi_waveform', 'plot_waveform_envelope',
    
    # Spectrogram plotting  
    'plot_spectrogram', 'plot_spectrogram_with_detections',
    'plot_average_spectrum', 'plot_concatenated_spectrogram',
    
    # Detection analysis
    'plot_detection_overview', 'plot_click_parameters',
    'plot_whistle_contours', 'plot_ici_analysis',
    
    # Study-level analysis
    'plot_study_overview', 'plot_temporal_patterns',
    'plot_spatial_distribution', 'plot_species_comparison',
    
    # Publication tools
    'PublicationTheme', 'MultipanelFigure', 'create_publication_spectrogram',
    'create_parameter_comparison_figure', 'create_detection_summary_figure',
    'export_figure_formats', 'create_journal_template',
    
    # Optimization tools
    'VisualizationCache', 'MemoryManager', 'DataDownsampler', 'PlotExporter',
    'cached_computation', 'clear_cache', 'get_cache_stats'
]

# Add interactive tools if available
if INTERACTIVE_AVAILABLE:
    __all__.extend([
        'plot_interactive_spectrogram', 'plot_interactive_waveform',
        'plot_detection_browser', 'create_detection_dashboard',
        'launch_detection_explorer', 'save_interactive_plot'
    ])

# Add Jupyter tools if available
if JUPYTER_AVAILABLE:
    __all__.extend([
        'DetectionExplorerWidget', 'StudyDashboard', 'LivePlotter',
        'create_parameter_selector', 'display_detection_table'
    ])