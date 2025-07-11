"""
PAMpal: Python package for loading and processing passive acoustic data
"""

# Core modules that don't require sqlite3
try:
    from .settings import PAMpalSettings
    from .acoustic_event import AcousticEvent
    from .acoustic_study import AcousticStudy
except ImportError:
    # These modules might not be available in limited environments
    pass

# Processing modules that require sqlite3
try:
    from .processing import process_detections
except ImportError:
    # sqlite3 not available - skip processing imports
    pass

# Signal processing modules
try:
    from .signal_processing import (
        AcousticParameters,
        extract_waveform_data,
        calculate_spectrogram,
        calculate_click_parameters,
        extract_whistle_contour,
        calculate_cepstrum,
        calculate_inter_click_intervals,
        analyze_detection_sequence
    )
except ImportError:
    # Signal processing not available - skip imports
    pass

# Import visualization module (should work independently)
try:
    from . import viz
except ImportError:
    # Visualization dependencies not available
    pass

__version__ = '0.1.0'
