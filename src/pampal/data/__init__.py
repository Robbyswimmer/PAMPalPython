"""
PAMpal Data Module

This module provides access to example datasets and data loading utilities
for the PAMpal Python package.
"""

from .load_data import (
    load_test_click, load_test_whistle, load_test_cepstrum, 
    load_test_gpl, load_example_study, load_all_test_data,
    list_available_datasets, get_dataset_info, create_sample_analysis_data,
    get_click_waveform, get_whistle_contour, get_cepstrum_data,
    get_gpl_detection_data, get_study_detections, DataLoadError
)

__all__ = [
    'load_test_click', 'load_test_whistle', 'load_test_cepstrum', 
    'load_test_gpl', 'load_example_study', 'load_all_test_data',
    'list_available_datasets', 'get_dataset_info', 'create_sample_analysis_data',
    'get_click_waveform', 'get_whistle_contour', 'get_cepstrum_data',
    'get_gpl_detection_data', 'get_study_detections', 'DataLoadError'
]