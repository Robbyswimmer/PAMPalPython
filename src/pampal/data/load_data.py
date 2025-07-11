"""
Data loading module for PAMpal Python example datasets.

This module provides convenient functions to load example datasets that
demonstrate PAMpal's capabilities, including test detector data and complete
study examples.
"""

import os
import pickle
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union

import numpy as np
import pandas as pd

# Get the package root directory
PACKAGE_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PACKAGE_ROOT / 'data' / 'real_examples'


class DataLoadError(Exception):
    """Exception raised when data loading fails."""
    pass


def _ensure_data_dir_exists():
    """Ensure the data directory exists."""
    if not DATA_DIR.exists():
        raise DataLoadError(f"Data directory not found: {DATA_DIR}")


def _load_pickle_file(filename: str) -> Dict[str, Any]:
    """Load a pickle file from the data directory."""
    _ensure_data_dir_exists()
    
    file_path = DATA_DIR / f"{filename}.pkl"
    if not file_path.exists():
        raise DataLoadError(f"Data file not found: {file_path}")
    
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        raise DataLoadError(f"Failed to load {filename}: {str(e)}")


def load_test_click() -> Dict[str, Any]:
    """
    Load test click data: two-channel recording of a delphinid click.
    
    Returns:
        Dictionary containing:
        - wave: numpy array (800 x 2) with waveform data for two channels
        - sr: sample rate (500,000 Hz)
        - metadata: information about the synthetic click
    
    Example:
        >>> data = load_test_click()
        >>> waveform = data['wave']
        >>> sample_rate = data['sr']
        >>> print(f"Click duration: {len(waveform) / sample_rate * 1000:.1f} ms")
    """
    return _load_pickle_file('testClick')


def load_test_whistle() -> Dict[str, Any]:
    """
    Load test whistle data: frequency contour of a synthetic whistle.
    
    Returns:
        Dictionary containing:
        - freq: frequency contour values in Hz
        - time: time values in seconds
        - metadata: information about the synthetic whistle
    
    Example:
        >>> data = load_test_whistle()
        >>> freq = data['freq']
        >>> time = data['time']
        >>> import matplotlib.pyplot as plt
        >>> plt.plot(time, freq)
        >>> plt.xlabel('Time (s)')
        >>> plt.ylabel('Frequency (Hz)')
    """
    return _load_pickle_file('testWhistle')


def load_test_cepstrum() -> Dict[str, Any]:
    """
    Load test cepstrum data: cepstral analysis of echolocation clicks.
    
    Returns:
        Dictionary containing:
        - cepstrum: cepstrum data matrix
        - quefrency: quefrency values (time-like domain)
        - time: time values for multiple frames
        - sr: sample rate
        - metadata: information about the cepstrum analysis
    
    Example:
        >>> data = load_test_cepstrum()
        >>> cepstrum = data['cepstrum']
        >>> quefrency = data['quefrency']
        >>> time = data['time']
        >>> # Plot cepstrogram
        >>> import matplotlib.pyplot as plt
        >>> plt.imshow(cepstrum, aspect='auto', origin='lower')
    """
    return _load_pickle_file('testCeps')


def load_test_gpl() -> Dict[str, Any]:
    """
    Load test GPL (Generalized Power Law) detector data.
    
    Returns:
        Dictionary containing:
        - freq: frequency values
        - time: time values
        - timeRes: time resolution
        - freqRes: frequency resolution
        - energy: energy matrix
        - points: DataFrame of detected points
        - metadata: information about the GPL detection
    
    Example:
        >>> data = load_test_gpl()
        >>> energy = data['energy']
        >>> freq = data['freq']
        >>> time = data['time']
        >>> points = data['points']
        >>> print(f"Detected {len(points)} GPL events")
    """
    return _load_pickle_file('testGPL')


def load_example_study() -> Dict[str, Any]:
    """
    Load example study data: complete AcousticStudy with multiple events.
    
    Returns:
        Dictionary containing:
        - study: AcousticStudy object
        - events: dictionary of AcousticEvent objects
        - gps_data: GPS tracking data
        - metadata: information about the study
    
    Example:
        >>> data = load_example_study()
        >>> study = data['study']
        >>> events = data['events']
        >>> print(f"Study contains {len(events)} acoustic events")
        >>> for event_id, event in events.items():
        ...     print(f"{event_id}: {len(event.detectors)} detector types")
    """
    return _load_pickle_file('exStudy')


def load_all_test_data() -> Dict[str, Dict[str, Any]]:
    """
    Load all test data files at once.
    
    Returns:
        Dictionary with keys: 'click', 'whistle', 'cepstrum', 'gpl', 'study'
    
    Example:
        >>> all_data = load_all_test_data()
        >>> click_data = all_data['click']
        >>> whistle_data = all_data['whistle']
        >>> study_data = all_data['study']
    """
    return {
        'click': load_test_click(),
        'whistle': load_test_whistle(),
        'cepstrum': load_test_cepstrum(),
        'gpl': load_test_gpl(),
        'study': load_example_study()
    }


def list_available_datasets() -> Dict[str, str]:
    """
    List all available datasets with descriptions.
    
    Returns:
        Dictionary mapping dataset names to descriptions.
    """
    return {
        'testClick': 'Two-channel delphinid click recording (800 samples at 500kHz)',
        'testWhistle': 'Synthetic whistle frequency contour (1-3.1 kHz over 1.5s)',
        'testCeps': 'Cepstral analysis of echolocation clicks with harmonic structure',
        'testGPL': 'GPL detector output with transient acoustic events',
        'exStudy': 'Complete AcousticStudy with 2 events and mixed detection types'
    }


def get_dataset_info(dataset_name: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific dataset.
    
    Args:
        dataset_name: Name of the dataset ('testClick', 'testWhistle', etc.)
    
    Returns:
        Dictionary with dataset metadata and statistics.
    
    Example:
        >>> info = get_dataset_info('testClick')
        >>> print(f"Sample rate: {info['sr']} Hz")
        >>> print(f"Duration: {info['metadata']['duration_ms']} ms")
    """
    if dataset_name not in list_available_datasets():
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(list_available_datasets().keys())}")
    
    data = _load_pickle_file(dataset_name)
    
    # Extract key statistics based on dataset type
    info = {
        'source_file': data.get('source_file', dataset_name),
        'description': data.get('description', 'No description available'),
        'data_type': data.get('data_type', 'unknown'),
        'creation_timestamp': data.get('creation_timestamp', 'unknown'),
        'metadata': data.get('metadata', {})
    }
    
    # Add dataset-specific statistics
    if dataset_name == 'testClick':
        if 'wave' in data:
            wave = np.array(data['wave'])
            info.update({
                'sr': data.get('sr', 'unknown'),
                'shape': wave.shape,
                'duration_ms': (wave.shape[0] / data.get('sr', 1)) * 1000 if data.get('sr') else 'unknown',
                'channels': wave.shape[1] if len(wave.shape) > 1 else 1,
                'amplitude_range': [float(wave.min()), float(wave.max())]
            })
    
    elif dataset_name == 'testWhistle':
        if 'freq' in data and 'time' in data:
            freq = np.array(data['freq'])
            time = np.array(data['time'])
            info.update({
                'points': len(freq),
                'duration_s': float(time.max() - time.min()),
                'freq_range_hz': [float(freq.min()), float(freq.max())],
                'time_range_s': [float(time.min()), float(time.max())]
            })
    
    elif dataset_name == 'testCeps':
        if 'cepstrum' in data:
            cepstrum = np.array(data['cepstrum'])
            info.update({
                'shape': cepstrum.shape,
                'frames': cepstrum.shape[1] if len(cepstrum.shape) > 1 else 1,
                'quefrency_samples': cepstrum.shape[0] if len(cepstrum.shape) > 0 else 0,
                'sr': data.get('sr', 'unknown')
            })
    
    elif dataset_name == 'testGPL':
        if 'energy' in data and 'points' in data:
            energy = np.array(data['energy'])
            points = data['points']
            info.update({
                'energy_shape': energy.shape,
                'detected_events': len(points) if hasattr(points, '__len__') else 0,
                'freq_range_hz': [data.get('freq', [0, 0])[0], data.get('freq', [0, 0])[-1]] if 'freq' in data else 'unknown',
                'time_range_s': [data.get('time', [0, 0])[0], data.get('time', [0, 0])[-1]] if 'time' in data else 'unknown'
            })
    
    elif dataset_name == 'exStudy':
        if 'events' in data:
            events = data['events']
            total_detections = 0
            detector_types = set()
            
            for event in events.values():
                if hasattr(event, 'detectors'):
                    for det_type, det_data in event.detectors.items():
                        detector_types.add(det_type)
                        if hasattr(det_data, '__len__'):
                            total_detections += len(det_data)
            
            info.update({
                'num_events': len(events),
                'total_detections': total_detections,
                'detector_types': list(detector_types),
                'has_gps': 'gps_data' in data and len(data['gps_data']) > 0
            })
    
    return info


def create_sample_analysis_data() -> Dict[str, Any]:
    """
    Create a combined dataset suitable for analysis demonstrations.
    
    This function loads and combines multiple test datasets into a structure
    that's convenient for demonstrating complete analysis workflows.
    
    Returns:
        Dictionary with combined analysis-ready data.
    
    Example:
        >>> analysis_data = create_sample_analysis_data()
        >>> waveforms = analysis_data['waveforms']
        >>> detections = analysis_data['detections']
        >>> # Perform complete workflow analysis...
    """
    # Load all individual datasets
    click_data = load_test_click()
    whistle_data = load_test_whistle()
    ceps_data = load_test_cepstrum()
    gpl_data = load_test_gpl()
    study_data = load_example_study()
    
    # Create combined analysis structure
    analysis_data = {
        'waveforms': {
            'click_example': {
                'data': click_data['wave'],
                'sr': click_data['sr'],
                'type': 'click'
            }
        },
        'contours': {
            'whistle_example': {
                'freq': whistle_data['freq'],
                'time': whistle_data['time'],
                'type': 'whistle'
            }
        },
        'spectral_analysis': {
            'cepstrum_example': {
                'data': ceps_data['cepstrum'],
                'quefrency': ceps_data['quefrency'],
                'time': ceps_data['time'],
                'sr': ceps_data['sr'],
                'type': 'cepstrum'
            },
            'gpl_example': {
                'energy': gpl_data['energy'],
                'freq': gpl_data['freq'],
                'time': gpl_data['time'],
                'points': gpl_data['points'],
                'type': 'gpl'
            }
        },
        'study_data': study_data,
        'metadata': {
            'created_by': 'PAMpal Python data loader',
            'description': 'Combined analysis dataset for workflow demonstrations',
            'datasets_included': ['testClick', 'testWhistle', 'testCeps', 'testGPL', 'exStudy']
        }
    }
    
    return analysis_data


# Convenience functions for quick access
def get_click_waveform() -> tuple:
    """Get click waveform and sample rate quickly."""
    data = load_test_click()
    return np.array(data['wave']), data['sr']


def get_whistle_contour() -> tuple:
    """Get whistle frequency contour and time vectors quickly."""
    data = load_test_whistle()
    return np.array(data['freq']), np.array(data['time'])


def get_cepstrum_data() -> tuple:
    """Get cepstrum data, quefrency, and time vectors quickly."""
    data = load_test_cepstrum()
    return np.array(data['cepstrum']), np.array(data['quefrency']), np.array(data['time'])


def get_gpl_detection_data() -> tuple:
    """Get GPL energy matrix, frequency/time axes, and detection points quickly."""
    data = load_test_gpl()
    return (np.array(data['energy']), np.array(data['freq']), 
            np.array(data['time']), data['points'])


def get_study_detections() -> pd.DataFrame:
    """Get all detections from the example study as a single DataFrame."""
    data = load_example_study()
    
    all_detections = []
    events = data['events']
    
    for event_id, event in events.items():
        # Handle both direct dict access and object attribute access
        if isinstance(event, dict) and 'detectors' in event:
            detectors = event['detectors']
        elif hasattr(event, 'detectors'):
            detectors = event.detectors
        else:
            continue
        
        for det_type, det_data in detectors.items():
            if isinstance(det_data, pd.DataFrame):
                df = det_data.copy()
                df['event_id'] = event_id
                df['detector_type'] = det_type
                
                # Add standard columns for analysis
                if 'peak_freq' not in df.columns:
                    # Map detector-specific frequency columns to standard names
                    if 'PeakHz_10dB' in df.columns:
                        df['peak_freq'] = df['PeakHz_10dB']
                    elif 'freqMean' in df.columns:
                        df['peak_freq'] = df['freqMean']
                    else:
                        df['peak_freq'] = 0
                
                if 'amplitude' not in df.columns:
                    # Map detector-specific amplitude columns  
                    if 'peak' in df.columns:
                        df['amplitude'] = df['peak']
                    else:
                        df['amplitude'] = 1.0
                
                all_detections.append(df)
    
    if all_detections:
        return pd.concat(all_detections, ignore_index=True)
    else:
        return pd.DataFrame()


# Module-level convenience for quick imports
__all__ = [
    'load_test_click', 'load_test_whistle', 'load_test_cepstrum', 
    'load_test_gpl', 'load_example_study', 'load_all_test_data',
    'list_available_datasets', 'get_dataset_info', 'create_sample_analysis_data',
    'get_click_waveform', 'get_whistle_contour', 'get_cepstrum_data',
    'get_gpl_detection_data', 'get_study_detections', 'DataLoadError'
]