"""
Utility functions for the PAMpal package.

This module provides general utility functions used throughout the PAMpal package.
"""
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
import datetime
import struct


def format_notes(notes: Dict[str, Any], n: int = 6) -> List[str]:
    """
    Format notes for display, limiting to n items.
    
    Args:
        notes: Dictionary of notes
        n: Maximum number of notes to format
        
    Returns:
        List of formatted note strings
    """
    formatted = []
    for i, (key, value) in enumerate(notes.items()):
        if i >= n:
            remaining = len(notes) - n
            if remaining > 0:
                formatted.append(f"... ({remaining} more not shown)")
            break
        formatted.append(f"{key}: {value}")
    return formatted


def read_pgdf_header(file_path: str) -> Dict[str, Any]:
    """
    Read the header of a Pamguard binary file (.pgdf).
    
    Args:
        file_path: Path to the binary file
        
    Returns:
        Dictionary containing header information
        
    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: If the file is not a valid Pamguard binary file
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    header = {}
    
    try:
        with open(file_path, 'rb') as f:
            # Read magic number
            magic = f.read(4)
            if magic != b'PGDF':
                raise ValueError(f"Not a valid Pamguard binary file: {file_path}")
                
            # Read file format version
            header['version'] = struct.unpack('>i', f.read(4))[0]
            
            # Read file creation date (milliseconds since epoch)
            ms_since_epoch = struct.unpack('>q', f.read(8))[0]
            header['create_time'] = datetime.datetime.fromtimestamp(ms_since_epoch / 1000)
            
            # Read analysis time
            ms_since_epoch = struct.unpack('>q', f.read(8))[0]
            header['analysis_time'] = datetime.datetime.fromtimestamp(ms_since_epoch / 1000)
            
            # Read file type (module identifier)
            type_len = struct.unpack('>h', f.read(2))[0]
            header['file_type'] = f.read(type_len).decode('utf-8')
            
            # Read module type
            module_len = struct.unpack('>h', f.read(2))[0]
            header['module_type'] = f.read(module_len).decode('utf-8')
            
            # Read module name
            module_name_len = struct.unpack('>h', f.read(2))[0]
            header['module_name'] = f.read(module_name_len).decode('utf-8')
            
            # Read stream name
            stream_name_len = struct.unpack('>h', f.read(2))[0]
            header['stream_name'] = f.read(stream_name_len).decode('utf-8')
            
            # Read data blocks start position
            header['data_start'] = f.tell()
            
    except Exception as e:
        raise ValueError(f"Error reading binary file header: {str(e)}")
        
    return header


def match_time_data(events: List[Any], time_data: pd.DataFrame, 
                   time_column: str = 'datetime', 
                   buffer: float = 300) -> Dict[str, pd.DataFrame]:
    """
    Match time-based data to acoustic events based on temporal proximity.
    
    This function matches data from a time series to acoustic events based on the 
    timestamp of each event and a buffer window.
    
    Args:
        events: List of AcousticEvent objects
        time_data: DataFrame containing time series data
        time_column: Name of the column in time_data containing timestamps
        buffer: Time buffer in seconds to look for matches before and after event time
        
    Returns:
        Dictionary mapping event IDs to matched data
    """
    # Ensure time_data[time_column] is datetime type
    if not pd.api.types.is_datetime64_any_dtype(time_data[time_column]):
        try:
            time_data[time_column] = pd.to_datetime(time_data[time_column])
        except:
            raise ValueError(f"Column '{time_column}' cannot be converted to datetime")
    
    # Sort time_data by time_column
    time_data = time_data.sort_values(time_column)
    
    result = {}
    
    for event in events:
        # Get event time (assuming each event has a representative time)
        # In real implementation, this would extract from the event object
        event_time = datetime.datetime.now()  # Placeholder
        
        # Find records within buffer window
        before = event_time - datetime.timedelta(seconds=buffer)
        after = event_time + datetime.timedelta(seconds=buffer)
        
        # Filter time_data
        matched = time_data[(time_data[time_column] >= before) & 
                           (time_data[time_column] <= after)]
        
        if not matched.empty:
            result[event.id] = matched
    
    return result


def calculate_spectrogram(signal: np.ndarray, sample_rate: int, 
                         window_size: int = 512, overlap: float = 0.5, 
                         window_type: str = 'hann') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate a spectrogram from an audio signal.
    
    Args:
        signal: 1D numpy array containing audio samples
        sample_rate: Sample rate in Hz
        window_size: FFT window size
        overlap: Overlap between windows (0-1)
        window_type: Type of window function to use
        
    Returns:
        Tuple of (spectrogram, frequency bins, time bins)
    """
    # Validate inputs
    if overlap < 0 or overlap >= 1:
        raise ValueError("Overlap must be between 0 and 1")
        
    # Calculate hop size
    hop_size = int(window_size * (1 - overlap))
    
    # Create window function
    if window_type == 'hann':
        window = np.hanning(window_size)
    elif window_type == 'hamming':
        window = np.hamming(window_size)
    elif window_type == 'blackman':
        window = np.blackman(window_size)
    else:
        window = np.ones(window_size)  # Rectangle window
    
    # Calculate number of time frames
    n_frames = 1 + (len(signal) - window_size) // hop_size
    
    # Create output spectrogram matrix
    spectrogram = np.zeros((window_size // 2 + 1, n_frames), dtype=complex)
    
    # Calculate FFT for each frame
    for i in range(n_frames):
        start = i * hop_size
        end = start + window_size
        frame = signal[start:end]
        
        # Apply window function
        windowed_frame = frame * window
        
        # Calculate FFT
        fft_result = np.fft.rfft(windowed_frame)
        
        # Store in spectrogram matrix
        spectrogram[:, i] = fft_result
    
    # Calculate magnitude and convert to dB
    spectrogram_magnitude = np.abs(spectrogram)
    spectrogram_db = 20 * np.log10(np.maximum(spectrogram_magnitude, 1e-10))
    
    # Calculate frequency and time bins
    freq_bins = np.fft.rfftfreq(window_size, d=1/sample_rate)
    time_bins = np.arange(n_frames) * hop_size / sample_rate
    
    return spectrogram_db, freq_bins, time_bins
