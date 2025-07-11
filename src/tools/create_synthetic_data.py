#!/usr/bin/env python3
"""
Synthetic Data Generator for PAMpal Python

This script creates synthetic data that matches the structure of the original
R PAMpal example datasets, based on the documentation specifications.

This provides us with clean, well-structured example data for testing and
demonstrations without needing to parse complex R S4 objects.
"""

import os
import sys
import pickle
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directory to path for importing PAMpal modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pampal.acoustic_event import AcousticEvent
from pampal.acoustic_study import AcousticStudy
from pampal.settings import PAMpalSettings


def create_test_click():
    """
    Create testClick data: A two-channel recording of a delphinid click
    
    Format (from R documentation):
    A list with two items:
    - wave: a matrix with two columns of 800 samples, each column is a separate recording channel
    - sr: the sample rate of the recording (500kHz)
    """
    # Create realistic click waveform (exponentially decaying sinusoid)
    sr = 500000  # 500 kHz sample rate
    n_samples = 800
    
    # Time vector
    t = np.linspace(0, n_samples/sr, n_samples)
    
    # Create a realistic delphinid click (multiple frequency components)
    # Delphinid clicks typically have energy from 10-150 kHz
    freq1 = 50000  # 50 kHz primary component
    freq2 = 80000  # 80 kHz secondary component
    freq3 = 120000 # 120 kHz tertiary component
    
    # Channel 1: Primary click
    decay = np.exp(-t * 50000)  # Exponential decay
    channel1 = (0.8 * np.sin(2 * np.pi * freq1 * t) + 
                0.4 * np.sin(2 * np.pi * freq2 * t) + 
                0.2 * np.sin(2 * np.pi * freq3 * t)) * decay
    
    # Channel 2: Slightly different (simulating hydrophone separation)
    # Add small delay and amplitude difference
    delay_samples = 2  # Small delay between channels
    channel2 = np.zeros_like(channel1)
    channel2[delay_samples:] = 0.7 * channel1[:-delay_samples]
    
    # Add realistic noise
    noise_level = 0.1
    channel1 += noise_level * np.random.randn(n_samples)
    channel2 += noise_level * np.random.randn(n_samples)
    
    # Create wave matrix (800 x 2)
    wave = np.column_stack([channel1, channel2])
    
    return {
        'wave': wave,
        'sr': sr,
        'source': 'Synthetic delphinid click based on R PAMpal testClick specification',
        'description': 'Two-channel recording of synthetic delphinid click, 800 samples at 500kHz',
        'metadata': {
            'channels': 2,
            'samples': n_samples,
            'duration_ms': (n_samples / sr) * 1000,
            'peak_frequencies': [freq1, freq2, freq3],
            'simulated_features': ['exponential_decay', 'multi_frequency', 'channel_delay']
        }
    }


def create_test_whistle():
    """
    Create testWhistle data: A fake whistle contour
    
    Format (from R documentation):
    A list with two items:
    - freq: a vector of the frequency contour values in hertz
    - time: a vector of the time values of the contour in seconds
    
    Description: ranging from 1kHz to 3.1kHz
    """
    # Create time vector for whistle contour (typical whistle duration 0.5-2 seconds)
    duration = 1.5  # 1.5 seconds
    n_points = 150  # High resolution contour
    time = np.linspace(0, duration, n_points)
    
    # Create realistic whistle contour (frequency modulated)
    # Start at 1kHz, rise to 3.1kHz with some modulation
    base_freq = 1000  # 1 kHz start
    max_freq = 3100   # 3.1 kHz end
    
    # Primary frequency sweep with sinusoidal modulation
    freq_sweep = base_freq + (max_freq - base_freq) * (time / duration)
    
    # Add frequency modulation (typical of dolphin whistles)
    modulation_depth = 200  # 200 Hz modulation depth
    modulation_rate = 8     # 8 Hz modulation rate
    freq_modulation = modulation_depth * np.sin(2 * np.pi * modulation_rate * time)
    
    # Add some random jitter (realistic contour extraction noise)
    jitter = 50 * np.random.randn(n_points)
    
    freq = freq_sweep + freq_modulation + jitter
    
    # Ensure frequency stays within realistic bounds
    freq = np.clip(freq, 800, 3300)
    
    return {
        'freq': freq,
        'time': time,
        'source': 'Synthetic whistle contour based on R PAMpal testWhistle specification',
        'description': 'Fake whistle contour ranging from 1kHz to 3.1kHz over 1.5 seconds',
        'metadata': {
            'duration_s': duration,
            'points': n_points,
            'freq_range_hz': [freq.min(), freq.max()],
            'modulation_features': ['frequency_sweep', 'sinusoidal_modulation', 'realistic_jitter']
        }
    }


def create_test_ceps():
    """
    Create testCeps data: Cepstrum analysis data
    
    Based on cepstrum analysis structure from signal processing module
    """
    # Create synthetic cepstrum data
    sr = 192000  # 192 kHz sample rate
    n_samples = 2048  # FFT size
    
    # Quefrency vector (time-like domain of cepstrum)
    quefrency = np.arange(n_samples) / sr
    
    # Create time vector for multiple cepstrum frames
    n_frames = 50
    time_resolution = 0.01  # 10ms between frames
    time = np.arange(n_frames) * time_resolution
    
    # Create synthetic cepstrum data (typical of echolocation clicks)
    cepstrum_data = np.zeros((n_samples//2, n_frames))  # Half due to symmetry
    
    for i, t in enumerate(time):
        # Simulate cepstrum of click with harmonic structure
        # Peaks at quefrencies corresponding to harmonic intervals
        base_quefrency = 1.0 / 50000  # 50kHz fundamental (typical click)
        
        # Add peaks at harmonic quefrencies
        for harmonic in range(1, 5):
            q_peak = base_quefrency / harmonic
            q_idx = int(q_peak * sr)
            if q_idx < len(cepstrum_data):
                # Gaussian peak
                sigma = 5
                for j in range(max(0, q_idx-15), min(len(cepstrum_data), q_idx+15)):
                    cepstrum_data[j, i] += np.exp(-0.5 * ((j - q_idx) / sigma)**2) / harmonic
        
        # Add noise
        cepstrum_data[:, i] += 0.1 * np.random.randn(len(cepstrum_data))
    
    quefrency_half = quefrency[:len(cepstrum_data)]
    
    return {
        'cepstrum': cepstrum_data,
        'quefrency': quefrency_half,
        'time': time,
        'sr': sr,
        'source': 'Synthetic cepstrum data based on echolocation click analysis',
        'description': 'Cepstrum analysis showing harmonic structure typical of biosonar clicks',
        'metadata': {
            'frames': n_frames,
            'quefrency_samples': len(quefrency_half),
            'time_resolution_s': time_resolution,
            'fundamental_freq_hz': 50000,
            'harmonics_modeled': 4
        }
    }


def create_test_gpl():
    """
    Create testGPL data: Generalized Power Law detector data
    
    Based on typical GPL detector output structure
    """
    # Create synthetic GPL (Generalized Power Law) detector data
    # This is typically used for detecting transient sounds
    
    # Frequency and time axes
    freq = np.logspace(np.log10(1000), np.log10(100000), 128)  # 1kHz to 100kHz, log spaced
    time = np.linspace(0, 2.0, 200)  # 2 seconds, 200 time points
    
    # Resolution
    freq_res = np.diff(freq).mean()
    time_res = np.diff(time).mean()
    
    # Create synthetic energy matrix
    energy = np.zeros((len(freq), len(time)))
    
    # Add some transient events (typical of GPL detections)
    for event_time in [0.3, 0.8, 1.2, 1.7]:
        event_idx = np.argmin(np.abs(time - event_time))
        
        # Create transient energy peak
        for i, f in enumerate(freq):
            # Energy decreases with frequency (typical of many marine mammal sounds)
            base_energy = 10 * np.exp(-f / 20000)
            
            # Temporal envelope (brief transient)
            temporal_width = 5  # samples
            for j in range(max(0, event_idx-temporal_width), 
                          min(len(time), event_idx+temporal_width)):
                envelope = np.exp(-0.5 * ((j - event_idx) / (temporal_width/3))**2)
                energy[i, j] += base_energy * envelope
    
    # Add background noise
    energy += 0.5 * np.random.rand(*energy.shape)
    
    # Create points of interest (detected events)
    points = []
    for event_time in [0.3, 0.8, 1.2, 1.7]:
        for freq_val in [5000, 15000, 25000]:  # Multiple frequency components
            points.append({
                'time': event_time,
                'frequency': freq_val,
                'energy': 8.5 + np.random.randn() * 0.5
            })
    
    points_df = pd.DataFrame(points)
    
    return {
        'freq': freq,
        'time': time,
        'timeRes': time_res,
        'freqRes': freq_res,
        'energy': energy,
        'points': points_df,
        'source': 'Synthetic GPL detector data',
        'description': 'Generalized Power Law detector output with transient events',
        'metadata': {
            'freq_range_hz': [freq.min(), freq.max()],
            'time_range_s': [time.min(), time.max()],
            'energy_shape': energy.shape,
            'detected_events': len(points),
            'frequency_scaling': 'logarithmic'
        }
    }


def create_example_study():
    """
    Create exStudy data: Example AcousticStudy object
    
    Format: AcousticStudy object containing two AcousticEvent objects
    """
    # Create example settings
    settings = PAMpalSettings()
    
    # Create two example acoustic events
    events = {}
    
    # Event 1: Click detections
    event1_detections = pd.DataFrame({
        'UTC': pd.date_range('2024-01-15 14:30:00', periods=25, freq='500ms'),
        'detection_type': ['click'] * 25,
        'peak_freq': 45000 + 10000 * np.random.randn(25),
        'duration': 0.0001 + 0.00005 * np.random.randn(25),
        'amplitude': 0.7 + 0.2 * np.random.randn(25),
        'bandwidth': 15000 + 3000 * np.random.randn(25),
        'UID': range(1, 26)
    })
    
    event1 = AcousticEvent(
        id='Event_001',
        detectors={'click': event1_detections},
        settings={'sr': 192000}
    )
    events['Event_001'] = event1
    
    # Event 2: Mixed detections (clicks and whistles)
    event2_clicks = pd.DataFrame({
        'UTC': pd.date_range('2024-01-15 15:15:00', periods=15, freq='300ms'),
        'detection_type': ['click'] * 15,
        'peak_freq': 38000 + 8000 * np.random.randn(15),
        'duration': 0.00008 + 0.00003 * np.random.randn(15),
        'amplitude': 0.6 + 0.25 * np.random.randn(15),
        'bandwidth': 12000 + 2500 * np.random.randn(15),
        'UID': range(26, 41)
    })
    
    event2_whistles = pd.DataFrame({
        'UTC': pd.date_range('2024-01-15 15:15:05', periods=8, freq='2s'),
        'detection_type': ['whistle'] * 8,
        'peak_freq': 8000 + 2000 * np.random.randn(8),
        'duration': 0.8 + 0.3 * np.random.randn(8),
        'amplitude': 0.4 + 0.15 * np.random.randn(8),
        'bandwidth': 1500 + 400 * np.random.randn(8),
        'UID': range(41, 49)
    })
    
    event2 = AcousticEvent(
        id='Event_002',
        detectors={'click': event2_clicks, 'whistle': event2_whistles},
        settings={'sr': 192000}
    )
    events['Event_002'] = event2
    
    # Create study object
    study = AcousticStudy(
        id='ExampleStudy_2024',
        pps=settings,
        files={
            'db': '/path/to/example.sqlite3',
            'binaries': '/path/to/binaries/'
        }
    )
    
    # Add events to study
    for event in events.values():
        study.add_event(event)
    
    # Add some GPS data
    gps_data = pd.DataFrame({
        'UTC': pd.date_range('2024-01-15 14:00:00', periods=100, freq='2min'),
        'latitude': 36.7 + 0.01 * np.random.randn(100),
        'longitude': -121.9 + 0.01 * np.random.randn(100),
        'depth': 100 + 20 * np.random.randn(100)
    })
    
    return {
        'study': study,
        'events': events,
        'gps_data': gps_data,
        'source': 'Synthetic AcousticStudy based on R PAMpal exStudy specification',
        'description': 'Example AcousticStudy object containing two AcousticEvent objects',
        'metadata': {
            'num_events': len(events),
            'total_detections': sum(len(ev.detectors.get('click', [])) + 
                                  len(ev.detectors.get('whistle', [])) 
                                  for ev in events.values()),
            'time_span': '2024-01-15 14:30:00 to 15:31:05',
            'detection_types': ['click', 'whistle']
        }
    }


def save_data(data, filename, output_dir):
    """Save data in both pickle and JSON formats."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Add metadata
    data_with_meta = {
        'creation_timestamp': datetime.now().isoformat(),
        'created_by': 'PAMpal Python synthetic data generator',
        'format_version': '1.0',
        **data
    }
    
    # Save as pickle
    pickle_path = output_dir / f"{filename}.pkl"
    with open(pickle_path, 'wb') as f:
        pickle.dump(data_with_meta, f)
    
    # Save JSON-serializable version
    try:
        json_data = convert_to_json_serializable(data_with_meta)
        json_path = output_dir / f"{filename}.json"
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        print(f"✅ Created {filename}.pkl and {filename}.json")
    except Exception as e:
        print(f"✅ Created {filename}.pkl (JSON failed: {e})")


def convert_to_json_serializable(obj):
    """Convert objects to JSON-serializable format."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict('records')
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, (datetime, pd.Timestamp)):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif hasattr(obj, '__dict__'):
        # For custom objects, convert their dict representation
        return convert_to_json_serializable(obj.__dict__)
    else:
        return str(obj)  # Fallback to string representation


def main():
    """Generate all synthetic data files."""
    output_dir = Path('./data/real_examples')
    
    print("PAMpal Python Synthetic Data Generator")
    print("=" * 50)
    print(f"Output directory: {output_dir}")
    print()
    
    # Generate each dataset
    datasets = [
        ('testClick', create_test_click),
        ('testWhistle', create_test_whistle),
        ('testCeps', create_test_ceps),
        ('testGPL', create_test_gpl),
        ('exStudy', create_example_study),
    ]
    
    for name, create_func in datasets:
        print(f"Generating {name}...")
        try:
            data = create_func()
            save_data(data, name, output_dir)
        except Exception as e:
            print(f"❌ Failed to create {name}: {e}")
    
    print()
    print("✨ Synthetic data generation complete!")
    print(f"Files saved to: {output_dir.absolute()}")


if __name__ == '__main__':
    main()