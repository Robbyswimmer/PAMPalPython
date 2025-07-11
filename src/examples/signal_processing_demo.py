#!/usr/bin/env python3
"""
Signal Processing Demo for PAMpal

This script demonstrates the signal processing capabilities of PAMpal,
including waveform analysis, spectrogram generation, and acoustic parameter calculations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pampal.signal_processing import (
    calculate_spectrogram,
    calculate_click_parameters,
    extract_whistle_contour,
    calculate_cepstrum,
    calculate_inter_click_intervals,
    analyze_detection_sequence
)


def create_synthetic_click(sample_rate=192000, duration=0.001, frequency=50000, amplitude=1.0):
    """Create a synthetic echolocation click for demonstration."""
    t = np.linspace(0, duration, int(sample_rate * duration))
    envelope = np.exp(-t * 5000)  # Exponential decay
    waveform = amplitude * envelope * np.sin(2 * np.pi * frequency * t)
    
    # Add some noise
    noise_level = 0.1
    noise = noise_level * np.random.randn(len(waveform))
    waveform += noise
    
    return waveform, t


def create_synthetic_whistle(sample_rate=48000, duration=0.5):
    """Create a synthetic whistle with frequency modulation."""
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Frequency sweep from 5kHz to 15kHz
    freq_sweep = 5000 + 10000 * t / duration
    waveform = np.sin(2 * np.pi * freq_sweep * t)
    
    # Add amplitude modulation
    amplitude_mod = 0.8 + 0.2 * np.sin(2 * np.pi * 2 * t)
    waveform *= amplitude_mod
    
    return waveform, t


def demo_click_analysis():
    """Demonstrate click analysis capabilities."""
    print("=== Click Analysis Demo ===")
    
    # Create synthetic click
    waveform, t = create_synthetic_click()
    sample_rate = 192000
    
    print(f"Analyzing synthetic click:")
    print(f"  Duration: {len(waveform)/sample_rate*1000:.2f} ms")
    print(f"  Sample rate: {sample_rate} Hz")
    
    # Calculate acoustic parameters
    params = calculate_click_parameters(waveform, sample_rate)
    
    print(f"\nAcoustic Parameters:")
    print(f"  Peak frequency: {params.peak_frequency:.0f} Hz")
    print(f"  Centroid frequency: {params.centroid_frequency:.0f} Hz")
    print(f"  Bandwidth (-3dB): {params.bandwidth:.0f} Hz")
    print(f"  Q-factor: {params.q_factor:.2f}")
    print(f"  Peak amplitude: {params.peak_amplitude:.3f}")
    print(f"  RMS amplitude: {params.rms_amplitude:.3f}")
    print(f"  SNR: {params.snr:.1f} dB")
    
    # Calculate spectrogram
    Sxx, frequencies, times = calculate_spectrogram(waveform, sample_rate)
    print(f"\nSpectrogram:")
    print(f"  Frequency bins: {len(frequencies)}")
    print(f"  Time bins: {len(times)}")
    print(f"  Frequency range: {frequencies[0]:.0f} - {frequencies[-1]:.0f} Hz")
    
    # Calculate cepstrum
    cepstrum, quefrency = calculate_cepstrum(waveform, sample_rate)
    print(f"\nCepstrum:")
    print(f"  Quefrency bins: {len(quefrency)}")
    print(f"  Max quefrency: {quefrency[-1]*1000:.2f} ms")


def demo_whistle_analysis():
    """Demonstrate whistle analysis capabilities."""
    print("\n=== Whistle Analysis Demo ===")
    
    # Create synthetic whistle
    waveform, t = create_synthetic_whistle()
    sample_rate = 48000
    
    print(f"Analyzing synthetic whistle:")
    print(f"  Duration: {len(waveform)/sample_rate:.2f} s")
    print(f"  Sample rate: {sample_rate} Hz")
    
    # Extract frequency contour
    contour = extract_whistle_contour(waveform, sample_rate)
    
    print(f"\nWhistle Contour:")
    print(f"  Contour points: {len(contour)}")
    if len(contour) > 0:
        print(f"  Frequency range: {contour['frequency'].min():.0f} - {contour['frequency'].max():.0f} Hz")
        print(f"  Duration: {contour['time'].max():.2f} s")
        
        # Calculate frequency statistics
        freq_mean = contour['frequency'].mean()
        freq_std = contour['frequency'].std()
        print(f"  Mean frequency: {freq_mean:.0f} Hz")
        print(f"  Frequency variation: {freq_std:.0f} Hz")


def demo_ici_analysis():
    """Demonstrate inter-click interval analysis."""
    print("\n=== Inter-Click Interval Analysis Demo ===")
    
    # Create regular click sequence
    regular_clicks = np.array([0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
    print("Regular click sequence (50ms intervals):")
    
    ici_regular = calculate_inter_click_intervals(regular_clicks)
    print(f"  Mean ICI: {ici_regular['mean_ici']*1000:.1f} ms")
    print(f"  ICI std: {ici_regular['std_ici']*1000:.2f} ms")
    print(f"  Coefficient of variation: {ici_regular['ici_cv']:.3f}")
    print(f"  Regular clicks: {ici_regular['regular_clicks']}")
    
    # Create irregular click sequence
    irregular_clicks = np.array([0.0, 0.03, 0.08, 0.15, 0.18, 0.35, 0.42])
    print("\nIrregular click sequence:")
    
    ici_irregular = calculate_inter_click_intervals(irregular_clicks)
    print(f"  Mean ICI: {ici_irregular['mean_ici']*1000:.1f} ms")
    print(f"  ICI std: {ici_irregular['std_ici']*1000:.1f} ms")
    print(f"  Coefficient of variation: {ici_irregular['ici_cv']:.3f}")
    print(f"  Regular clicks: {ici_irregular['regular_clicks']}")


def demo_detection_sequence_analysis():
    """Demonstrate comprehensive detection sequence analysis."""
    print("\n=== Detection Sequence Analysis Demo ===")
    
    # Create mock detection data
    detections = pd.DataFrame({
        'UTC': pd.to_datetime([
            '2022-01-01 12:00:00.000',
            '2022-01-01 12:00:00.050',
            '2022-01-01 12:00:00.100',
            '2022-01-01 12:00:00.150',
            '2022-01-01 12:00:00.200',
            '2022-01-01 12:00:00.250'
        ]),
        'peak_freq': [45000, 47000, 46000, 48000, 46500, 47500],
        'amplitude': [0.8, 0.9, 0.7, 1.0, 0.85, 0.75]
    })
    
    print(f"Analyzing detection sequence with {len(detections)} detections")
    
    # Analyze sequence
    results = analyze_detection_sequence(detections, detector_type='click')
    
    print(f"\nSequence Statistics:")
    print(f"  Number of detections: {results['n_detections']}")
    print(f"  Duration: {results['duration']:.3f} s")
    print(f"  Detection rate: {results['detection_rate']:.1f} detections/s")
    print(f"  Mean ICI: {results['mean_ici']*1000:.1f} ms")
    print(f"  Regular clicks: {results['regular_clicks']}")
    
    if 'freq_stats' in results:
        freq_stats = results['freq_stats']
        print(f"\nFrequency Statistics:")
        print(f"  Mean frequency: {freq_stats['mean_freq']:.0f} Hz")
        print(f"  Frequency range: {freq_stats['min_freq']:.0f} - {freq_stats['max_freq']:.0f} Hz")
        print(f"  Frequency std: {freq_stats['std_freq']:.0f} Hz")


def main():
    """Run all signal processing demos."""
    print("PAMpal Signal Processing Demo")
    print("=" * 40)
    
    # Set random seed for reproducible results
    np.random.seed(42)
    
    try:
        demo_click_analysis()
        demo_whistle_analysis()
        demo_ici_analysis()
        demo_detection_sequence_analysis()
        
        print("\n" + "=" * 40)
        print("Demo completed successfully!")
        print("\nThis demonstrates PAMpal's signal processing capabilities:")
        print("- Acoustic parameter calculations for clicks")
        print("- Spectrogram generation and analysis")
        print("- Whistle contour extraction")
        print("- Cepstrum analysis for harmonic content")
        print("- Inter-click interval analysis")
        print("- Comprehensive detection sequence analysis")
        
    except Exception as e:
        print(f"\nError during demo: {e}")
        print("Make sure PAMpal is properly installed with all dependencies.")


if __name__ == "__main__":
    main()
