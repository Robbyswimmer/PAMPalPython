#!/usr/bin/env python3
"""
PAMpal Visualization System Demo

This script demonstrates the comprehensive visualization capabilities of PAMpal Python,
including waveform plots, spectrograms, detection analysis, and advanced visualizations.

The demo creates synthetic acoustic data to showcase all visualization features
without requiring actual PAMGuard data files.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
from datetime import datetime, timedelta

# Add parent directory to path to import pampal modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pampal
from pampal.viz import (
    set_style, plot_waveform, plot_multi_waveform, plot_waveform_envelope,
    plot_spectrogram, plot_spectrogram_with_detections, plot_average_spectrum,
    plot_detection_overview, plot_click_parameters, plot_ici_analysis,
    plot_study_overview, plot_temporal_patterns, plot_spatial_distribution
)


def create_synthetic_click_train():
    """Create a synthetic train of echolocation clicks."""
    sample_rate = 192000
    duration = 10.0  # 10 seconds
    n_samples = int(sample_rate * duration)
    
    # Create time vector
    t = np.linspace(0, duration, n_samples)
    
    # Generate click train with varying ICI
    click_times = []
    current_time = 0.5
    
    while current_time < duration - 0.5:
        click_times.append(current_time)
        # Variable ICI between 0.1 and 0.8 seconds
        ici = 0.1 + 0.7 * np.random.random()
        current_time += ici
    
    # Create full waveform
    waveform = np.zeros(n_samples)
    detections = []
    
    for i, click_time in enumerate(click_times):
        # Create individual click
        click_duration = 0.002  # 2ms
        click_samples = int(sample_rate * click_duration)
        click_center = int(click_time * sample_rate)
        
        # Generate click waveform (damped sinusoid)
        click_t = np.linspace(0, click_duration, click_samples)
        center_freq = 30000 + 20000 * np.random.random()  # 30-50 kHz
        amplitude = 0.5 + 0.5 * np.random.random()
        
        envelope = np.exp(-click_t * 3000)  # Exponential decay
        click_waveform = amplitude * envelope * np.sin(2 * np.pi * center_freq * click_t)
        
        # Add to main waveform
        start_idx = max(0, click_center - click_samples // 2)
        end_idx = min(n_samples, click_center + click_samples // 2)
        actual_samples = end_idx - start_idx
        
        if actual_samples > 0:
            waveform[start_idx:end_idx] += click_waveform[:actual_samples]
            
            # Store detection info
            detections.append({
                'time': click_time,
                'peak_freq': center_freq,
                'amplitude': amplitude,
                'duration': click_duration,
                'detection_type': 'click',
                'UTC': datetime.now() + timedelta(seconds=click_time)
            })
    
    # Add some noise
    noise_level = 0.05
    waveform += noise_level * np.random.randn(n_samples)
    
    return waveform, sample_rate, pd.DataFrame(detections)


def create_synthetic_whistle():
    """Create a synthetic whistle contour."""
    sample_rate = 192000
    duration = 2.0  # 2 seconds
    n_samples = int(sample_rate * duration)
    
    # Create time vector
    t = np.linspace(0, duration, n_samples)
    
    # Create frequency modulated whistle
    f0 = 8000  # Base frequency
    f1 = 15000  # End frequency
    
    # Frequency sweep with some modulation
    freq_inst = f0 + (f1 - f0) * (t / duration) + 1000 * np.sin(2 * np.pi * 3 * t)
    
    # Create whistle waveform
    phase = 2 * np.pi * np.cumsum(freq_inst) / sample_rate
    amplitude = 0.3 * np.exp(-((t - duration/2) / 0.5)**2)  # Gaussian envelope
    waveform = amplitude * np.sin(phase)
    
    # Add noise
    noise_level = 0.05
    waveform += noise_level * np.random.randn(n_samples)
    
    # Create contour data
    contour_times = np.linspace(0, duration, 200)
    contour_freqs = f0 + (f1 - f0) * (contour_times / duration) + 1000 * np.sin(2 * np.pi * 3 * contour_times)
    contour_amps = 0.3 * np.exp(-((contour_times - duration/2) / 0.5)**2)
    
    contour_df = pd.DataFrame({
        'time': contour_times,
        'frequency': contour_freqs,
        'amplitude': 20 * np.log10(contour_amps + 1e-6)  # Convert to dB
    })
    
    detection_data = pd.DataFrame([{
        'time': duration / 2,
        'frequency': np.mean(contour_freqs),
        'detection_type': 'whistle',
        'UTC': datetime.now() + timedelta(seconds=duration/2)
    }])
    
    return waveform, sample_rate, contour_df, detection_data


def demo_waveform_plots():
    """Demonstrate waveform plotting capabilities."""
    print("=== Waveform Plotting Demo ===")
    
    # Create synthetic data
    click_waveform, sample_rate, click_detections = create_synthetic_click_train()
    whistle_waveform, _, whistle_contour, whistle_detections = create_synthetic_whistle()
    
    # Single click for detailed analysis
    single_click_start = int(0.5 * sample_rate)
    single_click_end = int(0.52 * sample_rate)
    single_click = click_waveform[single_click_start:single_click_end]
    
    # Demo 1: Basic waveform plot
    print("1. Basic waveform plot...")
    fig1, ax1 = plot_waveform(single_click, sample_rate, 
                             title="Echolocation Click Waveform",
                             show_envelope=True)
    plt.savefig('examples/waveform_basic.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Demo 2: Multi-waveform comparison
    print("2. Multi-waveform comparison...")
    waveforms = {
        'Click': single_click,
        'Whistle': whistle_waveform[::4]  # Downsample for comparison
    }
    fig2, axes2 = plot_multi_waveform(waveforms, sample_rate,
                                     title="Acoustic Signal Comparison",
                                     stack_vertical=True)
    plt.savefig('examples/waveform_multi.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Demo 3: Waveform envelope analysis
    print("3. Waveform envelope analysis...")
    fig3, ax3 = plot_waveform_envelope(single_click, sample_rate,
                                      method='hilbert',
                                      title="Click Envelope Analysis")
    plt.savefig('examples/waveform_envelope.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("   ✓ Waveform plots saved to examples/")


def demo_spectrogram_plots():
    """Demonstrate spectrogram plotting capabilities."""
    print("\n=== Spectrogram Plotting Demo ===")
    
    # Create synthetic data
    click_waveform, sample_rate, click_detections = create_synthetic_click_train()
    whistle_waveform, _, whistle_contour, whistle_detections = create_synthetic_whistle()
    
    # Demo 1: Basic spectrogram
    print("1. Basic spectrogram...")
    fig1, ax1 = plot_spectrogram(click_waveform[:sample_rate*2], sample_rate,
                                freq_range=(5000, 80000),
                                title="Click Train Spectrogram")
    plt.savefig('examples/spectrogram_basic.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Demo 2: Spectrogram with detections
    print("2. Spectrogram with detection overlays...")
    # Filter detections to first 2 seconds
    early_detections = click_detections[click_detections['time'] <= 2.0].copy()
    
    fig2, ax2 = plot_spectrogram_with_detections(
        click_waveform[:sample_rate*2], early_detections, sample_rate,
        freq_range=(5000, 80000),
        title="Click Train with Detection Markers"
    )
    plt.savefig('examples/spectrogram_detections.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Demo 3: Average spectrum comparison
    print("3. Average spectrum comparison...")
    click_segment = click_waveform[:sample_rate]
    whistle_segment = whistle_waveform
    
    spectra = {
        'Click Train': click_segment,
        'Whistle': whistle_segment
    }
    
    fig3, ax3 = plot_average_spectrum(spectra, sample_rate,
                                     freq_range=(1000, 50000),
                                     title="Average Power Spectra Comparison")
    plt.savefig('examples/spectrum_average.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("   ✓ Spectrogram plots saved to examples/")


def demo_detection_analysis():
    """Demonstrate detection analysis plotting capabilities."""
    print("\n=== Detection Analysis Demo ===")
    
    # Create synthetic data
    click_waveform, sample_rate, click_detections = create_synthetic_click_train()
    
    # Add some additional parameters for analysis
    np.random.seed(42)  # For reproducible results
    n_detections = len(click_detections)
    
    click_detections['centroid_freq'] = click_detections['peak_freq'] * (0.9 + 0.2 * np.random.random(n_detections))
    click_detections['bandwidth'] = 5000 + 10000 * np.random.random(n_detections)
    click_detections['q_factor'] = click_detections['peak_freq'] / click_detections['bandwidth']
    click_detections['snr'] = 10 + 20 * np.random.random(n_detections)
    click_detections['rms_amplitude'] = click_detections['amplitude'] * (0.5 + 0.5 * np.random.random(n_detections))
    
    # Demo 1: Detection overview
    print("1. Detection overview...")
    fig1, axes1 = plot_detection_overview(click_detections, detection_type='click')
    plt.savefig('examples/detection_overview.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Demo 2: Click parameter analysis
    print("2. Click parameter analysis...")
    parameters = ['peak_freq', 'centroid_freq', 'bandwidth', 'q_factor', 'amplitude', 'snr']
    fig2, axes2 = plot_click_parameters(click_detections, parameters=parameters)
    plt.savefig('examples/click_parameters.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Demo 3: ICI analysis
    print("3. Inter-click interval analysis...")
    fig3, axes3 = plot_ici_analysis(click_detections, max_ici=1.0)
    plt.savefig('examples/ici_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("   ✓ Detection analysis plots saved to examples/")


def demo_advanced_plots():
    """Demonstrate advanced plotting capabilities."""
    print("\n=== Advanced Analysis Demo ===")
    
    # Import advanced plotting functions
    from pampal.viz.advanced import plot_cepstrogram, plot_depth_analysis
    
    # Create synthetic data
    click_waveform, sample_rate, click_detections = create_synthetic_click_train()
    
    # Demo 1: Cepstrogram
    print("1. Cepstrogram analysis...")
    single_click_start = int(0.5 * sample_rate)
    single_click_end = int(0.52 * sample_rate)
    single_click = click_waveform[single_click_start:single_click_end]
    
    try:
        fig1, ax1 = plot_cepstrogram(single_click, sample_rate,
                                    quefrency_range=(0, 0.001),
                                    title="Click Cepstrogram")
        plt.savefig('examples/cepstrogram.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("   ✓ Cepstrogram saved")
    except Exception as e:
        print(f"   ⚠ Cepstrogram demo skipped: {e}")
    
    # Demo 2: Synthetic depth analysis
    print("2. Depth analysis...")
    
    # Create synthetic depth data
    n_depths = 50
    times = np.linspace(0, 10, n_depths)
    true_depth = 100 + 20 * np.sin(0.5 * times)  # Sinusoidal depth profile
    noise = 5 * np.random.randn(n_depths)
    measured_depths = true_depth + noise
    
    depth_data = pd.DataFrame({
        'time': times,
        'depth': measured_depths,
        'echo_delay': measured_depths / 750,  # Rough sound speed in water
        'confidence': 0.7 + 0.3 * np.random.random(n_depths)
    })
    
    confidence_intervals = np.column_stack([
        measured_depths - 10,
        measured_depths + 10
    ])
    
    try:
        fig2, axes2 = plot_depth_analysis(depth_data, measured_depths, confidence_intervals)
        plt.savefig('examples/depth_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("   ✓ Depth analysis saved")
    except Exception as e:
        print(f"   ⚠ Depth analysis demo skipped: {e}")
    
    print("   ✓ Advanced plots saved to examples/")


def demo_publication_quality():
    """Demonstrate publication-quality plotting."""
    print("\n=== Publication Quality Demo ===")
    
    # Set publication style
    set_style('publication')
    
    # Create synthetic data
    click_waveform, sample_rate, click_detections = create_synthetic_click_train()
    
    # Create a publication-quality figure
    single_click_start = int(0.5 * sample_rate)
    single_click_end = int(0.52 * sample_rate)
    single_click = click_waveform[single_click_start:single_click_end]
    
    # Multi-panel figure
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    
    # Panel A: Waveform
    time = np.linspace(0, len(single_click)/sample_rate, len(single_click))
    if np.max(np.abs(single_click)) > 0:
        normalized_click = single_click / np.max(np.abs(single_click))
    else:
        normalized_click = single_click
    
    axes[0, 0].plot(time * 1000, normalized_click, 'k-', linewidth=1)
    axes[0, 0].set_xlabel('Time (ms)')
    axes[0, 0].set_ylabel('Normalized Amplitude')
    axes[0, 0].set_title('A) Waveform')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Panel B: Spectrogram
    Sxx, freqs, times = pampal.calculate_spectrogram(single_click, sample_rate, window_size=256)
    im = axes[0, 1].pcolormesh(times * 1000, freqs / 1000, Sxx, shading='auto', cmap='gray')
    axes[0, 1].set_xlabel('Time (ms)')
    axes[0, 1].set_ylabel('Frequency (kHz)')
    axes[0, 1].set_title('B) Spectrogram')
    
    # Panel C: Frequency distribution
    freq_data = click_detections['peak_freq'] / 1000  # Convert to kHz
    axes[1, 0].hist(freq_data, bins=15, color='gray', alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Peak Frequency (kHz)')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('C) Frequency Distribution')
    
    # Panel D: ICI distribution
    times_sec = pd.to_datetime(click_detections['UTC'])
    time_seconds = (times_sec - times_sec.iloc[0]).dt.total_seconds().values
    intervals = np.diff(time_seconds)
    
    axes[1, 1].hist(intervals, bins=15, color='gray', alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel('Inter-Click Interval (s)')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('D) ICI Distribution')
    
    plt.tight_layout()
    plt.savefig('examples/publication_figure.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Reset to default style
    set_style('default')
    
    print("   ✓ Publication-quality figure saved to examples/publication_figure.png")


def demo_study_level_plots():
    """Demonstrate study-level plotting capabilities."""
    print("\n=== Study-level Visualization Demo ===")
    
    # Create synthetic study data
    click_waveform, sample_rate, click_detections = create_synthetic_click_train()
    
    # Add synthetic location data
    np.random.seed(42)
    n_detections = len(click_detections)
    
    # Simulate a study area (e.g., around California coast)
    base_lat, base_lon = 36.0, -121.0
    click_detections['latitude'] = base_lat + 0.5 * np.random.randn(n_detections)
    click_detections['longitude'] = base_lon + 0.5 * np.random.randn(n_detections)
    
    # Create study data structure
    study_data = {
        'detections': click_detections,
        'survey_info': {
            'name': 'Monterey Bay Acoustic Survey',
            'location': 'Monterey Bay, California',
            'start_date': '2024-01-01',
            'end_date': '2024-01-10'
        },
        'effort_data': {
            'total_hours': 240,  # 10 days * 24 hours
            'active_hours': 180,
            'recording_duty_cycle': 0.75
        }
    }
    
    # Demo 1: Study overview
    print("1. Study overview plot...")
    try:
        fig1, axes1 = plot_study_overview(study_data)
        plt.savefig('examples/study_overview.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("   ✓ Study overview saved")
    except Exception as e:
        print(f"   ⚠ Study overview demo skipped: {e}")
    
    # Demo 2: Temporal patterns
    print("2. Temporal patterns analysis...")
    try:
        fig2, axes2 = plot_temporal_patterns(click_detections, time_grouping='hour')
        plt.savefig('examples/temporal_patterns.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("   ✓ Temporal patterns saved")
    except Exception as e:
        print(f"   ⚠ Temporal patterns demo skipped: {e}")
    
    # Demo 3: Spatial distribution
    print("3. Spatial distribution plot...")
    try:
        fig3, ax3 = plot_spatial_distribution(click_detections)
        plt.savefig('examples/spatial_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("   ✓ Spatial distribution saved")
    except Exception as e:
        print(f"   ⚠ Spatial distribution demo skipped: {e}")
    
    print("   ✓ Study-level plots saved to examples/")


def main():
    """Run the complete visualization demo."""
    print("Starting PAMpal Visualization System Demo...\n")
    
    # Create output directory
    os.makedirs('examples', exist_ok=True)
    
    # Run all demos
    demo_waveform_plots()
    demo_spectrogram_plots()
    demo_detection_analysis()
    demo_advanced_plots()
    demo_study_level_plots()
    demo_publication_quality()
    
    print("\n=== Demo Complete ===")
    print("\nThe PAMpal visualization system provides:")
    print("• High-quality waveform and spectrogram plotting")
    print("• Comprehensive detection analysis visualizations")
    print("• Advanced acoustic analysis plots (cepstrograms, depth analysis)")
    print("• Study-level overview and temporal/spatial analysis")
    print("• Publication-ready figure generation")
    print("• Flexible styling and theming options")
    print("• Interactive capabilities (coming soon)")
    print("\nAll demo plots have been saved to the examples/ directory.")
    print("For more information, see the visualization system documentation.")


if __name__ == "__main__":
    main()