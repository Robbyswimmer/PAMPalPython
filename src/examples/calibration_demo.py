#!/usr/bin/env python3
"""
PAMpal Calibration System Demo

This script demonstrates how to use the PAMpal calibration system for 
acoustic data processing. It shows how to:

1. Load calibration data from CSV files
2. Create and use calibration functions  
3. Apply calibration to acoustic measurements
4. Integrate calibration with PAMpal processing workflow

This demo uses synthetic acoustic data to illustrate the calibration workflow
without requiring actual PAMGuard binary files.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os

# Add parent directory to path to import pampal modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pampal.calibration import (
    CalibrationFunction, load_calibration_file, 
    apply_calibration_to_spectrum
)
from pampal.settings import PAMpalSettings
from pampal.signal_processing import calculate_spectrogram, calculate_click_parameters


def create_synthetic_click():
    """Create a synthetic echolocation click for demonstration."""
    # Parameters for synthetic click
    sample_rate = 192000  # 192 kHz
    duration = 0.002  # 2 ms
    center_freq = 40000  # 40 kHz center frequency
    bandwidth = 20000  # 20 kHz bandwidth
    
    # Generate time vector
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Create chirp-like click (frequency modulated)
    freq_sweep = np.linspace(center_freq - bandwidth/2, 
                           center_freq + bandwidth/2, len(t))
    
    # Exponential envelope (typical of biosonar clicks)
    envelope = np.exp(-t * 2000)
    
    # Generate click waveform
    waveform = envelope * np.sin(2 * np.pi * freq_sweep * t)
    
    # Add some noise
    noise_level = 0.05
    waveform += noise_level * np.random.randn(len(waveform))
    
    return waveform, sample_rate


def demo_basic_calibration():
    """Demonstrate basic calibration functionality."""
    print("=== PAMpal Calibration Demo ===\n")
    
    print("1. Loading calibration from CSV file...")
    
    # Load calibration from example file
    cal_file = os.path.join(os.path.dirname(__file__), 'example_calibration.csv')
    
    try:
        calibration_function = load_calibration_file(
            file_path=cal_file,
            unit_type=3,  # uPa/FullScale
            name="example_hydrophone"
        )
        
        print(f"   ✓ Loaded calibration: {calibration_function.name}")
        freq_min, freq_max = calibration_function.get_frequency_range()
        print(f"   ✓ Frequency range: {freq_min:.0f} - {freq_max:.0f} Hz")
        
    except Exception as e:
        print(f"   ✗ Error loading calibration: {e}")
        # Create a simple calibration for demo purposes
        frequencies = np.logspace(2, 5.3, 50)  # 100 Hz to 200 kHz
        sensitivities = 80 - 10 * np.log10(frequencies / 1000)  # Decreasing with frequency
        
        calibration_function = CalibrationFunction(
            frequencies=frequencies,
            sensitivities=sensitivities,
            unit_type=3,
            name="demo_calibration"
        )
        print(f"   ✓ Created demo calibration function")
    
    return calibration_function


def demo_spectrum_calibration(calibration_function):
    """Demonstrate calibration application to acoustic spectra."""
    print("\n2. Demonstrating spectrum calibration...")
    
    # Create synthetic click
    waveform, sample_rate = create_synthetic_click()
    
    # Calculate uncalibrated spectrogram
    Sxx_uncal, frequencies, times = calculate_spectrogram(
        waveform, sample_rate, window_size=256, overlap=0.8
    )
    
    # Calculate calibrated spectrogram
    Sxx_cal, _, _ = calculate_spectrogram(
        waveform, sample_rate, window_size=256, overlap=0.8,
        calibration_function=calibration_function
    )
    
    print(f"   ✓ Calculated uncalibrated spectrogram: {Sxx_uncal.shape}")
    print(f"   ✓ Calculated calibrated spectrogram: {Sxx_cal.shape}")
    
    # Calculate the calibration correction applied
    calibration_values = calibration_function(frequencies)
    avg_correction = np.mean(calibration_values)
    print(f"   ✓ Average calibration correction: {avg_correction:.1f} dB")
    
    return frequencies, Sxx_uncal, Sxx_cal


def demo_parameter_calibration(calibration_function):
    """Demonstrate calibration application to acoustic parameter calculations."""
    print("\n3. Demonstrating parameter calibration...")
    
    # Create synthetic click
    waveform, sample_rate = create_synthetic_click()
    
    # Calculate uncalibrated parameters
    params_uncal = calculate_click_parameters(
        waveform, sample_rate, freq_range=(10000, 80000)
    )
    
    # Calculate calibrated parameters
    params_cal = calculate_click_parameters(
        waveform, sample_rate, freq_range=(10000, 80000),
        calibration_function=calibration_function
    )
    
    print(f"   Uncalibrated peak frequency: {params_uncal.peak_frequency:.0f} Hz")
    print(f"   Calibrated peak frequency: {params_cal.peak_frequency:.0f} Hz")
    print(f"   Uncalibrated centroid frequency: {params_uncal.centroid_frequency:.0f} Hz")
    print(f"   Calibrated centroid frequency: {params_cal.centroid_frequency:.0f} Hz")
    print(f"   Uncalibrated Q-factor: {params_uncal.q_factor:.1f}")
    print(f"   Calibrated Q-factor: {params_cal.q_factor:.1f}")
    
    return params_uncal, params_cal


def demo_pampal_settings_integration():
    """Demonstrate calibration integration with PAMpalSettings."""
    print("\n4. Demonstrating PAMpalSettings integration...")
    
    # Create PAMpalSettings object
    settings = PAMpalSettings()
    
    # Add calibration file
    cal_file = os.path.join(os.path.dirname(__file__), 'example_calibration.csv')
    
    try:
        settings.add_calibration_file(
            file_path=cal_file,
            module="ClickDetector",
            unit_type=3,
            name="hydrophone_cal"
        )
        
        print(f"   ✓ Added calibration to ClickDetector module")
        
        # Check calibration status
        has_cal = settings.has_calibration("ClickDetector")
        print(f"   ✓ ClickDetector has calibration: {has_cal}")
        
        # List calibrations
        calibrations = settings.list_calibrations()
        print(f"   ✓ Available calibrations: {calibrations}")
        
        # Get calibration function
        cal_func = settings.get_calibration("ClickDetector")
        if cal_func:
            print(f"   ✓ Retrieved calibration function: {cal_func.name}")
        
    except Exception as e:
        print(f"   ✗ Error with PAMpalSettings integration: {e}")


def plot_calibration_curves(calibration_function, frequencies, Sxx_uncal, Sxx_cal):
    """Plot calibration function and example spectra."""
    print("\n5. Creating calibration plots...")
    
    try:
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('PAMpal Calibration System Demo', fontsize=16)
        
        # Plot 1: Calibration function
        cal_freqs = np.logspace(2, 5.3, 1000)  # 100 Hz to 200 kHz
        cal_values = calibration_function(cal_freqs)
        
        axes[0, 0].semilogx(cal_freqs, cal_values)
        axes[0, 0].set_xlabel('Frequency (Hz)')
        axes[0, 0].set_ylabel('Sensitivity (dB)')
        axes[0, 0].set_title('Hydrophone Calibration Function')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Uncalibrated spectrum
        im1 = axes[0, 1].pcolormesh(
            np.arange(Sxx_uncal.shape[1]), frequencies / 1000, Sxx_uncal,
            shading='auto', cmap='viridis'
        )
        axes[0, 1].set_xlabel('Time (samples)')
        axes[0, 1].set_ylabel('Frequency (kHz)')
        axes[0, 1].set_title('Uncalibrated Spectrogram')
        plt.colorbar(im1, ax=axes[0, 1], label='Power (dB)')
        
        # Plot 3: Calibrated spectrum
        im2 = axes[1, 0].pcolormesh(
            np.arange(Sxx_cal.shape[1]), frequencies / 1000, Sxx_cal,
            shading='auto', cmap='viridis'
        )
        axes[1, 0].set_xlabel('Time (samples)')
        axes[1, 0].set_ylabel('Frequency (kHz)')
        axes[1, 0].set_title('Calibrated Spectrogram')
        plt.colorbar(im2, ax=axes[1, 0], label='Power (dB)')
        
        # Plot 4: Calibration difference
        diff = Sxx_cal - Sxx_uncal
        im3 = axes[1, 1].pcolormesh(
            np.arange(diff.shape[1]), frequencies / 1000, diff,
            shading='auto', cmap='RdBu_r'
        )
        axes[1, 1].set_xlabel('Time (samples)')
        axes[1, 1].set_ylabel('Frequency (kHz)')
        axes[1, 1].set_title('Calibration Correction Applied')
        plt.colorbar(im3, ax=axes[1, 1], label='Correction (dB)')
        
        plt.tight_layout()
        
        # Save plot
        plot_file = os.path.join(os.path.dirname(__file__), 'calibration_demo_plots.png')
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"   ✓ Saved calibration plots to: {plot_file}")
        
        # Show plot if running interactively
        plt.show()
        
    except ImportError:
        print("   ⚠ Matplotlib not available - skipping plots")
    except Exception as e:
        print(f"   ✗ Error creating plots: {e}")


def main():
    """Run the complete calibration demo."""
    print("Starting PAMpal Calibration System Demo...\n")
    
    # Demo 1: Basic calibration loading
    calibration_function = demo_basic_calibration()
    
    # Demo 2: Spectrum calibration
    frequencies, Sxx_uncal, Sxx_cal = demo_spectrum_calibration(calibration_function)
    
    # Demo 3: Parameter calibration
    params_uncal, params_cal = demo_parameter_calibration(calibration_function)
    
    # Demo 4: PAMpalSettings integration
    demo_pampal_settings_integration()
    
    # Demo 5: Visualization
    plot_calibration_curves(calibration_function, frequencies, Sxx_uncal, Sxx_cal)
    
    print("\n=== Demo Complete ===")
    print("\nThe PAMpal calibration system provides:")
    print("• Flexible calibration data loading from CSV files")
    print("• Multiple unit type support (dB re V/uPa, uPa/Counts, uPa/FullScale)")
    print("• Frequency-dependent calibration with interpolation")
    print("• Seamless integration with signal processing functions")
    print("• Easy integration with PAMpalSettings workflow")
    print("\nFor more information, see the calibration system documentation.")


if __name__ == "__main__":
    main()