#!/usr/bin/env python3
"""
Click Analysis Example with Real Data

This script demonstrates how to analyze marine mammal click detections using
real example data from the PAMpal package. It covers:

1. Loading click waveform data
2. Visualizing time-domain signals
3. Performing spectral analysis
4. Extracting acoustic parameters
5. Creating publication-quality plots

This example uses the testClick dataset - a two-channel recording of a
synthetic delphinid click at 500kHz sample rate.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add the package to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from pampal.data import load_test_click, get_dataset_info
from pampal.viz import plot_waveform, plot_spectrogram, set_style
from pampal.signal_processing import calculate_click_parameters, calculate_spectrogram

def main():
    """Run the complete click analysis demonstration."""
    
    print("🐋 PAMpal Python - Click Analysis Example")
    print("=" * 50)
    
    # Load the test click data
    print("📊 Loading test click data...")
    click_data = load_test_click()
    
    # Get dataset information
    info = get_dataset_info('testClick')
    print(f"✅ Loaded click data:")
    print(f"   Duration: {info['duration_ms']:.1f} ms")
    print(f"   Sample rate: {info['sr']:,} Hz")
    print(f"   Channels: {info['channels']}")
    print(f"   Description: {info['description']}")
    
    # Extract waveform and parameters
    waveform = np.array(click_data['wave'])
    sample_rate = click_data['sr']
    
    # Set publication style
    set_style('publication')
    
    print("\n📈 Creating visualizations...")
    
    # 1. Plot time-domain waveform
    print("   🔸 Plotting waveform...")
    fig1, ax1 = plot_waveform(
        waveform[:, 0],  # Use first channel
        sample_rate=sample_rate,
        title="Delphinid Click Waveform - Channel 1",
        normalize=True
    )
    
    # Add some analysis annotations
    ax1.text(0.02, 0.95, 
             f'Peak amplitude: {np.max(np.abs(waveform[:, 0])):.3f}\n'
             f'Duration: {len(waveform)/sample_rate*1000:.1f} ms\n'
             f'Sample rate: {sample_rate/1000:.0f} kHz',
             transform=ax1.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('click_waveform.png', dpi=300, bbox_inches='tight')
    print("     💾 Saved: click_waveform.png")
    
    # 2. Compare both channels
    print("   🔸 Plotting multi-channel comparison...")
    fig2, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    
    time_ms = np.linspace(0, len(waveform)/sample_rate*1000, len(waveform))
    
    for i, (ax, label) in enumerate(zip(axes, ['Channel 1', 'Channel 2'])):
        ax.plot(time_ms, waveform[:, i], 'b-', linewidth=1)
        ax.set_ylabel('Amplitude')
        ax.set_title(f'{label} - Delphinid Click')
        ax.grid(True, alpha=0.3)
        
        # Add peak marker
        peak_idx = np.argmax(np.abs(waveform[:, i]))
        peak_time = time_ms[peak_idx]
        peak_amp = waveform[peak_idx, i]
        ax.plot(peak_time, peak_amp, 'ro', markersize=8, label=f'Peak: {peak_amp:.3f}')
        ax.legend()
    
    axes[-1].set_xlabel('Time (ms)')
    plt.suptitle('Two-Channel Delphinid Click Recording')
    plt.tight_layout()
    plt.savefig('click_multichannel.png', dpi=300, bbox_inches='tight')
    print("     💾 Saved: click_multichannel.png")
    
    # 3. Spectral analysis
    print("   🔸 Performing spectral analysis...")
    
    # Calculate spectrogram for first channel
    Sxx_db, frequencies, times = calculate_spectrogram(
        waveform[:, 0], 
        sample_rate,
        window_size=256,
        overlap=0.75
    )
    
    # Plot spectrogram
    fig3, ax3 = plot_spectrogram(
        waveform[:, 0],
        sample_rate=sample_rate,
        title="Delphinid Click Spectrogram",
        freq_range=(5000, 150000),  # Focus on echolocation range
        colormap='viridis'
    )
    
    # Add frequency annotations
    ax3.axhline(y=50000, color='red', linestyle='--', alpha=0.7, label='50 kHz')
    ax3.axhline(y=80000, color='orange', linestyle='--', alpha=0.7, label='80 kHz')
    ax3.axhline(y=120000, color='yellow', linestyle='--', alpha=0.7, label='120 kHz')
    ax3.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('click_spectrogram.png', dpi=300, bbox_inches='tight')
    print("     💾 Saved: click_spectrogram.png")
    
    # 4. Calculate acoustic parameters
    print("   🔸 Extracting acoustic parameters...")
    
    try:
        params = calculate_click_parameters(waveform[:, 0], sample_rate)
        
        print(f"\n📋 Acoustic Parameters:")
        print(f"   Peak frequency: {params.get('peak_freq', 0)/1000:.1f} kHz")
        print(f"   Centroid frequency: {params.get('centroid_freq', 0)/1000:.1f} kHz")
        print(f"   Bandwidth (-3dB): {params.get('bandwidth', 0)/1000:.1f} kHz")
        print(f"   Q-factor: {params.get('q_factor', 0):.1f}")
        print(f"   Duration: {params.get('duration', 0)*1000:.3f} ms")
        print(f"   Peak amplitude: {params.get('peak_amplitude', 0):.3f}")
        print(f"   RMS amplitude: {params.get('rms_amplitude', 0):.3f}")
        
        # Create parameter summary plot
        fig4, ((ax4a, ax4b), (ax4c, ax4d)) = plt.subplots(2, 2, figsize=(12, 8))
        
        # Power spectrum
        freqs = np.fft.fftfreq(len(waveform[:, 0]), 1/sample_rate)[:len(waveform)//2]
        fft_vals = np.abs(np.fft.fft(waveform[:, 0]))[:len(waveform)//2]
        power_db = 20 * np.log10(fft_vals + 1e-10)
        
        ax4a.plot(freqs/1000, power_db, 'b-', linewidth=1)
        ax4a.axvline(params.get('peak_freq', 0)/1000, color='red', linestyle='--', 
                     label=f"Peak: {params.get('peak_freq', 0)/1000:.0f} kHz")
        ax4a.axvline(params.get('centroid_freq', 0)/1000, color='orange', linestyle='--',
                     label=f"Centroid: {params.get('centroid_freq', 0)/1000:.0f} kHz")
        ax4a.set_xlabel('Frequency (kHz)')
        ax4a.set_ylabel('Power (dB)')
        ax4a.set_title('Power Spectrum')
        ax4a.set_xlim(0, 200)
        ax4a.legend()
        ax4a.grid(True, alpha=0.3)
        
        # Envelope
        from scipy.signal import hilbert
        analytic_signal = hilbert(waveform[:, 0])
        envelope = np.abs(analytic_signal)
        
        ax4b.plot(time_ms, waveform[:, 0], 'b-', alpha=0.7, label='Waveform')
        ax4b.plot(time_ms, envelope, 'r-', linewidth=2, label='Envelope')
        ax4b.plot(time_ms, -envelope, 'r-', linewidth=2)
        ax4b.set_xlabel('Time (ms)')
        ax4b.set_ylabel('Amplitude')
        ax4b.set_title('Signal Envelope')
        ax4b.legend()
        ax4b.grid(True, alpha=0.3)
        
        # Parameter bar chart
        param_names = ['Peak Freq\n(kHz)', 'Centroid\n(kHz)', 'Bandwidth\n(kHz)', 'Q-factor', 'Duration\n(ms)']
        param_values = [
            params.get('peak_freq', 0)/1000,
            params.get('centroid_freq', 0)/1000,
            params.get('bandwidth', 0)/1000,
            params.get('q_factor', 0),
            params.get('duration', 0)*1000
        ]
        
        bars = ax4c.bar(param_names, param_values, color=['red', 'orange', 'green', 'blue', 'purple'])
        ax4c.set_ylabel('Value')
        ax4c.set_title('Acoustic Parameters')
        ax4c.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, param_values):
            height = bar.get_height()
            ax4c.text(bar.get_x() + bar.get_width()/2., height,
                     f'{value:.1f}', ha='center', va='bottom')
        
        # Time-frequency characteristics
        ax4d.imshow(Sxx_db, aspect='auto', origin='lower', extent=[
            times[0]*1000, times[-1]*1000, 
            frequencies[0]/1000, frequencies[-1]/1000
        ], cmap='viridis')
        ax4d.set_xlabel('Time (ms)')
        ax4d.set_ylabel('Frequency (kHz)')
        ax4d.set_title('Time-Frequency Analysis')
        ax4d.set_ylim(0, 200)
        
        plt.suptitle('Delphinid Click - Comprehensive Analysis', fontsize=14, y=0.95)
        plt.tight_layout()
        plt.savefig('click_analysis_summary.png', dpi=300, bbox_inches='tight')
        print("     💾 Saved: click_analysis_summary.png")
        
    except Exception as e:
        print(f"⚠️  Parameter calculation failed: {e}")
        print("   This is expected if signal_processing module needs refinement")
    
    # 5. Channel comparison analysis
    print("   🔸 Analyzing channel differences...")
    
    fig5, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Cross-correlation
    correlation = np.correlate(waveform[:, 0], waveform[:, 1], mode='full')
    lags = np.arange(-len(waveform)+1, len(waveform))
    lag_times = lags / sample_rate * 1e6  # Convert to microseconds
    
    axes[0, 0].plot(lag_times, correlation, 'b-', linewidth=1)
    axes[0, 0].set_xlabel('Lag (μs)')
    axes[0, 0].set_ylabel('Cross-correlation')
    axes[0, 0].set_title('Channel Cross-correlation')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xlim(-50, 50)  # Focus on small lags
    
    # Amplitude comparison
    axes[0, 1].scatter(waveform[:, 0], waveform[:, 1], alpha=0.6, s=1)
    axes[0, 1].set_xlabel('Channel 1 Amplitude')
    axes[0, 1].set_ylabel('Channel 2 Amplitude')
    axes[0, 1].set_title('Channel Amplitude Correlation')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Compute correlation coefficient
    corr_coeff = np.corrcoef(waveform[:, 0], waveform[:, 1])[0, 1]
    axes[0, 1].text(0.05, 0.95, f'r = {corr_coeff:.3f}', 
                    transform=axes[0, 1].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Power spectral density comparison
    from scipy.signal import welch
    f1, psd1 = welch(waveform[:, 0], sample_rate, nperseg=256)
    f2, psd2 = welch(waveform[:, 1], sample_rate, nperseg=256)
    
    axes[1, 0].loglog(f1/1000, psd1, 'b-', label='Channel 1', alpha=0.8)
    axes[1, 0].loglog(f2/1000, psd2, 'r-', label='Channel 2', alpha=0.8)
    axes[1, 0].set_xlabel('Frequency (kHz)')
    axes[1, 0].set_ylabel('PSD')
    axes[1, 0].set_title('Power Spectral Density')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xlim(1, 250)
    
    # Phase difference
    fft1 = np.fft.fft(waveform[:, 0])
    fft2 = np.fft.fft(waveform[:, 1])
    phase_diff = np.angle(fft2) - np.angle(fft1)
    freqs_fft = np.fft.fftfreq(len(waveform), 1/sample_rate)
    
    # Only plot positive frequencies
    pos_freqs = freqs_fft[:len(freqs_fft)//2]
    pos_phase_diff = phase_diff[:len(phase_diff)//2]
    
    axes[1, 1].plot(pos_freqs/1000, pos_phase_diff, 'g-', linewidth=1, alpha=0.7)
    axes[1, 1].set_xlabel('Frequency (kHz)')
    axes[1, 1].set_ylabel('Phase Difference (rad)')
    axes[1, 1].set_title('Inter-channel Phase Difference')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xlim(0, 200)
    axes[1, 1].set_ylim(-np.pi, np.pi)
    
    plt.suptitle('Two-Channel Analysis Comparison', fontsize=14)
    plt.tight_layout()
    plt.savefig('click_channel_analysis.png', dpi=300, bbox_inches='tight')
    print("     💾 Saved: click_channel_analysis.png")
    
    print(f"\n🎯 Analysis Summary:")
    print(f"   Created 5 analysis plots demonstrating:")
    print(f"   • Time-domain waveform visualization")
    print(f"   • Multi-channel comparison")
    print(f"   • Spectral analysis and spectrogram")
    print(f"   • Acoustic parameter extraction")
    print(f"   • Inter-channel analysis")
    print(f"\n✨ Click analysis complete! Check the generated PNG files.")
    
    # Show plots if running interactively
    if __name__ == '__main__':
        try:
            plt.show()
        except:
            pass  # Don't fail if no display available

if __name__ == '__main__':
    main()