#!/usr/bin/env python3
"""
Comprehensive PAMpal Workflow Example

This script demonstrates a complete marine mammal acoustic analysis workflow
using all available example datasets. It showcases:

1. Loading multiple data types (clicks, whistles, cepstrum, GPL, study data)
2. Comprehensive visualization suite
3. Signal processing and parameter extraction
4. Study-level analysis and reporting
5. Publication-quality figure generation

This example integrates all PAMpal Python capabilities in a realistic
research workflow.
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Add the package to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from pampal.data import (
    load_all_test_data, create_sample_analysis_data, 
    get_study_detections, list_available_datasets
)
from pampal.viz import (
    plot_waveform, plot_spectrogram, plot_detection_overview,
    plot_study_overview, set_style, reset_style,
    MultipanelFigure, PublicationTheme
)
from pampal.signal_processing import calculate_spectrogram

def print_section_header(title):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f"🔬 {title}")
    print(f"{'='*60}")

def print_step(step_num, description):
    """Print a formatted step description."""
    print(f"\n📋 Step {step_num}: {description}")
    print("-" * 40)

def main():
    """Run the comprehensive PAMpal workflow demonstration."""
    
    print("🐋 PAMpal Python - Comprehensive Workflow Example")
    print("This example demonstrates a complete marine mammal acoustic analysis")
    print("workflow using all available datasets and visualization capabilities.")
    
    # Set publication style for all plots
    set_style('publication')
    
    print_section_header("Data Loading and Overview")
    
    # Load all available datasets
    print_step(1, "Loading all example datasets")
    all_data = load_all_test_data()
    
    print("✅ Loaded datasets:")
    for dataset_name, description in list_available_datasets().items():
        print(f"   • {dataset_name}: {description}")
    
    # Create analysis-ready data structure
    analysis_data = create_sample_analysis_data()
    
    print("\n📊 Dataset overview:")
    print(f"   • Waveforms: {len(analysis_data['waveforms'])} examples")
    print(f"   • Contours: {len(analysis_data['contours'])} examples")
    print(f"   • Spectral analysis: {len(analysis_data['spectral_analysis'])} examples")
    print(f"   • Study data: {len(analysis_data['study_data']['events'])} events")
    
    print_section_header("Signal Analysis Workflow")
    
    # Analyze click data
    print_step(2, "Click detection analysis")
    click_data = all_data['click']
    waveform = np.array(click_data['wave'])
    sample_rate = click_data['sr']
    
    # Create comprehensive click analysis figure
    fig1 = MultipanelFigure((2, 2), figsize=(12, 10))
    
    # Waveform plot
    ax1 = fig1.axes[0, 0]
    time_ms = np.linspace(0, len(waveform)/sample_rate*1000, len(waveform))
    ax1.plot(time_ms, waveform[:, 0], 'b-', linewidth=1)
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Amplitude')
    ax1.set_title('Delphinid Click Waveform')
    ax1.grid(True, alpha=0.3)
    fig1.add_panel_label(ax1, 'A')
    
    # Spectrogram
    ax2 = fig1.axes[0, 1]
    Sxx_db, frequencies, times = calculate_spectrogram(waveform[:, 0], sample_rate)
    frequencies = np.array(frequencies)
    times = np.array(times)
    im = ax2.imshow(Sxx_db, aspect='auto', origin='lower', 
                    extent=[times[0]*1000, times[-1]*1000, 
                           frequencies[0]/1000, frequencies[-1]/1000],
                    cmap='viridis')
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Frequency (kHz)')
    ax2.set_title('Click Spectrogram')
    ax2.set_ylim(0, 200)
    plt.colorbar(im, ax=ax2, label='Power (dB)')
    fig1.add_panel_label(ax2, 'B')
    
    # Power spectrum
    ax3 = fig1.axes[1, 0]
    freqs = np.fft.fftfreq(len(waveform[:, 0]), 1/sample_rate)[:len(waveform)//2]
    fft_vals = np.abs(np.fft.fft(waveform[:, 0]))[:len(waveform)//2]
    power_db = 20 * np.log10(fft_vals + 1e-10)
    ax3.plot(freqs/1000, power_db, 'b-', linewidth=1)
    ax3.set_xlabel('Frequency (kHz)')
    ax3.set_ylabel('Power (dB)')
    ax3.set_title('Power Spectrum')
    ax3.set_xlim(0, 250)
    ax3.grid(True, alpha=0.3)
    fig1.add_panel_label(ax3, 'C')
    
    # Cross-correlation between channels
    ax4 = fig1.axes[1, 1]
    correlation = np.correlate(waveform[:, 0], waveform[:, 1], mode='full')
    lags = np.arange(-len(waveform)+1, len(waveform))
    lag_times = lags / sample_rate * 1e6  # microseconds
    ax4.plot(lag_times, correlation, 'g-', linewidth=1)
    ax4.set_xlabel('Lag (μs)')
    ax4.set_ylabel('Cross-correlation')
    ax4.set_title('Inter-channel Correlation')
    ax4.set_xlim(-50, 50)
    ax4.grid(True, alpha=0.3)
    fig1.add_panel_label(ax4, 'D')
    
    plt.suptitle('Click Detection Analysis', fontsize=16, y=0.95)
    plt.tight_layout()
    plt.savefig('workflow_click_analysis.png', dpi=300, bbox_inches='tight')
    print("   💾 Saved: workflow_click_analysis.png")
    
    # Analyze whistle data
    print_step(3, "Whistle contour analysis")
    whistle_data = all_data['whistle']
    freq_contour = np.array(whistle_data['freq'])
    time_contour = np.array(whistle_data['time'])
    
    fig2, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    # Frequency contour
    ax1.plot(time_contour, freq_contour/1000, 'r-', linewidth=2)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Frequency (kHz)')
    ax1.set_title('Whistle Frequency Contour')
    ax1.grid(True, alpha=0.3)
    
    # Frequency modulation analysis
    freq_derivative = np.gradient(freq_contour, time_contour)
    ax2.plot(time_contour[:-1], freq_derivative[:-1], 'orange', linewidth=1)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Frequency Rate (Hz/s)')
    ax2.set_title('Frequency Modulation Rate')
    ax2.grid(True, alpha=0.3)
    
    # Frequency distribution
    ax3.hist(freq_contour/1000, bins=30, alpha=0.7, color='purple', edgecolor='black')
    ax3.set_xlabel('Frequency (kHz)')
    ax3.set_ylabel('Count')
    ax3.set_title('Frequency Distribution')
    ax3.grid(True, alpha=0.3)
    
    # Contour statistics
    stats_text = f"""Contour Statistics:
Duration: {time_contour[-1] - time_contour[0]:.2f} s
Freq Range: {freq_contour.min()/1000:.1f} - {freq_contour.max()/1000:.1f} kHz
Mean Freq: {freq_contour.mean()/1000:.1f} kHz
Freq Std: {freq_contour.std()/1000:.1f} kHz
Max FM Rate: {np.max(np.abs(freq_derivative)):.0f} Hz/s"""
    
    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, 
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    ax4.set_title('Whistle Parameters')
    
    plt.suptitle('Whistle Contour Analysis', fontsize=14)
    plt.tight_layout()
    plt.savefig('workflow_whistle_analysis.png', dpi=300, bbox_inches='tight')
    print("   💾 Saved: workflow_whistle_analysis.png")
    
    # Analyze cepstrum data
    print_step(4, "Cepstral analysis")
    ceps_data = all_data['cepstrum']
    cepstrum = np.array(ceps_data['cepstrum'])
    quefrency = np.array(ceps_data['quefrency'])
    ceps_time = np.array(ceps_data['time'])
    
    fig3, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    # Cepstrogram
    im1 = ax1.imshow(cepstrum, aspect='auto', origin='lower',
                     extent=[ceps_time[0]*1000, ceps_time[-1]*1000,
                            quefrency[0]*1e6, quefrency[-1]*1e6],
                     cmap='hot')
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Quefrency (μs)')
    ax1.set_title('Cepstrogram')
    ax1.set_ylim(0, 100)  # Focus on low quefrencies
    plt.colorbar(im1, ax=ax1, label='Cepstral Magnitude')
    
    # Average cepstrum
    avg_cepstrum = np.mean(cepstrum, axis=1)
    ax2.plot(quefrency*1e6, avg_cepstrum, 'b-', linewidth=2)
    ax2.set_xlabel('Quefrency (μs)')
    ax2.set_ylabel('Average Cepstral Magnitude')
    ax2.set_title('Average Cepstrum')
    ax2.set_xlim(0, 100)
    ax2.grid(True, alpha=0.3)
    
    # Find peaks (potential harmonic intervals)
    from scipy.signal import find_peaks
    peaks, properties = find_peaks(avg_cepstrum, height=np.max(avg_cepstrum)*0.1)
    peak_quefrencies = quefrency[peaks] * 1e6
    peak_magnitudes = avg_cepstrum[peaks]
    
    ax2.plot(peak_quefrencies, peak_magnitudes, 'ro', markersize=8, label='Peaks')
    ax2.legend()
    
    # Corresponding fundamental frequencies
    fund_freqs = 1 / (quefrency[peaks] + 1e-10)  # Avoid division by zero
    
    ax3.bar(range(len(fund_freqs)), fund_freqs/1000, color='green', alpha=0.7)
    ax3.set_xlabel('Peak Number')
    ax3.set_ylabel('Fundamental Frequency (kHz)')
    ax3.set_title('Detected Fundamental Frequencies')
    ax3.grid(True, alpha=0.3)
    
    # Cepstral statistics
    ceps_stats = f"""Cepstral Analysis:
Frames: {cepstrum.shape[1]}
Duration: {ceps_time[-1] - ceps_time[0]:.0f} ms
Peaks Found: {len(peaks)}
Primary F0: {fund_freqs[0]/1000:.1f} kHz (if valid)
Quefrency Range: 0 - {quefrency[-1]*1e6:.0f} μs"""
    
    ax4.text(0.05, 0.95, ceps_stats, transform=ax4.transAxes,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    ax4.set_title('Cepstral Parameters')
    
    plt.suptitle('Cepstral Analysis - Harmonic Structure Detection', fontsize=14)
    plt.tight_layout()
    plt.savefig('workflow_cepstral_analysis.png', dpi=300, bbox_inches='tight')
    print("   💾 Saved: workflow_cepstral_analysis.png")
    
    print_section_header("Study-Level Analysis")
    
    # Analyze study data
    print_step(5, "Complete study analysis")
    study_data = all_data['study']
    
    # Get all detections as a single DataFrame
    all_detections = get_study_detections()
    
    if not all_detections.empty:
        print(f"   📊 Study contains {len(all_detections)} total detections")
        print(f"   📊 Detection types: {list(all_detections['detector_type'].unique())}")
        print(f"   📊 Events: {list(all_detections['event_id'].unique())}")
        
        # Create enhanced study overview figure
        fig4 = plt.figure(figsize=(16, 12))
        
        # Detection type counts (more informative than timeline for test data)
        ax1 = plt.subplot(3, 3, 1)
        detection_types = all_detections['detector_type']
        type_counts = detection_types.value_counts()
        
        bars = ax1.bar(range(len(type_counts)), type_counts.values, 
                      color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.7)
        ax1.set_xticks(range(len(type_counts)))
        ax1.set_xticklabels([t.replace('_', '\n') for t in type_counts.index], fontsize=8)
        ax1.set_ylabel('Detection Count')
        ax1.set_title('Detections by Type')
        ax1.grid(True, alpha=0.3)
        
        # Add count labels on bars
        for bar, count in zip(bars, type_counts.values):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.2,
                    str(count), ha='center', va='bottom', fontweight='bold')
        
        # Frequency distribution by type (enhanced)
        ax2 = plt.subplot(3, 3, 2)
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        for i, det_type in enumerate(detection_types.unique()):
            mask = detection_types == det_type
            subset = all_detections[mask]
            
            # Use appropriate frequency column for each detector type
            if det_type == 'Click_Detector_1' and 'PeakHz_10dB' in subset.columns:
                freqs = subset['PeakHz_10dB'].dropna()
                freq_label = 'Peak Hz (10dB)'
            elif 'freqMean' in subset.columns:
                freqs = subset['freqMean'].dropna()
                freq_label = 'Mean Frequency'
            else:
                freqs = subset['peak_freq'].dropna()
                freq_label = 'Peak Frequency'
            
            if len(freqs) > 0:
                ax2.hist(freqs/1000, bins=10, alpha=0.6, 
                        label=f'{det_type.replace("_", " ")} (n={len(freqs)})', 
                        color=colors[i % len(colors)], density=True)
        
        ax2.set_xlabel('Frequency (kHz)')
        ax2.set_ylabel('Density')
        ax2.set_title('Frequency Distributions')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        # Duration analysis
        ax3 = plt.subplot(3, 3, 3)
        durations_by_type = []
        labels = []
        
        for det_type in detection_types.unique():
            mask = detection_types == det_type
            subset = all_detections[mask]
            if 'duration' in subset.columns:
                durations = subset['duration'].dropna() * 1000  # Convert to ms
                if len(durations) > 0:
                    durations_by_type.append(durations)
                    labels.append(det_type.replace('_', '\n'))
        
        if durations_by_type:
            bp = ax3.boxplot(durations_by_type, labels=labels, patch_artist=True)
            for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
        
        ax3.set_ylabel('Duration (ms)')
        ax3.set_title('Duration by Detector Type')
        ax3.grid(True, alpha=0.3)
        plt.setp(ax3.get_xticklabels(), fontsize=8)
        
        # Whistle-specific parameters (for Whistle_and_Moan_Detector)
        ax4 = plt.subplot(3, 3, 4)
        whistle_data = all_detections[all_detections['detector_type'] == 'Whistle_and_Moan_Detector']
        if not whistle_data.empty and 'freqBeg' in whistle_data.columns and 'freqEnd' in whistle_data.columns:
            freq_beg = whistle_data['freqBeg'].dropna() / 1000
            freq_end = whistle_data['freqEnd'].dropna() / 1000
            
            ax4.scatter(freq_beg, freq_end, alpha=0.7, s=40, color='orange')
            ax4.plot([freq_beg.min(), freq_beg.max()], [freq_beg.min(), freq_beg.max()], 'k--', alpha=0.5)
            ax4.set_xlabel('Start Frequency (kHz)')
            ax4.set_ylabel('End Frequency (kHz)')
            ax4.set_title('Whistle Frequency Modulation')
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'No Whistle\nModulation Data', ha='center', va='center', 
                    transform=ax4.transAxes, fontsize=12)
            ax4.set_title('Whistle Frequency Modulation')
        
        # Bandwidth analysis (for Click_Detector_1)
        ax5 = plt.subplot(3, 3, 5)
        click_data = all_detections[all_detections['detector_type'] == 'Click_Detector_1']
        if not click_data.empty and 'BW_10dB' in click_data.columns and 'centerHz_10dB' in click_data.columns:
            bandwidth = click_data['BW_10dB'].dropna() / 1000
            center_freq = click_data['centerHz_10dB'].dropna() / 1000
            
            ax5.scatter(center_freq, bandwidth, alpha=0.7, s=60, color='blue')
            ax5.set_xlabel('Center Frequency (kHz)')
            ax5.set_ylabel('Bandwidth (kHz)')
            ax5.set_title('Click Bandwidth vs Center Frequency')
            ax5.grid(True, alpha=0.3)
        else:
            ax5.text(0.5, 0.5, 'No Click\nBandwidth Data', ha='center', va='center', 
                    transform=ax5.transAxes, fontsize=12)
            ax5.set_title('Click Bandwidth Analysis')
        
        # Inter-Click Interval (ICI) analysis for Cepstrum
        ax6 = plt.subplot(3, 3, 6)
        cepstrum_data = all_detections[all_detections['detector_type'] == 'Cepstrum_Detector']
        if not cepstrum_data.empty and 'ici' in cepstrum_data.columns:
            ici_values = cepstrum_data['ici'].dropna()
            
            ax6.hist(ici_values, bins=8, alpha=0.7, color='green', edgecolor='black')
            ax6.set_xlabel('Inter-Click Interval (s)')
            ax6.set_ylabel('Count')
            ax6.set_title('ICI Distribution (Cepstrum)')
            ax6.grid(True, alpha=0.3)
        else:
            ax6.text(0.5, 0.5, 'No ICI Data\nAvailable', ha='center', va='center', 
                    transform=ax6.transAxes, fontsize=12)
            ax6.set_title('Inter-Click Interval Analysis')
        
        # Event-based comparison
        ax7 = plt.subplot(3, 3, 7)
        event_counts = all_detections.groupby(['event_id', 'detector_type']).size().unstack(fill_value=0)
        
        event_counts.plot(kind='bar', ax=ax7, color=colors[:len(event_counts.columns)], alpha=0.7)
        ax7.set_xlabel('Event ID')
        ax7.set_ylabel('Detection Count')
        ax7.set_title('Detections per Event')
        ax7.legend(title='Detector Type', fontsize=8, title_fontsize=8)
        ax7.grid(True, alpha=0.3)
        plt.setp(ax7.get_xticklabels(), rotation=45)
        
        # Acoustic parameter correlations (for whistles)
        ax8 = plt.subplot(3, 3, 8)
        if not whistle_data.empty and 'freqMean' in whistle_data.columns and 'freqStdDev' in whistle_data.columns:
            freq_mean = whistle_data['freqMean'].dropna() / 1000
            freq_std = whistle_data['freqStdDev'].dropna() / 1000
            
            ax8.scatter(freq_mean, freq_std, alpha=0.7, s=50, color='purple')
            ax8.set_xlabel('Mean Frequency (kHz)')
            ax8.set_ylabel('Frequency Std Dev (kHz)')
            ax8.set_title('Whistle Frequency Variability')
            ax8.grid(True, alpha=0.3)
        else:
            ax8.text(0.5, 0.5, 'No Frequency\nVariability Data', ha='center', va='center', 
                    transform=ax8.transAxes, fontsize=12)
            ax8.set_title('Frequency Variability Analysis')
        
        # Enhanced summary statistics
        ax9 = plt.subplot(3, 3, 9)
        
        # Calculate detector-specific statistics
        click_stats = click_data[['PeakHz_10dB', 'BW_10dB', 'duration']].describe() if not click_data.empty else None
        whistle_stats = whistle_data[['freqMean', 'freqStdDev', 'duration']].describe() if not whistle_data.empty else None
        
        summary_text = f"""Real Study Data Summary:
Study ID: ExampleData_10-12-2020
Source: PAMpal R Package extdata

Total Detections: {len(all_detections)}
Events: {len(all_detections['event_id'].unique())}
Detector Types: {len(detection_types.unique())}

Detection Breakdown:
• Clicks: {len(click_data)} detections
• Whistles: {len(whistle_data)} detections  
• Cepstrum: {len(cepstrum_data)} detections

Frequency Ranges:
• Clicks: {click_data['PeakHz_10dB'].min():.1f} - {click_data['PeakHz_10dB'].max():.1f} Hz
• Whistles: {whistle_data['freqMean'].min()/1000:.1f} - {whistle_data['freqMean'].max()/1000:.1f} kHz

Data Type: Test/Demo Data
Temporal Scope: {(pd.to_datetime(all_detections['UTC']).max() - pd.to_datetime(all_detections['UTC']).min()).total_seconds():.1f} sec"""

        ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes,
                 verticalalignment='top', fontfamily='monospace', fontsize=8,
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        ax9.set_xlim(0, 1)
        ax9.set_ylim(0, 1)
        ax9.axis('off')
        ax9.set_title('Enhanced Study Statistics')
        
        plt.suptitle('Complete Study Analysis Overview', fontsize=16)
        plt.tight_layout()
        plt.savefig('workflow_study_overview.png', dpi=300, bbox_inches='tight')
        print("   💾 Saved: workflow_study_overview.png")
    
    else:
        print("   ⚠️  No detection data found in study - skipping study analysis")
    
    print_section_header("Publication Figure Generation")
    
    # Create a publication-quality summary figure
    print_step(6, "Creating publication summary figure")
    
    # Use publication theme
    pub_theme = PublicationTheme('nature')
    colors = pub_theme.get_colors()
    
    fig5 = plt.figure(figsize=(8.5, 11))  # Nature single column
    
    # Recalculate spectrogram data for publication figure
    Sxx_db_pub, frequencies_pub, times_pub = calculate_spectrogram(waveform[:, 0], sample_rate)
    frequencies_pub = np.array(frequencies_pub)
    times_pub = np.array(times_pub)
    
    # Top panel: Click waveform and spectrogram
    ax1 = plt.subplot(4, 2, (1, 2))
    ax1.plot(time_ms, waveform[:, 0], 'k-', linewidth=0.8)
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Amplitude')
    ax1.set_title('A. Delphinid Click Waveform', loc='left', fontweight='bold')
    ax1.grid(True, alpha=0.2)
    
    ax2 = plt.subplot(4, 2, (3, 4))
    im = ax2.imshow(Sxx_db_pub, aspect='auto', origin='lower',
                    extent=[times_pub[0]*1000, times_pub[-1]*1000,
                           frequencies_pub[0]/1000, frequencies_pub[-1]/1000],
                    cmap='viridis')
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Frequency (kHz)')
    ax2.set_title('B. Spectrogram', loc='left', fontweight='bold')
    ax2.set_ylim(0, 150)
    
    # Middle panel: Whistle contour
    ax3 = plt.subplot(4, 2, 5)
    ax3.plot(time_contour, freq_contour/1000, color=colors['secondary'], linewidth=1.5)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Frequency (kHz)')
    ax3.set_title('C. Whistle Contour', loc='left', fontweight='bold')
    ax3.grid(True, alpha=0.2)
    
    # Cepstrogram
    ax4 = plt.subplot(4, 2, 6)
    im2 = ax4.imshow(cepstrum, aspect='auto', origin='lower',
                     extent=[ceps_time[0]*1000, ceps_time[-1]*1000,
                            quefrency[0]*1e6, quefrency[-1]*1e6],
                     cmap='hot')
    ax4.set_xlabel('Time (ms)')
    ax4.set_ylabel('Quefrency (μs)')
    ax4.set_title('D. Cepstrogram', loc='left', fontweight='bold')
    ax4.set_ylim(0, 50)
    
    # Detection overview if available
    if not all_detections.empty:
        ax5 = plt.subplot(4, 2, 7)
        for det_type in detection_types.unique():
            mask = detection_types == det_type
            freqs = all_detections.loc[mask, 'peak_freq'].dropna()
            if len(freqs) > 0:
                ax5.hist(freqs/1000, bins=15, alpha=0.6, label=det_type, density=True)
        ax5.set_xlabel('Peak Frequency (kHz)')
        ax5.set_ylabel('Density')
        ax5.set_title('E. Detection Frequencies', loc='left', fontweight='bold')
        ax5.legend(fontsize=8)
        ax5.grid(True, alpha=0.2)
        
        ax6 = plt.subplot(4, 2, 8)
        detection_counts = detection_types.value_counts()
        bars = ax6.bar(range(len(detection_counts)), detection_counts.values,
                      color=[colors['primary'], colors['secondary']], alpha=0.7)
        ax6.set_xticks(range(len(detection_counts)))
        ax6.set_xticklabels(detection_counts.index, rotation=45)
        ax6.set_ylabel('Count')
        ax6.set_title('F. Detection Summary', loc='left', fontweight='bold')
        
        # Add count labels
        for i, (bar, count) in enumerate(zip(bars, detection_counts.values)):
            ax6.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                    str(count), ha='center', va='bottom', fontsize=8)
    
    plt.suptitle('PAMpal Python: Marine Mammal Acoustic Analysis Suite', 
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('workflow_publication_summary.png', dpi=300, bbox_inches='tight')
    print("   💾 Saved: workflow_publication_summary.png")
    
    print_section_header("Workflow Complete")
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    workflow_summary = f"""
🎯 Comprehensive PAMpal Workflow Complete!

Generated Outputs:
✅ workflow_click_analysis.png - Complete click detection analysis
✅ workflow_whistle_analysis.png - Whistle contour characterization  
✅ workflow_cepstral_analysis.png - Harmonic structure analysis
✅ workflow_study_overview.png - Study-level detection patterns
✅ workflow_publication_summary.png - Publication-ready summary

Analysis Highlights:
• Processed {len(list_available_datasets())} different data types
• Demonstrated time-domain, frequency-domain, and cepstral analysis
• Generated publication-quality visualizations
• Extracted acoustic parameters and detection statistics
• Showcased complete research workflow from raw data to publication

Workflow completed: {timestamp}
"""
    
    print(workflow_summary)
    
    # Show plots if running interactively
    if __name__ == '__main__':
        try:
            plt.show()
        except:
            pass  # Don't fail if no display available

if __name__ == '__main__':
    main()