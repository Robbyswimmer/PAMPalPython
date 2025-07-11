"""
Advanced analysis visualization module for PAMpal.

This module provides functions for specialized acoustic analysis visualizations,
including cepstrograms, Wigner-Ville distributions, depth analysis, and bearing analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Union, Tuple
import warnings

from .core import VisualizationBase, ColorSchemes

# Optional import for signal processing
try:
    from ..signal_processing import calculate_cepstrum
except ImportError:
    # Signal processing not available - define dummy function
    def calculate_cepstrum(waveform, sample_rate, **kwargs):
        # Simple fake cepstrum for testing
        import numpy as np
        n_quefrency = len(waveform) // 2
        cepstrum = np.random.random(n_quefrency) * 20 - 40  # dB range
        quefrency_bins = np.linspace(0, 0.01, n_quefrency)  # 0-10ms quefrency
        return cepstrum, quefrency_bins


class AdvancedPlotter(VisualizationBase):
    """Specialized plotter for advanced acoustic analysis visualizations."""
    
    def __init__(self, theme='default'):
        super().__init__(theme)
        self.colors = ColorSchemes()


def plot_cepstrogram(waveform: np.ndarray, sample_rate: int = 192000,
                    window_size: int = 512, overlap: float = 0.75,
                    quefrency_range: Tuple[float, float] = None,
                    time_range: Tuple[float, float] = None,
                    colormap: str = 'viridis', title: str = None,
                    figsize: Tuple[float, float] = None,
                    theme: str = 'default', **kwargs) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot a cepstrogram (cepstral analysis over time) of an acoustic signal.
    
    Args:
        waveform: 1D array of audio samples
        sample_rate: Sample rate in Hz
        window_size: Window size for cepstral analysis
        overlap: Overlap between windows (0-1)
        quefrency_range: Quefrency range to display (min_q, max_q) in seconds
        time_range: Time range to display (start_time, end_time) in seconds
        colormap: Colormap name
        title: Plot title
        figsize: Figure size tuple (width, height)
        theme: Visual theme to use
        **kwargs: Additional parameters
        
    Returns:
        Tuple of (figure, axes) objects
    """
    plotter = AdvancedPlotter(theme)
    fig, ax = plotter._setup_figure(figsize=figsize)
    
    # Calculate windowed cepstral analysis
    hop_size = int(window_size * (1 - overlap))
    n_windows = (len(waveform) - window_size) // hop_size + 1
    
    cepstrograms = []
    quefrencies = None
    
    for i in range(n_windows):
        start_idx = i * hop_size
        end_idx = start_idx + window_size
        window_data = waveform[start_idx:end_idx]
        
        # Apply window function
        window_data = window_data * np.hanning(len(window_data))
        
        # Calculate cepstrum
        cepstrum, quefrency_bins = calculate_cepstrum(window_data, sample_rate)
        
        if quefrencies is None:
            quefrencies = quefrency_bins
        
        cepstrograms.append(cepstrum)
    
    # Convert to 2D array
    cepstrogram_matrix = np.array(cepstrograms).T
    
    # Create time axis
    times = np.arange(n_windows) * hop_size / sample_rate
    
    # Apply quefrency range filter
    if quefrency_range is not None:
        q_mask = (quefrencies >= quefrency_range[0]) & (quefrencies <= quefrency_range[1])
        quefrencies = quefrencies[q_mask]
        cepstrogram_matrix = cepstrogram_matrix[q_mask, :]
    
    # Apply time range filter
    if time_range is not None:
        t_mask = (times >= time_range[0]) & (times <= time_range[1])
        times = times[t_mask]
        cepstrogram_matrix = cepstrogram_matrix[:, t_mask]
    
    # Plot cepstrogram
    im = ax.pcolormesh(times, quefrencies, cepstrogram_matrix, 
                      shading='auto', cmap=colormap)
    
    # Format axes
    plotter._format_time_axis(ax, times[-1] - times[0] if len(times) > 0 else None)
    ax.set_ylabel('Quefrency (s)')
    
    # Add colorbar
    plotter._add_colorbar(im, ax, label='Cepstral Magnitude')
    
    # Add title
    if title:
        ax.set_title(title)
    else:
        duration = len(waveform) / sample_rate
        ax.set_title(f'Cepstrogram (Duration: {duration:.3f}s, SR: {sample_rate}Hz)')
    
    plt.tight_layout()
    return fig, ax


def plot_wigner_ville(waveform: np.ndarray, sample_rate: int = 192000,
                     freq_range: Tuple[float, float] = None,
                     time_range: Tuple[float, float] = None,
                     smoothing_window: int = 9,
                     colormap: str = 'viridis', title: str = None,
                     figsize: Tuple[float, float] = None,
                     theme: str = 'default') -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot Wigner-Ville distribution of an acoustic signal.
    
    Args:
        waveform: 1D array of audio samples
        sample_rate: Sample rate in Hz
        freq_range: Frequency range to display (min_freq, max_freq) in Hz
        time_range: Time range to display (start_time, end_time) in seconds
        smoothing_window: Window size for smoothing (odd number)
        colormap: Colormap name
        title: Plot title
        figsize: Figure size tuple (width, height)
        theme: Visual theme to use
        
    Returns:
        Tuple of (figure, axes) objects
    """
    plotter = AdvancedPlotter(theme)
    fig, ax = plotter._setup_figure(figsize=figsize)
    
    # Calculate Wigner-Ville distribution
    wv_dist, frequencies, times = _calculate_wigner_ville(
        waveform, sample_rate, smoothing_window
    )
    
    # Apply frequency range filter
    if freq_range is not None:
        freq_mask = (frequencies >= freq_range[0]) & (frequencies <= freq_range[1])
        frequencies = frequencies[freq_mask]
        wv_dist = wv_dist[freq_mask, :]
    
    # Apply time range filter
    if time_range is not None:
        time_mask = (times >= time_range[0]) & (times <= time_range[1])
        times = times[time_mask]
        wv_dist = wv_dist[:, time_mask]
    
    # Plot Wigner-Ville distribution
    im = ax.pcolormesh(times, frequencies, wv_dist, shading='auto', cmap=colormap)
    
    # Format axes
    plotter._format_time_axis(ax, times[-1] - times[0] if len(times) > 0 else None)
    plotter._format_frequency_axis(ax, np.max(frequencies) if len(frequencies) > 0 else None)
    
    # Add colorbar
    plotter._add_colorbar(im, ax, label='Wigner-Ville Magnitude')
    
    # Add title
    if title:
        ax.set_title(title)
    else:
        duration = len(waveform) / sample_rate
        ax.set_title(f'Wigner-Ville Distribution (Duration: {duration:.3f}s)')
    
    plt.tight_layout()
    return fig, ax


def _calculate_wigner_ville(signal: np.ndarray, sample_rate: int, 
                          smoothing_window: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate smoothed Wigner-Ville distribution.
    
    This is a simplified implementation. For production use, consider using
    specialized libraries like PyTFTB for more advanced time-frequency analysis.
    """
    N = len(signal)
    
    # Ensure smoothing window is odd
    if smoothing_window % 2 == 0:
        smoothing_window += 1
    
    # Initialize WVD matrix
    wvd = np.zeros((N, N), dtype=complex)
    
    # Calculate WVD
    for n in range(N):
        # Define the range for m
        m_max = min(n, N - 1 - n, (smoothing_window - 1) // 2)
        
        for m in range(-m_max, m_max + 1):
            if n + m < N and n - m >= 0:
                wvd[n, n] += signal[n + m] * np.conj(signal[n - m]) * np.exp(-4j * np.pi * m * n / N)
    
    # Take real part and apply smoothing
    wvd_real = np.real(wvd)
    
    # Simple smoothing (can be improved with proper windowing)
    from scipy.ndimage import gaussian_filter
    wvd_smoothed = gaussian_filter(wvd_real, sigma=1.0)
    
    # Create frequency and time axes
    frequencies = np.fft.fftfreq(N, 1/sample_rate)[:N//2]
    times = np.arange(N) / sample_rate
    
    # Return only positive frequencies
    return wvd_smoothed[:N//2, :], frequencies, times


def plot_depth_analysis(depth_data: pd.DataFrame, 
                       depth_estimates: np.ndarray = None,
                       confidence_intervals: np.ndarray = None,
                       title: str = None, figsize: Tuple[float, float] = None,
                       theme: str = 'default') -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot depth analysis results including depth estimates and confidence intervals.
    
    Args:
        depth_data: DataFrame with depth analysis data
        depth_estimates: Array of depth estimates
        confidence_intervals: Array of confidence intervals [lower, upper]
        title: Plot title
        figsize: Figure size tuple (width, height)
        theme: Visual theme to use
        
    Returns:
        Tuple of (figure, axes array)
    """
    plotter = AdvancedPlotter(theme)
    
    if figsize is None:
        figsize = (12, 8)
    
    fig, axes = plotter._setup_subplots(nrows=2, ncols=2, figsize=figsize)
    
    # Depth estimates over time
    if 'time' in depth_data.columns and 'depth' in depth_data.columns:
        times = depth_data['time'].values
        depths = depth_data['depth'].values
        
        axes[0, 0].plot(times, depths, 'o-', color=plotter.colors.detection_colors()['click'],
                       alpha=0.7, markersize=4)
        
        if confidence_intervals is not None and len(confidence_intervals) == len(depths):
            axes[0, 0].fill_between(times, confidence_intervals[:, 0], confidence_intervals[:, 1],
                                   alpha=0.3, color=plotter.colors.detection_colors()['click'])
        
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Depth (m)')
        axes[0, 0].set_title('Depth Estimates Over Time')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].invert_yaxis()  # Depth increases downward
    else:
        axes[0, 0].text(0.5, 0.5, 'No depth time series data available',
                       ha='center', va='center', transform=axes[0, 0].transAxes)
        axes[0, 0].set_title('Depth Time Series Not Available')
    
    # Depth distribution histogram
    if 'depth' in depth_data.columns:
        depths = depth_data['depth'].dropna()
        if len(depths) > 0:
            axes[0, 1].hist(depths, bins=30, alpha=0.7, orientation='horizontal',
                           color=plotter.colors.detection_colors()['click'])
            axes[0, 1].set_ylabel('Depth (m)')
            axes[0, 1].set_xlabel('Count')
            axes[0, 1].set_title('Depth Distribution')
            axes[0, 1].invert_yaxis()
            axes[0, 1].grid(True, alpha=0.3)
        else:
            axes[0, 1].text(0.5, 0.5, 'No valid depth data',
                           ha='center', va='center', transform=axes[0, 1].transAxes)
    else:
        axes[0, 1].text(0.5, 0.5, 'No depth data available',
                       ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('Depth Data Not Available')
    
    # Echo delay analysis
    if 'echo_delay' in depth_data.columns:
        delays = depth_data['echo_delay'].dropna()
        if len(delays) > 0:
            axes[1, 0].hist(delays * 1000, bins=30, alpha=0.7,  # Convert to ms
                           color=plotter.colors.detection_colors()['click'])
            axes[1, 0].set_xlabel('Echo Delay (ms)')
            axes[1, 0].set_ylabel('Count')
            axes[1, 0].set_title('Echo Delay Distribution')
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'No valid delay data',
                           ha='center', va='center', transform=axes[1, 0].transAxes)
    else:
        axes[1, 0].text(0.5, 0.5, 'No echo delay data available',
                       ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Echo Delay Data Not Available')
    
    # Summary statistics
    stats_text = "Depth Analysis Summary:\n\n"
    
    if 'depth' in depth_data.columns:
        depths = depth_data['depth'].dropna()
        if len(depths) > 0:
            stats_text += f"Depth estimates: {len(depths)}\n"
            stats_text += f"Mean depth: {np.mean(depths):.1f} m\n"
            stats_text += f"Median depth: {np.median(depths):.1f} m\n"
            stats_text += f"Depth range: {np.min(depths):.1f}-{np.max(depths):.1f} m\n"
            stats_text += f"Std depth: {np.std(depths):.1f} m\n\n"
    
    if 'echo_delay' in depth_data.columns:
        delays = depth_data['echo_delay'].dropna()
        if len(delays) > 0:
            stats_text += f"Mean delay: {np.mean(delays)*1000:.2f} ms\n"
            stats_text += f"Delay range: {np.min(delays)*1000:.2f}-{np.max(delays)*1000:.2f} ms\n"
    
    if 'confidence' in depth_data.columns:
        conf = depth_data['confidence'].dropna()
        if len(conf) > 0:
            stats_text += f"Mean confidence: {np.mean(conf):.3f}\n"
    
    axes[1, 1].text(0.05, 0.95, stats_text, transform=axes[1, 1].transAxes,
                   verticalalignment='top', fontfamily='monospace', fontsize=9)
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_xticks([])
    axes[1, 1].set_yticks([])
    axes[1, 1].set_title('Depth Statistics')
    
    if title:
        fig.suptitle(title)
    else:
        n_estimates = len(depth_data)
        fig.suptitle(f'Depth Analysis (n={n_estimates} estimates)')
    
    plt.tight_layout()
    return fig, axes


def plot_bearing_analysis(bearing_data: pd.DataFrame,
                         array_geometry: Dict[str, Tuple[float, float]] = None,
                         title: str = None, figsize: Tuple[float, float] = None,
                         theme: str = 'default') -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot bearing analysis results for hydrophone array data.
    
    Args:
        bearing_data: DataFrame with bearing analysis data
        array_geometry: Dictionary mapping hydrophone IDs to (x, y) positions
        title: Plot title
        figsize: Figure size tuple (width, height)
        theme: Visual theme to use
        
    Returns:
        Tuple of (figure, axes array)
    """
    plotter = AdvancedPlotter(theme)
    
    if figsize is None:
        figsize = (12, 8)
    
    fig, axes = plotter._setup_subplots(nrows=2, ncols=2, figsize=figsize)
    
    # Bearing over time
    if 'time' in bearing_data.columns and 'bearing' in bearing_data.columns:
        times = bearing_data['time'].values
        bearings = bearing_data['bearing'].values
        
        axes[0, 0].plot(times, bearings, 'o-', 
                       color=plotter.colors.detection_colors()['click'],
                       alpha=0.7, markersize=4)
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Bearing (degrees)')
        axes[0, 0].set_title('Bearing Over Time')
        axes[0, 0].set_ylim(0, 360)
        axes[0, 0].grid(True, alpha=0.3)
    else:
        axes[0, 0].text(0.5, 0.5, 'No bearing time series data available',
                       ha='center', va='center', transform=axes[0, 0].transAxes)
        axes[0, 0].set_title('Bearing Time Series Not Available')
    
    # Bearing polar plot
    if 'bearing' in bearing_data.columns:
        bearings = bearing_data['bearing'].dropna()
        if len(bearings) > 0:
            # Convert to polar coordinates
            ax_polar = plt.subplot(2, 2, 2, projection='polar')
            bearings_rad = np.deg2rad(bearings)
            
            # Create histogram in polar coordinates
            bins = np.linspace(0, 2*np.pi, 37)  # 36 bins = 10 degree bins
            hist, _ = np.histogram(bearings_rad, bins=bins)
            
            # Plot polar histogram
            theta = bins[:-1] + np.diff(bins)/2
            ax_polar.bar(theta, hist, width=np.diff(bins)[0], alpha=0.7,
                        color=plotter.colors.detection_colors()['click'])
            
            ax_polar.set_theta_zero_location('N')  # North at top
            ax_polar.set_theta_direction(-1)  # Clockwise
            ax_polar.set_title('Bearing Distribution\n(Polar)', pad=20)
        else:
            axes[0, 1].text(0.5, 0.5, 'No valid bearing data',
                           ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Bearing Data Not Available')
    else:
        axes[0, 1].text(0.5, 0.5, 'No bearing data available',
                       ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('Bearing Data Not Available')
    
    # Array geometry plot
    if array_geometry:
        for hydrophone_id, (x, y) in array_geometry.items():
            axes[1, 0].plot(x, y, 'o', markersize=8, 
                           color=plotter.colors.detection_colors()['click'])
            axes[1, 0].annotate(hydrophone_id, (x, y), xytext=(5, 5), 
                               textcoords='offset points', fontsize=8)
        
        axes[1, 0].set_xlabel('X Position (m)')
        axes[1, 0].set_ylabel('Y Position (m)')
        axes[1, 0].set_title('Array Geometry')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axis('equal')
    else:
        axes[1, 0].text(0.5, 0.5, 'No array geometry data available',
                       ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Array Geometry Not Available')
    
    # Summary statistics
    stats_text = "Bearing Analysis Summary:\n\n"
    
    if 'bearing' in bearing_data.columns:
        bearings = bearing_data['bearing'].dropna()
        if len(bearings) > 0:
            stats_text += f"Bearing estimates: {len(bearings)}\n"
            stats_text += f"Mean bearing: {np.mean(bearings):.1f}°\n"
            stats_text += f"Median bearing: {np.median(bearings):.1f}°\n"
            stats_text += f"Bearing range: {np.min(bearings):.1f}°-{np.max(bearings):.1f}°\n"
            
            # Circular statistics
            bearings_rad = np.deg2rad(bearings)
            mean_sin = np.mean(np.sin(bearings_rad))
            mean_cos = np.mean(np.cos(bearings_rad))
            circular_mean = np.rad2deg(np.arctan2(mean_sin, mean_cos))
            if circular_mean < 0:
                circular_mean += 360
            
            circular_std = np.sqrt(-2 * np.log(np.sqrt(mean_sin**2 + mean_cos**2)))
            circular_std = np.rad2deg(circular_std)
            
            stats_text += f"Circular mean: {circular_mean:.1f}°\n"
            stats_text += f"Circular std: {circular_std:.1f}°\n\n"
    
    if 'confidence' in bearing_data.columns:
        conf = bearing_data['confidence'].dropna()
        if len(conf) > 0:
            stats_text += f"Mean confidence: {np.mean(conf):.3f}\n"
    
    if array_geometry:
        stats_text += f"Array elements: {len(array_geometry)}\n"
    
    axes[1, 1].text(0.05, 0.95, stats_text, transform=axes[1, 1].transAxes,
                   verticalalignment='top', fontfamily='monospace', fontsize=9)
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_xticks([])
    axes[1, 1].set_yticks([])
    axes[1, 1].set_title('Bearing Statistics')
    
    if title:
        fig.suptitle(title)
    else:
        n_estimates = len(bearing_data)
        fig.suptitle(f'Bearing Analysis (n={n_estimates} estimates)')
    
    plt.tight_layout()
    return fig, axes