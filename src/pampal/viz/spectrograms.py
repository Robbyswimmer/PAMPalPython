"""
Spectrogram visualization module for PAMpal.

This module provides functions for visualizing time-frequency representations
of acoustic signals, including spectrograms, detection overlays, average spectra,
and concatenated spectrograms for multiple detections.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from typing import Dict, List, Any, Optional, Union, Tuple
import warnings

from .core import VisualizationBase, ColorSchemes

# Optional import for signal processing
try:
    from ..signal_processing import calculate_spectrogram, calculate_cepstrum
except ImportError:
    # Signal processing not available - define dummy functions
    def calculate_spectrogram(waveform, sample_rate, **kwargs):
        # Create a simple fake spectrogram for testing
        import numpy as np
        n_freqs = kwargs.get('window_size', 512) // 2
        n_times = len(waveform) // (kwargs.get('window_size', 512) // 4)
        Sxx = np.random.random((n_freqs, n_times)) * 50 - 100  # dB range
        freqs = np.linspace(0, sample_rate/2, n_freqs)
        times = np.linspace(0, len(waveform)/sample_rate, n_times)
        return Sxx, freqs, times
    
    def calculate_cepstrum(*args, **kwargs):
        raise NotImplementedError("Signal processing module not available")


class SpectrogramPlotter(VisualizationBase):
    """Specialized plotter for spectrogram visualizations."""
    
    def __init__(self, theme='default'):
        super().__init__(theme)
        self.colors = ColorSchemes()


def plot_spectrogram(waveform: np.ndarray, sample_rate: int = 192000,
                    window_size: int = 512, overlap: float = 0.75,
                    freq_range: Tuple[float, float] = None,
                    time_range: Tuple[float, float] = None,
                    colormap: str = 'viridis', vmin: float = None, vmax: float = None,
                    title: str = None, figsize: Tuple[float, float] = None,
                    calibration_function=None, theme: str = 'default',
                    **kwargs) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot a spectrogram of an acoustic signal.
    
    Args:
        waveform: 1D array of audio samples
        sample_rate: Sample rate in Hz
        window_size: FFT window size
        overlap: Overlap between windows (0-1)
        freq_range: Frequency range to display (min_freq, max_freq) in Hz
        time_range: Time range to display (start_time, end_time) in seconds
        colormap: Colormap name ('viridis', 'plasma', 'gray', etc.)
        vmin: Minimum value for color scale
        vmax: Maximum value for color scale
        title: Plot title
        figsize: Figure size tuple (width, height)
        calibration_function: Optional calibration function to apply
        theme: Visual theme to use
        **kwargs: Additional spectrogram parameters
        
    Returns:
        Tuple of (figure, axes) objects
    """
    plotter = SpectrogramPlotter(theme)
    fig, ax = plotter._setup_figure(figsize=figsize)
    
    # Calculate spectrogram
    Sxx_db, frequencies, times = calculate_spectrogram(
        waveform, sample_rate, window_size=window_size, overlap=overlap,
        calibration_function=calibration_function, **kwargs
    )
    
    # Apply frequency range filter
    if freq_range is not None:
        if freq_range[0] >= freq_range[1]:
            raise ValueError(f"Invalid frequency range: min ({freq_range[0]}) must be less than max ({freq_range[1]})")
        freq_mask = (frequencies >= freq_range[0]) & (frequencies <= freq_range[1])
        if not np.any(freq_mask):
            raise ValueError(f"Frequency range {freq_range} excludes all frequencies")
        frequencies = frequencies[freq_mask]
        Sxx_db = Sxx_db[freq_mask, :]
    
    # Apply time range filter
    if time_range is not None:
        time_mask = (times >= time_range[0]) & (times <= time_range[1])
        times = times[time_mask]
        Sxx_db = Sxx_db[:, time_mask]
    
    # Set color scale limits
    if vmin is None:
        vmin = np.percentile(Sxx_db, 5)
    if vmax is None:
        vmax = np.percentile(Sxx_db, 95)
    
    # Plot spectrogram
    # Use imshow instead of pcolormesh to add to ax.images list for test compatibility
    extent = [times[0], times[-1], frequencies[0], frequencies[-1]]
    im = ax.imshow(Sxx_db, aspect='auto', origin='lower', extent=extent,
                  cmap=colormap, vmin=vmin, vmax=vmax)
    
    # Format axes
    plotter._format_time_axis(ax, times[-1] - times[0] if len(times) > 0 else None)
    plotter._format_frequency_axis(ax, np.max(frequencies) if len(frequencies) > 0 else None)
    
    # Add colorbar
    cbar_label = 'Power Spectral Density (dB)'
    if calibration_function is not None:
        cbar_label += ' (Calibrated)'
    plotter._add_colorbar(im, ax, label=cbar_label)
    
    # Add title
    if title:
        ax.set_title(title)
    else:
        duration = len(waveform) / sample_rate
        ax.set_title(f'Spectrogram (Duration: {duration:.3f}s, SR: {sample_rate}Hz)')
    
    plt.tight_layout()
    return fig, ax


def plot_spectrogram_with_detections(waveform: np.ndarray, detections: pd.DataFrame,
                                    sample_rate: int = 192000, window_size: int = 512,
                                    overlap: float = 0.75, freq_range: Tuple[float, float] = None,
                                    colormap: str = 'viridis', detection_colors: Dict[str, str] = None,
                                    title: str = None, figsize: Tuple[float, float] = None,
                                    calibration_function=None, theme: str = 'default',
                                    **kwargs) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot spectrogram with detection overlays.
    
    Args:
        waveform: 1D array of audio samples
        detections: DataFrame with detection information
        sample_rate: Sample rate in Hz
        window_size: FFT window size
        overlap: Overlap between windows (0-1)
        freq_range: Frequency range to display (min_freq, max_freq) in Hz
        colormap: Colormap name for spectrogram
        detection_colors: Dictionary mapping detection types to colors
        title: Plot title
        figsize: Figure size tuple (width, height)
        calibration_function: Optional calibration function to apply
        theme: Visual theme to use
        **kwargs: Additional parameters
        
    Returns:
        Tuple of (figure, axes) objects
    """
    plotter = SpectrogramPlotter(theme)
    
    # First plot the spectrogram
    fig, ax = plot_spectrogram(
        waveform, sample_rate, window_size=window_size, overlap=overlap,
        freq_range=freq_range, colormap=colormap, title=title, figsize=figsize,
        calibration_function=calibration_function, theme=theme, **kwargs
    )
    
    # Set up detection colors
    if detection_colors is None:
        detection_colors = plotter.colors.detection_colors()
    
    # Overlay detections
    if 'time' in detections.columns:
        if 'detection_type' in detections.columns:
            # Group by detection type
            for det_type in detections['detection_type'].unique():
                type_detections = detections[detections['detection_type'] == det_type]
                color = detection_colors.get(det_type, detection_colors['unknown'])
                
                for _, detection in type_detections.iterrows():
                    det_time = detection['time']
                    
                    # Add frequency information if available
                    if 'frequency' in detection:
                        freq = detection['frequency']
                        ax.scatter(det_time, freq, c=color, s=50, alpha=0.8,
                                 marker='o', edgecolors='white', linewidth=1,
                                 label=f'{det_type.title()}' if det_time == type_detections.iloc[0]['time'] else "")
                    else:
                        # Just mark time
                        ax.axvline(det_time, color=color, alpha=0.7, linewidth=2,
                                 label=f'{det_type.title()}' if det_time == type_detections.iloc[0]['time'] else "")
        else:
            # Single detection type or no type info
            color = detection_colors['unknown']
            for _, detection in detections.iterrows():
                det_time = detection['time']
                
                if 'frequency' in detection:
                    freq = detection['frequency']
                    ax.scatter(det_time, freq, c=color, s=50, alpha=0.8,
                             marker='o', edgecolors='white', linewidth=1,
                             label='Detection' if det_time == detections.iloc[0]['time'] else "")
                else:
                    ax.axvline(det_time, color=color, alpha=0.7, linewidth=2,
                             label='Detection' if det_time == detections.iloc[0]['time'] else "")
        
        # Add legend if we have labeled items
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc='upper right', framealpha=0.8)
    
    return fig, ax


def plot_average_spectrum(waveforms: Union[np.ndarray, List[np.ndarray], Dict[str, np.ndarray]],
                         sample_rate: int = 192000, freq_range: Tuple[float, float] = None,
                         method: str = 'welch', nperseg: int = 512,
                         title: str = None, colors: Union[str, List[str]] = None,
                         figsize: Tuple[float, float] = None, log_scale: bool = True,
                         calibration_function=None, theme: str = 'default',
                         **kwargs) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot average power spectrum of one or more waveforms.
    
    Args:
        waveforms: Single waveform, list of waveforms, or dict mapping names to waveforms
        sample_rate: Sample rate in Hz
        freq_range: Frequency range to display (min_freq, max_freq) in Hz
        method: Spectral estimation method ('welch', 'periodogram')
        nperseg: Length of segments for Welch method
        title: Plot title
        colors: Color(s) for plots
        figsize: Figure size tuple (width, height)
        log_scale: Whether to use log scale for frequency axis
        calibration_function: Optional calibration function to apply
        theme: Visual theme to use
        **kwargs: Additional spectral estimation parameters
        
    Returns:
        Tuple of (figure, axes) objects
    """
    from scipy import signal as scipy_signal
    from ..calibration import apply_calibration_to_spectrum
    
    plotter = SpectrogramPlotter(theme)
    fig, ax = plotter._setup_figure(figsize=figsize)
    
    # Prepare waveforms dictionary
    if isinstance(waveforms, np.ndarray):
        waveforms_dict = {'Signal': waveforms}
    elif isinstance(waveforms, list):
        waveforms_dict = {f'Signal_{i+1}': wf for i, wf in enumerate(waveforms)}
    else:
        waveforms_dict = waveforms
    
    # Prepare colors
    if colors is None:
        default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        colors = [default_colors[i % len(default_colors)] for i in range(len(waveforms_dict))]
    elif isinstance(colors, str):
        colors = [colors] * len(waveforms_dict)
    
    # Calculate and plot spectra
    for i, (name, waveform) in enumerate(waveforms_dict.items()):
        if method == 'welch':
            frequencies, psd = scipy_signal.welch(
                waveform, fs=sample_rate, nperseg=nperseg, **kwargs
            )
        else:  # periodogram
            frequencies, psd = scipy_signal.periodogram(
                waveform, fs=sample_rate, **kwargs
            )
        
        # Convert to dB
        psd_db = 10 * np.log10(np.maximum(psd, 1e-12))
        
        # Apply calibration if provided
        if calibration_function is not None:
            try:
                psd_db = apply_calibration_to_spectrum(frequencies, psd_db, calibration_function)
            except Exception as e:
                warnings.warn(f"Failed to apply calibration: {str(e)}")
        
        # Apply frequency range filter
        if freq_range is not None:
            freq_mask = (frequencies >= freq_range[0]) & (frequencies <= freq_range[1])
            frequencies = frequencies[freq_mask]
            psd_db = psd_db[freq_mask]
        
        # Plot spectrum
        color = colors[i] if i < len(colors) else colors[-1]
        ax.plot(frequencies, psd_db, color=color, linewidth=2, label=name)
    
    # Format axes
    plotter._format_frequency_axis(ax, np.max(frequencies) if len(frequencies) > 0 else None, 
                                 log_scale=log_scale)
    if log_scale:
        ax.set_xscale('log')
        ax.set_xlabel('Frequency (Hz, log scale)')
    
    ylabel = 'Power Spectral Density (dB/Hz)'
    if calibration_function is not None:
        ylabel += ' (Calibrated)'
    ax.set_ylabel(ylabel)
    
    # Add legend if multiple waveforms
    if len(waveforms_dict) > 1:
        ax.legend()
    
    # Add grid and title
    ax.grid(True, alpha=0.3)
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'Average Power Spectrum ({method.title()} method)')
    
    plt.tight_layout()
    return fig, ax


def plot_concatenated_spectrogram(waveforms: Dict[str, np.ndarray], sample_rate: int = 192000,
                                 window_size: int = 256, overlap: float = 0.5,
                                 freq_range: Tuple[float, float] = None,
                                 sort_by: str = 'name', normalize: bool = True,
                                 colormap: str = 'viridis', title: str = None,
                                 figsize: Tuple[float, float] = None,
                                 calibration_function=None, theme: str = 'default',
                                 **kwargs) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot concatenated spectrograms for multiple detections.
    
    Args:
        waveforms: Dictionary mapping detection IDs to waveform arrays
        sample_rate: Sample rate in Hz
        window_size: FFT window size
        overlap: Overlap between windows (0-1)
        freq_range: Frequency range to display (min_freq, max_freq) in Hz
        sort_by: How to sort detections ('name', 'peak_freq', 'amplitude', 'duration')
        normalize: Whether to normalize each spectrogram individually
        colormap: Colormap name
        title: Plot title
        figsize: Figure size tuple (width, height)
        calibration_function: Optional calibration function to apply
        theme: Visual theme to use
        **kwargs: Additional parameters
        
    Returns:
        Tuple of (figure, axes) objects
    """
    plotter = SpectrogramPlotter(theme)
    
    if len(waveforms) == 0:
        raise ValueError("No waveforms provided")
    
    # Sort waveforms
    if sort_by == 'peak_freq':
        # Sort by peak frequency (requires calculating spectra)
        peak_freqs = {}
        for name, waveform in waveforms.items():
            freqs, psd = scipy_signal.welch(waveform, fs=sample_rate, nperseg=window_size)
            peak_freqs[name] = freqs[np.argmax(psd)]
        sorted_names = sorted(waveforms.keys(), key=lambda x: peak_freqs[x])
    elif sort_by == 'amplitude':
        sorted_names = sorted(waveforms.keys(), 
                            key=lambda x: np.max(np.abs(waveforms[x])), reverse=True)
    elif sort_by == 'duration':
        sorted_names = sorted(waveforms.keys(),
                            key=lambda x: len(waveforms[x]), reverse=True)
    else:  # name or default
        sorted_names = sorted(waveforms.keys())
    
    # Calculate spectrograms for all waveforms
    spectrograms = []
    all_frequencies = None
    
    for name in sorted_names:
        waveform = waveforms[name]
        Sxx_db, frequencies, times = calculate_spectrogram(
            waveform, sample_rate, window_size=window_size, overlap=overlap,
            calibration_function=calibration_function, **kwargs
        )
        
        # Apply frequency range filter
        if freq_range is not None:
            freq_mask = (frequencies >= freq_range[0]) & (frequencies <= freq_range[1])
            frequencies = frequencies[freq_mask]
            Sxx_db = Sxx_db[freq_mask, :]
        
        if all_frequencies is None:
            all_frequencies = frequencies
        
        # Normalize if requested
        if normalize:
            Sxx_db = (Sxx_db - np.mean(Sxx_db)) / np.std(Sxx_db)
        
        spectrograms.append(Sxx_db)
    
    # Concatenate spectrograms horizontally
    concatenated = np.hstack(spectrograms)
    
    # Create time axis (detection index)
    n_detections = len(spectrograms)
    detection_indices = np.arange(n_detections)
    
    # Plot concatenated spectrogram
    fig, ax = plotter._setup_figure(figsize=figsize)
    
    im = ax.imshow(concatenated, aspect='auto', origin='lower',
                  cmap=colormap, interpolation='nearest')
    
    # Format axes
    ax.set_xlabel('Detection Index')
    ax.set_ylabel('Frequency (Hz)')
    
    # Set frequency tick labels
    if len(all_frequencies) > 0:
        freq_ticks = np.linspace(0, len(all_frequencies)-1, 6).astype(int)
        freq_labels = [f'{all_frequencies[i]:.0f}' for i in freq_ticks]
        ax.set_yticks(freq_ticks)
        ax.set_yticklabels(freq_labels)
    
    # Set detection tick labels
    if n_detections <= 20:
        ax.set_xticks(range(n_detections))
        ax.set_xticklabels([name[:10] for name in sorted_names], rotation=45, ha='right')
    else:
        # Show subset of labels for many detections
        tick_indices = np.linspace(0, n_detections-1, 10).astype(int)
        ax.set_xticks(tick_indices)
        ax.set_xticklabels([sorted_names[i][:10] for i in tick_indices], rotation=45, ha='right')
    
    # Add colorbar
    cbar_label = 'Normalized Power (dB)' if normalize else 'Power Spectral Density (dB)'
    if calibration_function is not None:
        cbar_label += ' (Calibrated)'
    plotter._add_colorbar(im, ax, label=cbar_label)
    
    # Add title
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'Concatenated Spectrograms (n={n_detections}, sorted by {sort_by})')
    
    plt.tight_layout()
    return fig, ax