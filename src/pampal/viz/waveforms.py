"""
Waveform visualization module for PAMpal.

This module provides functions for visualizing time-domain acoustic signals,
including single and multi-channel waveforms, envelope plots, and detection overlays.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Union, Tuple
import warnings

from .core import VisualizationBase, ColorSchemes

# Optional import for signal processing
try:
    from ..signal_processing import extract_waveform_data
except ImportError:
    # Signal processing not available - define a dummy function
    def extract_waveform_data(*args, **kwargs):
        raise NotImplementedError("Signal processing module not available")


class WaveformPlotter(VisualizationBase):
    """Specialized plotter for waveform visualizations."""
    
    def __init__(self, theme='default'):
        super().__init__(theme)
        self.colors = ColorSchemes()
    
    def _format_amplitude_axis(self, ax):
        """Format amplitude axis with appropriate labels."""
        ax.set_ylabel('Amplitude')


def plot_waveform(waveform: np.ndarray, sample_rate: int = 192000,
                 time_offset: float = 0, title: str = None,
                 color: str = None, figsize: Tuple[float, float] = None,
                 show_envelope: bool = False, normalize: bool = True,
                 theme: str = 'default', **kwargs) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot a single waveform in the time domain.
    
    Args:
        waveform: 1D array of audio samples
        sample_rate: Sample rate in Hz
        time_offset: Time offset for x-axis in seconds
        title: Plot title
        color: Line color (default: blue)
        figsize: Figure size tuple (width, height)
        show_envelope: Whether to show signal envelope
        normalize: Whether to normalize amplitude to [-1, 1]
        theme: Visual theme to use
        **kwargs: Additional plot parameters
        
    Returns:
        Tuple of (figure, axes) objects
    """
    # Validate parameters
    if sample_rate <= 0:
        raise ValueError(f"Sample rate must be positive, got {sample_rate}")
        
    plotter = WaveformPlotter(theme)
    fig, ax = plotter._setup_figure(figsize=figsize)
    
    # Prepare time axis
    duration = len(waveform) / sample_rate
    time = np.linspace(time_offset, time_offset + duration, len(waveform))
    
    # Normalize if requested
    if normalize and np.max(np.abs(waveform)) > 0:
        waveform = waveform / np.max(np.abs(waveform))
    
    # Set default color
    if color is None:
        color = plotter.colors.primary_color()
    
    # Plot waveform
    ax.plot(time, waveform, color=color, linewidth=1, **kwargs)
    
    # Add envelope if requested
    if show_envelope:
        from scipy.signal import hilbert
        analytic_signal = hilbert(waveform)
        envelope = np.abs(analytic_signal)
        ax.plot(time, envelope, color='red', alpha=0.7, linewidth=1)
        ax.plot(time, -envelope, color='red', alpha=0.7, linewidth=1)
    
    # Format axes
    plotter._format_time_axis(ax, duration)
    plotter._format_amplitude_axis(ax)
    
    # Add title
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"Waveform (Duration: {duration:.3f}s, SR: {sample_rate}Hz)")
    
    return fig, ax


def plot_multi_waveform(waveforms: Dict[str, np.ndarray], sample_rate: int = 192000,
                       time_offset: float = 0, title: str = None,
                       colors: Dict[str, str] = None, figsize: Tuple[float, float] = None,
                       stack_vertical: bool = True, normalize: bool = True,
                       theme: str = 'default', **kwargs) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot multiple waveforms for comparison.
    
    Args:
        waveforms: Dictionary mapping names to waveform arrays
        sample_rate: Sample rate in Hz
        time_offset: Time offset for x-axis in seconds
        title: Plot title
        colors: Dictionary mapping names to colors
        figsize: Figure size tuple (width, height)
        stack_vertical: Whether to stack waveforms vertically or overlay
        normalize: Whether to normalize each waveform individually
        theme: Visual theme to use
        **kwargs: Additional plot parameters
        
    Returns:
        Tuple of (figure, axes) objects
    """
    # Validate parameters
    if sample_rate <= 0:
        raise ValueError(f"Sample rate must be positive, got {sample_rate}")
    
    if not waveforms:
        raise ValueError("No waveforms provided")
    
    plotter = WaveformPlotter(theme)
    
    # Set up figure and axes based on layout
    if stack_vertical:
        n_waveforms = len(waveforms)
        if figsize is None:
            figsize = (8, 1.5 * n_waveforms)
        fig, axes = plt.subplots(n_waveforms, 1, figsize=figsize, sharex=True)
        
        if n_waveforms == 1:
            axes = [axes]  # Make iterable for single waveform case
    else:
        fig, ax = plotter._setup_figure(figsize=figsize)
        axes = [ax] * len(waveforms)  # Same axis for all waveforms
    
    # Set default colors if not provided
    if colors is None:
        color_cycle = plotter.colors.color_cycle()
        colors = {name: color_cycle[i % len(color_cycle)] 
                 for i, name in enumerate(waveforms.keys())}
    
    # Plot each waveform
    for i, (name, waveform) in enumerate(waveforms.items()):
        ax = axes[i if stack_vertical else 0]
        
        # Prepare time axis
        duration = len(waveform) / sample_rate
        time = np.linspace(time_offset, time_offset + duration, len(waveform))
        
        # Normalize if requested
        if normalize and np.max(np.abs(waveform)) > 0:
            waveform = waveform / np.max(np.abs(waveform))
        
        # Get color
        color = colors.get(name, plotter.colors.primary_color())
        
        # Plot waveform
        ax.plot(time, waveform, color=color, linewidth=1, label=name, **kwargs)
        
        # Format axes
        plotter._format_amplitude_axis(ax)
        
        if stack_vertical:
            ax.set_title(name)
        
        # Only add x-axis labels to bottom plot if stacked
        if i == len(waveforms) - 1 or not stack_vertical:
            plotter._format_time_axis(ax, duration)
    
    # Add legend if overlaid
    if not stack_vertical:
        axes[0].legend()
    
    # Add title
    if title:
        if stack_vertical:
            fig.suptitle(title)
        else:
            axes[0].set_title(title)
    
    # Adjust layout
    fig.tight_layout()
    if stack_vertical and title:
        fig.subplots_adjust(top=0.9)
    
    return fig, axes


def plot_waveform_envelope(waveform: np.ndarray, sample_rate: int = 192000,
                          time_offset: float = 0, title: str = None,
                          method: str = 'hilbert', window_size: int = None,
                          figsize: Tuple[float, float] = None,
                          theme: str = 'default', **kwargs) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot waveform with emphasis on signal envelope.
    
    Args:
        waveform: 1D array of audio samples
        sample_rate: Sample rate in Hz
        time_offset: Time offset for x-axis in seconds
        title: Plot title
        method: Envelope extraction method ('hilbert', 'peak', 'rms')
        window_size: Window size for peak/RMS methods
        figsize: Figure size tuple (width, height)
        theme: Visual theme to use
        **kwargs: Additional plot parameters
        
    Returns:
        Tuple of (figure, axes) objects
    """
    # Validate parameters
    if sample_rate <= 0:
        raise ValueError(f"Sample rate must be positive, got {sample_rate}")
    
    if method not in ['hilbert', 'peak', 'rms']:
        raise ValueError(f"Unknown envelope method: {method}")
    
    plotter = WaveformPlotter(theme)
    fig, ax = plotter._setup_figure(figsize=figsize)
    
    # Prepare time axis
    duration = len(waveform) / sample_rate
    time = np.linspace(time_offset, time_offset + duration, len(waveform))
    
    # Calculate envelope based on method
    if method == 'hilbert':
        from scipy.signal import hilbert
        analytic_signal = hilbert(waveform)
        envelope = np.abs(analytic_signal)
    
    elif method == 'peak':
        if window_size is None:
            window_size = int(sample_rate * 0.01)  # 10ms default
        
        envelope = np.zeros_like(waveform)
        for i in range(0, len(waveform), window_size):
            end = min(i + window_size, len(waveform))
            envelope[i:end] = np.max(np.abs(waveform[i:end]))
    
    elif method == 'rms':
        if window_size is None:
            window_size = int(sample_rate * 0.01)  # 10ms default
        
        envelope = np.zeros_like(waveform)
        for i in range(0, len(waveform), window_size):
            end = min(i + window_size, len(waveform))
            envelope[i:end] = np.sqrt(np.mean(waveform[i:end]**2))
    
    # Plot waveform
    ax.plot(time, waveform, color='gray', alpha=0.5, linewidth=0.5)
    
    # Plot envelope
    ax.plot(time, envelope, color='red', linewidth=1.5, label='Upper envelope')
    ax.plot(time, -envelope, color='red', linewidth=1.5, label='Lower envelope')
    
    # Fill between envelopes
    ax.fill_between(time, envelope, -envelope, color='red', alpha=0.2)
    
    # Format axes
    plotter._format_time_axis(ax, duration)
    plotter._format_amplitude_axis(ax)
    
    # Add title
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"Waveform Envelope ({method.upper()}, Duration: {duration:.3f}s)")
    
    return fig, ax
