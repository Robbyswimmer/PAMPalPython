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
from ..signal_processing import extract_waveform_data


class WaveformPlotter(VisualizationBase):
    """Specialized plotter for waveform visualizations."""
    
    def __init__(self, theme='default'):
        super().__init__(theme)
        self.colors = ColorSchemes()


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
