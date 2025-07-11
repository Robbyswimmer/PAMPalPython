"""
Detection analysis visualization module for PAMpal.

This module provides functions for visualizing acoustic detection parameters,
including click parameter distributions, whistle contours, inter-click intervals,
and comprehensive detection overview plots.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Union, Tuple
import warnings

from .core import VisualizationBase, ColorSchemes

# Optional import for signal processing
try:
    from ..signal_processing import calculate_inter_click_intervals, extract_whistle_contour
except ImportError:
    # Signal processing not available - define dummy functions
    def calculate_inter_click_intervals(times):
        # Simple ICI calculation for testing
        import numpy as np
        intervals = np.diff(times)
        return {
            'intervals': intervals,
            'mean_ici': np.mean(intervals),
            'median_ici': np.median(intervals),
            'std_ici': np.std(intervals),
            'ici_cv': np.std(intervals) / np.mean(intervals) if np.mean(intervals) > 0 else 0,
            'regular_clicks': len(intervals)
        }
    
    def extract_whistle_contour(*args, **kwargs):
        raise NotImplementedError("Signal processing module not available")


class DetectionPlotter(VisualizationBase):
    """Specialized plotter for detection analysis visualizations."""
    
    def __init__(self, theme='default'):
        super().__init__(theme)
        self.colors = ColorSchemes()


def plot_detection_overview(detections: pd.DataFrame, 
                           detection_type: str = 'click',
                           figsize: Tuple[float, float] = None,
                           theme: str = 'default') -> Tuple[plt.Figure, np.ndarray]:
    """
    Create a comprehensive overview plot of detection data.
    
    Args:
        detections: DataFrame with detection parameters
        detection_type: Type of detection ('click', 'whistle', 'moan')
        figsize: Figure size tuple (width, height)
        theme: Visual theme to use
        
    Returns:
        Tuple of (figure, axes array)
    """
    plotter = DetectionPlotter(theme)
    
    if figsize is None:
        figsize = (15, 10)
    
    fig, axes = plotter._setup_subplots(nrows=2, ncols=3, figsize=figsize)
    
    # Time series plot
    if 'UTC' in detections.columns:
        times = pd.to_datetime(detections['UTC'])
        axes[0, 0].plot(times, range(len(times)), 'o-', markersize=3, alpha=0.7)
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Detection Number')
        axes[0, 0].set_title('Detections Over Time')
        axes[0, 0].tick_params(axis='x', rotation=45)
    else:
        axes[0, 0].text(0.5, 0.5, 'No time data available', 
                       ha='center', va='center', transform=axes[0, 0].transAxes)
        axes[0, 0].set_title('Time Data Not Available')
    
    # Frequency distribution
    if 'peak_freq' in detections.columns:
        freq_data = detections['peak_freq'].dropna()
        if len(freq_data) > 0:
            axes[0, 1].hist(freq_data / 1000, bins=30, alpha=0.7, 
                           color=plotter.colors.detection_colors()[detection_type])
            axes[0, 1].set_xlabel('Peak Frequency (kHz)')
            axes[0, 1].set_ylabel('Count')
            axes[0, 1].set_title('Peak Frequency Distribution')
        else:
            axes[0, 1].text(0.5, 0.5, 'No frequency data', 
                           ha='center', va='center', transform=axes[0, 1].transAxes)
    else:
        axes[0, 1].text(0.5, 0.5, 'No frequency data available', 
                       ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('Frequency Data Not Available')
    
    # Amplitude distribution
    amp_col = None
    for col in ['amplitude', 'peak_amplitude', 'rms_amplitude']:
        if col in detections.columns:
            amp_col = col
            break
    
    if amp_col:
        amp_data = detections[amp_col].dropna()
        if len(amp_data) > 0:
            axes[0, 2].hist(amp_data, bins=30, alpha=0.7,
                           color=plotter.colors.detection_colors()[detection_type])
            axes[0, 2].set_xlabel(f'{amp_col.replace("_", " ").title()}')
            axes[0, 2].set_ylabel('Count')
            axes[0, 2].set_title('Amplitude Distribution')
        else:
            axes[0, 2].text(0.5, 0.5, 'No amplitude data', 
                           ha='center', va='center', transform=axes[0, 2].transAxes)
    else:
        axes[0, 2].text(0.5, 0.5, 'No amplitude data available', 
                       ha='center', va='center', transform=axes[0, 2].transAxes)
        axes[0, 2].set_title('Amplitude Data Not Available')
    
    # Duration distribution (for clicks)
    if detection_type == 'click' and 'duration' in detections.columns:
        dur_data = detections['duration'].dropna()
        if len(dur_data) > 0:
            axes[1, 0].hist(dur_data * 1000, bins=30, alpha=0.7,
                           color=plotter.colors.detection_colors()[detection_type])
            axes[1, 0].set_xlabel('Duration (ms)')
            axes[1, 0].set_ylabel('Count')
            axes[1, 0].set_title('Duration Distribution')
        else:
            axes[1, 0].text(0.5, 0.5, 'No duration data', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
    else:
        axes[1, 0].text(0.5, 0.5, 'Duration data not available', 
                       ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Duration Data Not Available')
    
    # Inter-detection intervals
    if 'UTC' in detections.columns and len(detections) > 1:
        times = pd.to_datetime(detections['UTC'])
        time_seconds = (times - times.iloc[0]).dt.total_seconds().values
        intervals = np.diff(time_seconds)
        
        if len(intervals) > 0:
            axes[1, 1].hist(intervals, bins=30, alpha=0.7,
                           color=plotter.colors.detection_colors()[detection_type])
            axes[1, 1].set_xlabel('Inter-Detection Interval (s)')
            axes[1, 1].set_ylabel('Count')
            axes[1, 1].set_title('Inter-Detection Intervals')
        else:
            axes[1, 1].text(0.5, 0.5, 'Insufficient data for intervals', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
    else:
        axes[1, 1].text(0.5, 0.5, 'No time data for intervals', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Interval Data Not Available')
    
    # Summary statistics
    stats_text = f"Detection Summary:\n"
    stats_text += f"Total detections: {len(detections)}\n"
    
    if 'peak_freq' in detections.columns:
        freq_data = detections['peak_freq'].dropna()
        if len(freq_data) > 0:
            stats_text += f"Freq range: {freq_data.min()/1000:.1f}-{freq_data.max()/1000:.1f} kHz\n"
            stats_text += f"Mean freq: {freq_data.mean()/1000:.1f} kHz\n"
    
    if amp_col and len(detections[amp_col].dropna()) > 0:
        amp_data = detections[amp_col].dropna()
        stats_text += f"Amp range: {amp_data.min():.2f}-{amp_data.max():.2f}\n"
        stats_text += f"Mean amp: {amp_data.mean():.2f}\n"
    
    axes[1, 2].text(0.05, 0.95, stats_text, transform=axes[1, 2].transAxes,
                   verticalalignment='top', fontfamily='monospace', fontsize=10)
    axes[1, 2].set_xlim(0, 1)
    axes[1, 2].set_ylim(0, 1)
    axes[1, 2].set_xticks([])
    axes[1, 2].set_yticks([])
    axes[1, 2].set_title('Summary Statistics')
    
    fig.suptitle(f'{detection_type.title()} Detection Overview (n={len(detections)})')
    plt.tight_layout()
    
    return fig, axes


def plot_click_parameters(detections: pd.DataFrame, 
                         parameters: List[str] = None,
                         color_by: str = None,
                         figsize: Tuple[float, float] = None,
                         theme: str = 'default') -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot click parameter distributions and relationships.
    
    Args:
        detections: DataFrame with click parameters
        parameters: List of parameters to plot (default: common click parameters)
        color_by: Column to use for color coding
        figsize: Figure size tuple (width, height)
        theme: Visual theme to use
        
    Returns:
        Tuple of (figure, axes) objects
    """
    plotter = DetectionPlotter(theme)
    
    # Default parameters to plot
    if parameters is None:
        available_params = []
        common_params = ['peak_freq', 'centroid_freq', 'bandwidth', 'duration', 
                        'peak_amplitude', 'rms_amplitude', 'q_factor', 'snr']
        for param in common_params:
            if param in detections.columns:
                available_params.append(param)
        parameters = available_params[:6]  # Limit to 6 for display
    
    if len(parameters) == 0:
        raise ValueError("No valid parameters found in detections DataFrame")
    
    # Create pairplot-style visualization
    n_params = len(parameters)
    
    if figsize is None:
        figsize = (3 * n_params, 3 * n_params)
    
    fig, axes = plotter._setup_subplots(nrows=n_params, ncols=n_params, figsize=figsize)
    
    # Ensure axes is 2D
    if n_params == 1:
        axes = np.array([[axes]])
    elif axes.ndim == 1:
        axes = axes.reshape(-1, 1) if n_params > 1 else axes.reshape(1, -1)
    
    # Color setup
    colors = None
    if color_by and color_by in detections.columns:
        unique_values = detections[color_by].unique()
        color_palette = plotter.colors.species_colors() if len(unique_values) <= 7 else None
        if color_palette:
            colors = [color_palette.get(str(val), '#1f77b4') for val in detections[color_by]]
        else:
            colors = plt.cm.viridis(np.linspace(0, 1, len(unique_values)))
    
    for i, param_i in enumerate(parameters):
        for j, param_j in enumerate(parameters):
            ax = axes[i, j]
            
            if i == j:
                # Diagonal: histogram
                data = detections[param_i].dropna()
                if len(data) > 0:
                    if colors is not None and color_by:
                        # Color-coded histogram
                        for val in detections[color_by].unique():
                            mask = detections[color_by] == val
                            subset_data = detections.loc[mask, param_i].dropna()
                            if len(subset_data) > 0:
                                ax.hist(subset_data, bins=20, alpha=0.6, 
                                       label=str(val), density=True)
                        ax.legend(fontsize=8)
                    else:
                        ax.hist(data, bins=30, alpha=0.7, density=True,
                               color=plotter.colors.detection_colors()['click'])
                
                ax.set_xlabel(param_i.replace('_', ' ').title())
                ax.set_ylabel('Density')
                
            elif i > j:
                # Lower triangle: scatter plot
                data_x = detections[param_j].dropna()
                data_y = detections[param_i].dropna()
                
                # Find common indices
                common_idx = detections[param_j].notna() & detections[param_i].notna()
                
                if common_idx.sum() > 0:
                    x_data = detections.loc[common_idx, param_j]
                    y_data = detections.loc[common_idx, param_i]
                    
                    if colors is not None and color_by:
                        color_data = detections.loc[common_idx, color_by]
                        scatter = ax.scatter(x_data, y_data, c=color_data, 
                                           alpha=0.6, s=20, cmap='viridis')
                    else:
                        ax.scatter(x_data, y_data, alpha=0.6, s=20,
                                 color=plotter.colors.detection_colors()['click'])
                
                ax.set_xlabel(param_j.replace('_', ' ').title())
                ax.set_ylabel(param_i.replace('_', ' ').title())
                
            else:
                # Upper triangle: correlation coefficient
                data_x = detections[param_j].dropna()
                data_y = detections[param_i].dropna()
                
                common_idx = detections[param_j].notna() & detections[param_i].notna()
                
                if common_idx.sum() > 2:
                    x_data = detections.loc[common_idx, param_j]
                    y_data = detections.loc[common_idx, param_i]
                    corr = np.corrcoef(x_data, y_data)[0, 1]
                    
                    ax.text(0.5, 0.5, f'r = {corr:.3f}', 
                           ha='center', va='center', transform=ax.transAxes,
                           fontsize=12, fontweight='bold')
                    
                    # Color background based on correlation strength
                    ax.set_facecolor(plt.cm.RdBu_r(0.5 + 0.5 * corr))
                    
                ax.set_xticks([])
                ax.set_yticks([])
    
    fig.suptitle('Click Parameter Analysis')
    plt.tight_layout()
    
    return fig, axes


def plot_whistle_contours(contours: Dict[str, pd.DataFrame],
                         time_range: Tuple[float, float] = None,
                         freq_range: Tuple[float, float] = None,
                         color_by: str = 'detection_id',
                         figsize: Tuple[float, float] = None,
                         theme: str = 'default') -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot whistle frequency contours over time.
    
    Args:
        contours: Dictionary mapping whistle IDs to contour DataFrames
        time_range: Time range to display (start_time, end_time) in seconds
        freq_range: Frequency range to display (min_freq, max_freq) in Hz
        color_by: How to color contours ('detection_id', 'amplitude', 'frequency')
        figsize: Figure size tuple (width, height)
        theme: Visual theme to use
        
    Returns:
        Tuple of (figure, axes) objects
    """
    plotter = DetectionPlotter(theme)
    fig, ax = plotter._setup_figure(figsize=figsize)
    
    if len(contours) == 0:
        ax.text(0.5, 0.5, 'No contour data available', 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title('No Whistle Contours')
        return fig, ax
    
    # Color setup
    colors = plt.cm.viridis(np.linspace(0, 1, len(contours)))
    
    # Plot each contour
    for i, (whistle_id, contour) in enumerate(contours.items()):
        if len(contour) == 0:
            continue
            
        time_data = contour['time'].values
        freq_data = contour['frequency'].values
        
        # Apply time range filter
        if time_range is not None:
            mask = (time_data >= time_range[0]) & (time_data <= time_range[1])
            time_data = time_data[mask]
            freq_data = freq_data[mask]
        
        # Apply frequency range filter
        if freq_range is not None:
            mask = (freq_data >= freq_range[0]) & (freq_data <= freq_range[1])
            time_data = time_data[mask]
            freq_data = freq_data[mask]
        
        if len(time_data) == 0:
            continue
        
        # Color by specified method
        if color_by == 'detection_id':
            color = colors[i]
            ax.plot(time_data, freq_data, color=color, linewidth=2, 
                   alpha=0.8, label=f'Whistle {whistle_id}')
        elif color_by == 'amplitude' and 'amplitude' in contour.columns:
            amp_data = contour['amplitude'].values
            if time_range is not None or freq_range is not None:
                # Apply same filtering to amplitude
                orig_mask = np.ones(len(contour), dtype=bool)
                if time_range is not None:
                    orig_mask &= (contour['time'] >= time_range[0]) & (contour['time'] <= time_range[1])
                if freq_range is not None:
                    orig_mask &= (contour['frequency'] >= freq_range[0]) & (contour['frequency'] <= freq_range[1])
                amp_data = amp_data[orig_mask]
            
            scatter = ax.scatter(time_data, freq_data, c=amp_data, 
                               cmap='plasma', s=20, alpha=0.8)
            if i == 0:  # Add colorbar once
                plotter._add_colorbar(scatter, ax, label='Amplitude (dB)')
        else:
            # Default coloring
            color = colors[i]
            ax.plot(time_data, freq_data, color=color, linewidth=2, 
                   alpha=0.8, label=f'Whistle {whistle_id}')
    
    # Format axes
    plotter._format_time_axis(ax)
    plotter._format_frequency_axis(ax, 
                                 np.max([np.max(c['frequency']) for c in contours.values() if len(c) > 0])
                                 if contours else None)
    
    # Add legend if not too many contours
    if len(contours) <= 10 and color_by == 'detection_id':
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    ax.grid(True, alpha=0.3)
    ax.set_title(f'Whistle Contours (n={len(contours)})')
    
    plt.tight_layout()
    return fig, ax


def plot_ici_analysis(detections: pd.DataFrame, 
                     max_ici: float = 2.0,
                     time_window: float = 60.0,
                     figsize: Tuple[float, float] = None,
                     theme: str = 'default') -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot inter-click interval (ICI) analysis.
    
    Args:
        detections: DataFrame with detection times
        max_ici: Maximum ICI to display in seconds
        time_window: Time window for rolling ICI analysis in seconds
        figsize: Figure size tuple (width, height)
        theme: Visual theme to use
        
    Returns:
        Tuple of (figure, axes array)
    """
    plotter = DetectionPlotter(theme)
    
    if 'UTC' not in detections.columns:
        raise ValueError("Detections DataFrame must have 'UTC' column for ICI analysis")
    
    if len(detections) < 2:
        raise ValueError("Need at least 2 detections for ICI analysis")
    
    # Calculate ICIs
    times = pd.to_datetime(detections['UTC'])
    time_seconds = (times - times.iloc[0]).dt.total_seconds().values
    ici_results = calculate_inter_click_intervals(time_seconds)
    
    if figsize is None:
        figsize = (12, 8)
    
    fig, axes = plotter._setup_subplots(nrows=2, ncols=2, figsize=figsize)
    
    # ICI histogram
    intervals = ici_results['intervals']
    valid_intervals = intervals[intervals <= max_ici]
    
    if len(valid_intervals) > 0:
        axes[0, 0].hist(valid_intervals, bins=50, alpha=0.7,
                       color=plotter.colors.detection_colors()['click'])
        axes[0, 0].axvline(ici_results['mean_ici'], color='red', linestyle='--',
                          linewidth=2, label=f"Mean: {ici_results['mean_ici']:.3f}s")
        axes[0, 0].axvline(ici_results['median_ici'], color='orange', linestyle='--',
                          linewidth=2, label=f"Median: {ici_results['median_ici']:.3f}s")
        axes[0, 0].legend()
    
    axes[0, 0].set_xlabel('Inter-Click Interval (s)')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('ICI Distribution')
    axes[0, 0].grid(True, alpha=0.3)
    
    # ICI over time
    if len(intervals) > 10:
        # Rolling window analysis
        window_samples = int(time_window / np.median(intervals))
        window_samples = max(5, min(window_samples, len(intervals) // 4))
        
        rolling_mean = pd.Series(intervals).rolling(window_samples, center=True).mean()
        rolling_std = pd.Series(intervals).rolling(window_samples, center=True).std()
        
        plot_times = time_seconds[1:]  # ICIs correspond to times between detections
        
        axes[0, 1].plot(plot_times, intervals, 'o', alpha=0.5, markersize=3, 
                       color=plotter.colors.detection_colors()['click'])
        axes[0, 1].plot(plot_times, rolling_mean, 'r-', linewidth=2, label='Rolling Mean')
        axes[0, 1].fill_between(plot_times, rolling_mean - rolling_std, 
                               rolling_mean + rolling_std, alpha=0.3, color='red',
                               label='±1 Std')
        axes[0, 1].legend()
    else:
        # Simple scatter plot
        axes[0, 1].plot(time_seconds[1:], intervals, 'o-', 
                       color=plotter.colors.detection_colors()['click'])
    
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Inter-Click Interval (s)')
    axes[0, 1].set_title('ICI Over Time')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Detection rate over time
    if len(time_seconds) > 10:
        # Calculate detection rate in sliding windows
        window_dur = 60  # 60 second windows
        max_time = time_seconds[-1]
        window_centers = np.arange(window_dur/2, max_time - window_dur/2, window_dur/4)
        
        detection_rates = []
        for center in window_centers:
            window_start = center - window_dur/2
            window_end = center + window_dur/2
            count = np.sum((time_seconds >= window_start) & (time_seconds <= window_end))
            detection_rates.append(count / window_dur)  # detections per second
        
        axes[1, 0].plot(window_centers, detection_rates, 'o-', 
                       color=plotter.colors.detection_colors()['click'])
        axes[1, 0].set_ylabel('Detection Rate (Hz)')
    else:
        axes[1, 0].text(0.5, 0.5, 'Insufficient data\nfor rate analysis', 
                       ha='center', va='center', transform=axes[1, 0].transAxes)
    
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_title('Detection Rate Over Time')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Summary statistics
    stats_text = "ICI Analysis Summary:\n\n"
    stats_text += f"Total intervals: {len(intervals)}\n"
    stats_text += f"Mean ICI: {ici_results['mean_ici']:.3f}s\n"
    stats_text += f"Median ICI: {ici_results['median_ici']:.3f}s\n"
    stats_text += f"Std ICI: {ici_results['std_ici']:.3f}s\n"
    stats_text += f"CV: {ici_results['ici_cv']:.3f}\n"
    stats_text += f"Regular clicks: {ici_results['regular_clicks']}\n\n"
    
    if len(intervals) > 0:
        stats_text += f"Min ICI: {np.min(intervals):.3f}s\n"
        stats_text += f"Max ICI: {np.max(intervals):.3f}s\n"
    
    axes[1, 1].text(0.05, 0.95, stats_text, transform=axes[1, 1].transAxes,
                   verticalalignment='top', fontfamily='monospace', fontsize=9)
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_xticks([])
    axes[1, 1].set_yticks([])
    axes[1, 1].set_title('ICI Statistics')
    
    fig.suptitle(f'Inter-Click Interval Analysis (n={len(detections)} detections)')
    plt.tight_layout()
    
    return fig, axes