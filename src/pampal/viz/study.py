"""
Study-level visualization module for PAMpal.

This module provides functions for visualizing data at the study level,
including overview plots, temporal patterns, spatial distributions,
and species comparison visualizations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Union, Tuple
import warnings
from datetime import datetime, timedelta

from .core import VisualizationBase, ColorSchemes


class StudyPlotter(VisualizationBase):
    """Specialized plotter for study-level visualizations."""
    
    def __init__(self, theme='default'):
        super().__init__(theme)
        self.colors = ColorSchemes()


def plot_study_overview(study_data: Dict[str, Any],
                       figsize: Tuple[float, float] = None,
                       theme: str = 'default') -> Tuple[plt.Figure, np.ndarray]:
    """
    Create a comprehensive overview plot of study data.
    
    Args:
        study_data: Dictionary containing study information
        figsize: Figure size tuple (width, height)
        theme: Visual theme to use
        
    Returns:
        Tuple of (figure, axes array)
    """
    plotter = StudyPlotter(theme)
    
    if figsize is None:
        figsize = (15, 12)
    
    fig, axes = plotter._setup_subplots(nrows=3, ncols=3, figsize=figsize)
    
    # Extract data components
    detections = study_data.get('detections', pd.DataFrame())
    survey_info = study_data.get('survey_info', {})
    effort_data = study_data.get('effort_data', pd.DataFrame())
    
    # 1. Detection counts by type
    if not detections.empty and 'detection_type' in detections.columns:
        type_counts = detections['detection_type'].value_counts()
        colors = [plotter.colors.detection_colors().get(t, '#1f77b4') for t in type_counts.index]
        
        axes[0, 0].bar(range(len(type_counts)), type_counts.values, color=colors)
        axes[0, 0].set_xticks(range(len(type_counts)))
        axes[0, 0].set_xticklabels(type_counts.index, rotation=45)
        axes[0, 0].set_ylabel('Detection Count')
        axes[0, 0].set_title('Detections by Type')
        axes[0, 0].grid(True, alpha=0.3)
    else:
        axes[0, 0].text(0.5, 0.5, 'No detection type data', 
                       ha='center', va='center', transform=axes[0, 0].transAxes)
        axes[0, 0].set_title('Detection Types Not Available')
    
    # 2. Detection rate over time
    if not detections.empty and 'UTC' in detections.columns:
        times = pd.to_datetime(detections['UTC'])
        
        # Create hourly bins
        time_bins = pd.date_range(start=times.min(), end=times.max(), freq='H')
        if len(time_bins) > 1:
            detection_counts = pd.cut(times, bins=time_bins).value_counts().sort_index()
            
            axes[0, 1].plot(time_bins[:-1], detection_counts.values, 'o-', alpha=0.7)
            axes[0, 1].set_xlabel('Time')
            axes[0, 1].set_ylabel('Detections per Hour')
            axes[0, 1].set_title('Detection Rate Over Time')
            axes[0, 1].tick_params(axis='x', rotation=45)
            axes[0, 1].grid(True, alpha=0.3)
        else:
            axes[0, 1].text(0.5, 0.5, 'Insufficient time range', 
                           ha='center', va='center', transform=axes[0, 1].transAxes)
    else:
        axes[0, 1].text(0.5, 0.5, 'No time data available', 
                       ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('Time Data Not Available')
    
    # 3. Frequency distribution
    if not detections.empty and 'peak_freq' in detections.columns:
        freq_data = detections['peak_freq'].dropna() / 1000  # Convert to kHz
        
        axes[0, 2].hist(freq_data, bins=30, alpha=0.7, edgecolor='black')
        axes[0, 2].set_xlabel('Peak Frequency (kHz)')
        axes[0, 2].set_ylabel('Count')
        axes[0, 2].set_title('Overall Frequency Distribution')
        axes[0, 2].grid(True, alpha=0.3)
    else:
        axes[0, 2].text(0.5, 0.5, 'No frequency data', 
                       ha='center', va='center', transform=axes[0, 2].transAxes)
        axes[0, 2].set_title('Frequency Data Not Available')
    
    # 4. Daily activity patterns
    if not detections.empty and 'UTC' in detections.columns:
        times = pd.to_datetime(detections['UTC'])
        hours = times.dt.hour
        
        hour_counts = hours.value_counts().sort_index()
        
        axes[1, 0].bar(hour_counts.index, hour_counts.values, alpha=0.7)
        axes[1, 0].set_xlabel('Hour of Day')
        axes[1, 0].set_ylabel('Detection Count')
        axes[1, 0].set_title('Daily Activity Pattern')
        axes[1, 0].set_xticks(range(0, 24, 4))
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, 'No time data available', 
                       ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Daily Pattern Not Available')
    
    # 5. Spatial distribution (if location data available)
    if not detections.empty and 'latitude' in detections.columns and 'longitude' in detections.columns:
        lats = detections['latitude'].dropna()
        lons = detections['longitude'].dropna()
        
        if len(lats) > 0 and len(lons) > 0:
            scatter = axes[1, 1].scatter(lons, lats, alpha=0.6, s=20)
            axes[1, 1].set_xlabel('Longitude')
            axes[1, 1].set_ylabel('Latitude')
            axes[1, 1].set_title('Spatial Distribution')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'No valid location data', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
    else:
        axes[1, 1].text(0.5, 0.5, 'No location data available', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Spatial Data Not Available')
    
    # 6. Detection duration distribution
    if not detections.empty and 'duration' in detections.columns:
        duration_data = detections['duration'].dropna() * 1000  # Convert to ms
        
        axes[1, 2].hist(duration_data, bins=30, alpha=0.7, edgecolor='black')
        axes[1, 2].set_xlabel('Duration (ms)')
        axes[1, 2].set_ylabel('Count')
        axes[1, 2].set_title('Detection Duration Distribution')
        axes[1, 2].grid(True, alpha=0.3)
    else:
        axes[1, 2].text(0.5, 0.5, 'No duration data', 
                       ha='center', va='center', transform=axes[1, 2].transAxes)
        axes[1, 2].set_title('Duration Data Not Available')
    
    # 7. Effort vs detections
    if effort_data and not detections.empty:
        # This is a simplified example - would need proper effort calculation
        effort_text = "Effort Analysis:\n\n"
        effort_text += f"Total recording time: {effort_data.get('total_hours', 'N/A')} hours\n"
        effort_text += f"Total detections: {len(detections)}\n"
        if 'total_hours' in effort_data and effort_data['total_hours'] > 0:
            det_rate = len(detections) / effort_data['total_hours']
            effort_text += f"Detection rate: {det_rate:.2f} det/hour\n"
        
        axes[2, 0].text(0.05, 0.95, effort_text, transform=axes[2, 0].transAxes,
                       verticalalignment='top', fontfamily='monospace', fontsize=10)
        axes[2, 0].set_xlim(0, 1)
        axes[2, 0].set_ylim(0, 1)
        axes[2, 0].set_xticks([])
        axes[2, 0].set_yticks([])
        axes[2, 0].set_title('Effort Summary')
    else:
        axes[2, 0].text(0.5, 0.5, 'No effort data available', 
                       ha='center', va='center', transform=axes[2, 0].transAxes)
        axes[2, 0].set_title('Effort Data Not Available')
    
    # 8. Amplitude distribution
    amp_col = None
    for col in ['amplitude', 'peak_amplitude', 'rms_amplitude']:
        if not detections.empty and col in detections.columns:
            amp_col = col
            break
    
    if amp_col:
        amp_data = detections[amp_col].dropna()
        if len(amp_data) > 0:
            axes[2, 1].hist(amp_data, bins=30, alpha=0.7, edgecolor='black')
            axes[2, 1].set_xlabel(f'{amp_col.replace("_", " ").title()}')
            axes[2, 1].set_ylabel('Count')
            axes[2, 1].set_title('Amplitude Distribution')
            axes[2, 1].grid(True, alpha=0.3)
        else:
            axes[2, 1].text(0.5, 0.5, 'No amplitude data', 
                           ha='center', va='center', transform=axes[2, 1].transAxes)
    else:
        axes[2, 1].text(0.5, 0.5, 'No amplitude data available', 
                       ha='center', va='center', transform=axes[2, 1].transAxes)
        axes[2, 1].set_title('Amplitude Data Not Available')
    
    # 9. Study summary statistics
    summary_text = "Study Summary:\n\n"
    summary_text += f"Study name: {survey_info.get('name', 'Unknown')}\n"
    summary_text += f"Location: {survey_info.get('location', 'Unknown')}\n"
    
    if not detections.empty:
        summary_text += f"Total detections: {len(detections)}\n"
        
        if 'UTC' in detections.columns:
            times = pd.to_datetime(detections['UTC'])
            start_time = times.min()
            end_time = times.max()
            duration = end_time - start_time
            summary_text += f"Survey duration: {duration.days} days\n"
            summary_text += f"Start: {start_time.strftime('%Y-%m-%d %H:%M')}\n"
            summary_text += f"End: {end_time.strftime('%Y-%m-%d %H:%M')}\n"
        
        if 'peak_freq' in detections.columns:
            freq_data = detections['peak_freq'].dropna()
            if len(freq_data) > 0:
                summary_text += f"Freq range: {freq_data.min()/1000:.1f}-{freq_data.max()/1000:.1f} kHz\n"
    
    axes[2, 2].text(0.05, 0.95, summary_text, transform=axes[2, 2].transAxes,
                   verticalalignment='top', fontfamily='monospace', fontsize=9)
    axes[2, 2].set_xlim(0, 1)
    axes[2, 2].set_ylim(0, 1)
    axes[2, 2].set_xticks([])
    axes[2, 2].set_yticks([])
    axes[2, 2].set_title('Study Statistics')
    
    fig.suptitle(f'PAMpal Study Overview: {survey_info.get("name", "Unknown Study")}')
    plt.tight_layout()
    
    return fig, axes


def plot_temporal_patterns(detections: pd.DataFrame,
                          time_grouping: str = 'day',
                          detection_types: List[str] = None,
                          figsize: Tuple[float, float] = None,
                          theme: str = 'default') -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot temporal patterns in detection data.
    
    Args:
        detections: DataFrame with detection data
        time_grouping: Time grouping ('hour', 'day', 'week', 'month')
        detection_types: List of detection types to include
        figsize: Figure size tuple (width, height)
        theme: Visual theme to use
        
    Returns:
        Tuple of (figure, axes array)
    """
    plotter = StudyPlotter(theme)
    
    if 'UTC' not in detections.columns:
        raise ValueError("Detections DataFrame must have 'UTC' column")
    
    if figsize is None:
        figsize = (15, 10)
    
    fig, axes = plotter._setup_subplots(nrows=2, ncols=2, figsize=figsize)
    
    times = pd.to_datetime(detections['UTC'])
    
    # Filter detection types if specified
    if detection_types and 'detection_type' in detections.columns:
        mask = detections['detection_type'].isin(detection_types)
        detections = detections[mask]
        times = times[mask]
    
    # 1. Time series plot
    freq_map = {'hour': 'H', 'day': 'D', 'week': 'W', 'month': 'M'}
    freq = freq_map.get(time_grouping, 'D')
    
    time_bins = pd.date_range(start=times.min(), end=times.max(), freq=freq)
    if len(time_bins) > 1:
        detection_counts = pd.cut(times, bins=time_bins).value_counts().sort_index()
        
        axes[0, 0].plot(time_bins[:-1], detection_counts.values, 'o-', alpha=0.7)
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel(f'Detections per {time_grouping.title()}')
        axes[0, 0].set_title(f'Detection Rate Over Time ({time_grouping.title()}ly)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Hourly patterns
    hours = times.dt.hour
    hour_counts = hours.value_counts().sort_index()
    
    axes[0, 1].bar(hour_counts.index, hour_counts.values, alpha=0.7)
    axes[0, 1].set_xlabel('Hour of Day')
    axes[0, 1].set_ylabel('Detection Count')
    axes[0, 1].set_title('Hourly Activity Pattern')
    axes[0, 1].set_xticks(range(0, 24, 4))
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Day of week patterns
    weekdays = times.dt.day_name()
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekday_counts = weekdays.value_counts().reindex(weekday_order, fill_value=0)
    
    axes[1, 0].bar(range(len(weekday_counts)), weekday_counts.values, alpha=0.7)
    axes[1, 0].set_xticks(range(len(weekday_counts)))
    axes[1, 0].set_xticklabels([day[:3] for day in weekday_order])
    axes[1, 0].set_ylabel('Detection Count')
    axes[1, 0].set_title('Weekly Activity Pattern')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Seasonal patterns (if data spans multiple months)
    months = times.dt.month
    month_counts = months.value_counts().sort_index()
    
    if len(month_counts) > 1:
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        axes[1, 1].bar(month_counts.index, month_counts.values, alpha=0.7)
        axes[1, 1].set_xticks(month_counts.index)
        axes[1, 1].set_xticklabels([month_names[i-1] for i in month_counts.index])
        axes[1, 1].set_ylabel('Detection Count')
        axes[1, 1].set_title('Seasonal Activity Pattern')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'Insufficient data\nfor seasonal analysis', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Seasonal Pattern Not Available')
    
    fig.suptitle(f'Temporal Patterns Analysis (n={len(detections)} detections)')
    plt.tight_layout()
    
    return fig, axes


def plot_spatial_distribution(detections: pd.DataFrame,
                             detection_types: List[str] = None,
                             map_background: bool = False,
                             figsize: Tuple[float, float] = None,
                             theme: str = 'default') -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot spatial distribution of detections.
    
    Args:
        detections: DataFrame with detection data including lat/lon
        detection_types: List of detection types to include
        map_background: Whether to include map background (requires cartopy)
        figsize: Figure size tuple (width, height)
        theme: Visual theme to use
        
    Returns:
        Tuple of (figure, axes) objects
    """
    plotter = StudyPlotter(theme)
    
    required_cols = ['latitude', 'longitude']
    missing_cols = [col for col in required_cols if col not in detections.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Filter out NaN coordinates
    valid_coords = detections.dropna(subset=['latitude', 'longitude'])
    
    if len(valid_coords) == 0:
        raise ValueError("No valid coordinate data found")
    
    # Filter detection types if specified
    if detection_types and 'detection_type' in valid_coords.columns:
        valid_coords = valid_coords[valid_coords['detection_type'].isin(detection_types)]
    
    fig, ax = plotter._setup_figure(figsize=figsize)
    
    lats = valid_coords['latitude']
    lons = valid_coords['longitude']
    
    # Color by detection type if available
    if 'detection_type' in valid_coords.columns:
        detection_colors = plotter.colors.detection_colors()
        
        for det_type in valid_coords['detection_type'].unique():
            type_data = valid_coords[valid_coords['detection_type'] == det_type]
            color = detection_colors.get(det_type, detection_colors['unknown'])
            
            ax.scatter(type_data['longitude'], type_data['latitude'], 
                      c=color, alpha=0.7, s=30, label=det_type.title())
        
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        ax.scatter(lons, lats, alpha=0.7, s=30)
    
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(f'Spatial Distribution (n={len(valid_coords)} detections)')
    ax.grid(True, alpha=0.3)
    
    # Set equal aspect ratio for accurate geographic representation
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    return fig, ax


def plot_species_comparison(detections_by_species: Dict[str, pd.DataFrame],
                           parameters: List[str] = None,
                           figsize: Tuple[float, float] = None,
                           theme: str = 'default') -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot comparison of parameters across different species or detection types.
    
    Args:
        detections_by_species: Dictionary mapping species names to detection DataFrames
        parameters: List of parameters to compare
        figsize: Figure size tuple (width, height)
        theme: Visual theme to use
        
    Returns:
        Tuple of (figure, axes) objects
    """
    plotter = StudyPlotter(theme)
    
    if not detections_by_species:
        raise ValueError("No species data provided")
    
    # Default parameters to compare
    if parameters is None:
        all_params = set()
        for df in detections_by_species.values():
            all_params.update(df.columns)
        
        common_params = ['peak_freq', 'duration', 'amplitude', 'bandwidth', 'centroid_freq']
        parameters = [p for p in common_params if p in all_params]
    
    if not parameters:
        raise ValueError("No valid parameters found for comparison")
    
    n_params = len(parameters)
    n_species = len(detections_by_species)
    
    if figsize is None:
        figsize = (4 * n_params, 3 * ((n_params + 1) // 2))
    
    fig, axes = plotter._setup_subplots(nrows=2, ncols=(n_params + 1) // 2, figsize=figsize)
    
    # Ensure axes is always 2D for consistent indexing
    if n_params == 1:
        axes = np.array([[axes]])
    elif axes.ndim == 1:
        axes = axes.reshape(1, -1)
    
    species_colors = plotter.colors.species_colors()
    color_list = list(species_colors.values())
    
    for i, param in enumerate(parameters):
        row = i // ((n_params + 1) // 2)
        col = i % ((n_params + 1) // 2)
        
        if row < axes.shape[0] and col < axes.shape[1]:
            ax = axes[row, col]
            
            # Collect data for all species
            species_data = []
            species_names = []
            
            for j, (species, detections) in enumerate(detections_by_species.items()):
                if param in detections.columns:
                    data = detections[param].dropna()
                    if len(data) > 0:
                        species_data.append(data)
                        species_names.append(species)
            
            if species_data:
                # Create box plot
                bp = ax.boxplot(species_data, labels=species_names, patch_artist=True)
                
                # Color the boxes
                for patch, species in zip(bp['boxes'], species_names):
                    color_idx = list(detections_by_species.keys()).index(species) % len(color_list)
                    patch.set_facecolor(color_list[color_idx])
                    patch.set_alpha(0.7)
                
                ax.set_ylabel(param.replace('_', ' ').title())
                ax.set_title(f'{param.replace("_", " ").title()} Comparison')
                ax.tick_params(axis='x', rotation=45)
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, f'No {param} data available', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{param.replace("_", " ").title()} Not Available')
    
    # Hide empty subplots
    for i in range(n_params, axes.shape[0] * axes.shape[1]):
        row = i // axes.shape[1]
        col = i % axes.shape[1]
        if row < axes.shape[0] and col < axes.shape[1]:
            axes[row, col].set_visible(False)
    
    fig.suptitle(f'Species Parameter Comparison (n={n_species} species)')
    plt.tight_layout()
    
    return fig, axes