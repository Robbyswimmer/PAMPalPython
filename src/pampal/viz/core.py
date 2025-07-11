"""
Core visualization infrastructure for PAMpal.

This module provides the foundational classes and utilities for creating 
high-quality, consistent visualizations across the PAMpal package.
"""

import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
import warnings
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import os


class ColorSchemes:
    """
    Scientific color schemes and palettes for acoustic data visualization.
    
    Provides accessibility-friendly, perceptually uniform color schemes
    optimized for acoustic analysis and publication-quality figures.
    """
    
    def __init__(self):
        """Initialize color schemes."""
        self._register_custom_colormaps()
    
    @staticmethod
    def acoustic_viridis():
        """Viridis colormap optimized for acoustic spectrograms."""
        return plt.cm.viridis
    
    @staticmethod  
    def acoustic_plasma():
        """Plasma colormap for high-contrast acoustic data."""
        return plt.cm.plasma
    
    @staticmethod
    def acoustic_cividis():
        """Colorblind-friendly colormap for acoustic analysis."""
        return plt.cm.cividis
    
    @staticmethod
    def acoustic_grayscale():
        """Traditional grayscale for publication figures."""
        return plt.cm.gray
    
    @staticmethod
    def detection_colors():
        """Standard colors for different detection types."""
        return {
            'click': '#1f77b4',      # Blue
            'whistle': '#ff7f0e',    # Orange  
            'moan': '#2ca02c',       # Green
            'cepstrum': '#d62728',   # Red
            'gpl': '#9467bd',        # Purple
            'noise': '#8c564b',      # Brown
            'unknown': '#7f7f7f'     # Gray
        }
            
    @staticmethod
    def scientific_palette(name):
        """Get a scientific color palette by name.
        
        Args:
            name: Name of the palette ('viridis', 'plasma', 'cividis', etc.)
            
        Returns:
            List of colors in the palette
        """
        if hasattr(plt.cm, name):
            cmap = getattr(plt.cm, name)
            return [cmap(i) for i in np.linspace(0, 1, 256)]  
        else:
            raise ValueError(f"Unknown palette: {name}")
    
    @staticmethod
    def species_colors():
        """Color palette for different marine mammal species."""
        return {
            'sperm_whale': '#1f77b4',
            'beaked_whale': '#ff7f0e', 
            'dolphin': '#2ca02c',
            'porpoise': '#d62728',
            'pilot_whale': '#9467bd',
            'killer_whale': '#8c564b',
            'unknown': '#7f7f7f'
        }
    
    @staticmethod
    def quality_colors():
        """Colors for data quality indicators."""
        return {
            'excellent': '#2ca02c',   # Green
            'good': '#ff7f0e',        # Orange
            'poor': '#d62728',        # Red
            'unknown': '#7f7f7f'      # Gray
        }
    
    def primary_color(self):
        """Get the primary color for the current color scheme."""
        return '#1f77b4'  # Default matplotlib blue
    
    def color_cycle(self):
        """Get a color cycle for multiple plots."""
        return ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    def _register_custom_colormaps(self):
        """Register custom colormaps for acoustic analysis."""
        # Create custom acoustic colormap
        colors = ['#000033', '#000055', '#0000aa', '#0055ff', '#55aaff', 
                 '#aaffff', '#ffff55', '#ffaa00', '#ff5500', '#aa0000']
        acoustic_cmap = LinearSegmentedColormap.from_list('acoustic', colors)
        
        # Check if the colormap already exists before registering
        if 'acoustic' not in mpl.colormaps:
            mpl.colormaps.register(name='acoustic', cmap=acoustic_cmap)


class PampalTheme:
    """
    Publication-quality theme for PAMpal visualizations.
    
    Provides consistent styling across all PAMpal plots with options
    for different output formats (screen, print, presentation).
    """
    
    def __init__(self, style='default'):
        """
        Initialize PAMpal theme.
        
        Args:
            style: Theme style ('default', 'publication', 'presentation', 'minimal')
        """
        self.style = style
        self.name = style  # Add name attribute to match test expectations
        self._setup_theme()
        
    def get_params(self):
        """Get the theme parameters.
        
        Returns:
            Dictionary of theme parameters
        """
        return self.params
    
    def _setup_theme(self):
        """Set up the visual theme parameters."""
        if self.style == 'publication':
            self.params = {
                'figure.figsize': (8, 6),
                'figure.dpi': 300,
                'font.size': 10,
                'axes.titlesize': 12,
                'axes.labelsize': 10,
                'xtick.labelsize': 8,
                'ytick.labelsize': 8,
                'legend.fontsize': 9,
                'font.family': 'serif',
                'axes.spines.top': False,
                'axes.spines.right': False,
                'axes.grid': True,
                'grid.alpha': 0.3
            }
        elif self.style == 'presentation':
            self.params = {
                'figure.figsize': (12, 8),
                'figure.dpi': 150,
                'font.size': 14,
                'axes.titlesize': 18,
                'axes.labelsize': 16,
                'xtick.labelsize': 12,
                'ytick.labelsize': 12,
                'legend.fontsize': 14,
                'font.family': 'sans-serif',
                'axes.spines.top': False,
                'axes.spines.right': False,
                'axes.grid': True,
                'grid.alpha': 0.3
            }
        elif self.style == 'minimal':
            self.params = {
                'figure.figsize': (8, 6),
                'figure.dpi': 150,
                'font.size': 10,
                'axes.spines.top': False,
                'axes.spines.right': False,
                'axes.spines.left': False,
                'axes.spines.bottom': False,
                'axes.grid': False,
                'xtick.bottom': False,
                'ytick.left': False
            }
        else:  # default
            self.params = {
                'figure.figsize': (10, 6),
                'figure.dpi': 150,
                'font.size': 11,
                'axes.titlesize': 14,
                'axes.labelsize': 12,
                'xtick.labelsize': 10,
                'ytick.labelsize': 10,
                'legend.fontsize': 10,
                'axes.spines.top': False,
                'axes.spines.right': False,
                'axes.grid': True,
                'grid.alpha': 0.3
            }
    
    def apply(self):
        """Apply the theme to matplotlib."""
        plt.rcParams.update(self.params)
    
    def context(self):
        """Return theme as context manager."""
        return plt.rc_context(self.params)


class VisualizationBase:
    """
    Base class for all PAMpal visualizations.
    
    Provides common functionality for plot creation, styling, and management.
    """
    
    def __init__(self, theme='default', color_scheme='viridis'):
        """
        Initialize visualization base.
        
        Args:
            theme: Visual theme to use
            color_scheme: Default color scheme for plots
        """
        self.theme = PampalTheme(theme)
        self.colors = ColorSchemes()
        self.color_scheme = color_scheme
        self._current_figures = []
    
    def _setup_figure(self, figsize=None, **kwargs):
        """
        Set up a new figure with PAMpal styling.
        
        Args:
            figsize: Figure size tuple (width, height)
            **kwargs: Additional figure parameters
            
        Returns:
            Tuple of (figure, axes)
        """
        with self.theme.context():
            if figsize is None:
                figsize = plt.rcParams['figure.figsize']
            
            fig, ax = plt.subplots(figsize=figsize, **kwargs)
            self._current_figures.append(fig)
            return fig, ax
    
    def _setup_subplots(self, nrows=1, ncols=1, figsize=None, **kwargs):
        """
        Set up subplots with PAMpal styling.
        
        Args:
            nrows: Number of rows
            ncols: Number of columns  
            figsize: Figure size tuple
            **kwargs: Additional subplot parameters
            
        Returns:
            Tuple of (figure, axes)
        """
        with self.theme.context():
            if figsize is None:
                figsize = plt.rcParams['figure.figsize']
                figsize = (figsize[0] * ncols, figsize[1] * nrows)
            
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, 
                                   figsize=figsize, **kwargs)
            self._current_figures.append(fig)
            return fig, axes
    
    def _format_frequency_axis(self, ax, max_freq=None, log_scale=False):
        """Format frequency axis with appropriate labels and scales."""
        if log_scale:
            ax.set_yscale('log')
            ax.set_ylabel('Frequency (Hz, log scale)')
        else:
            ax.set_ylabel('Frequency (Hz)')
        
        if max_freq is not None:
            if max_freq > 1000:
                # Use kHz for high frequencies
                ax.set_ylabel('Frequency (kHz)')
                formatter = plt.FuncFormatter(lambda x, p: f'{x/1000:.0f}')
                ax.yaxis.set_major_formatter(formatter)
    
    def _format_time_axis(self, ax, duration=None):
        """Format time axis with appropriate labels."""
        ax.set_xlabel('Time (s)')
        
        if duration is not None:
            if duration < 1:
                # Use milliseconds for short durations
                ax.set_xlabel('Time (ms)')
                formatter = plt.FuncFormatter(lambda x, p: f'{x*1000:.0f}')
                ax.xaxis.set_major_formatter(formatter)
            elif duration > 3600:
                # Use hours for long durations
                ax.set_xlabel('Time (hours)')
                formatter = plt.FuncFormatter(lambda x, p: f'{x/3600:.1f}')
                ax.xaxis.set_major_formatter(formatter)
    
    def _add_colorbar(self, mappable, ax, label='', **kwargs):
        """Add a properly formatted colorbar."""
        cbar = plt.colorbar(mappable, ax=ax, **kwargs)
        cbar.set_label(label, rotation=270, labelpad=20)
        return cbar
    
    def save_figure(self, filename, dpi=300, format='png', **kwargs):
        """
        Save the current figure with high quality settings.
        
        Args:
            filename: Output filename
            dpi: Resolution in dots per inch
            format: Output format ('png', 'pdf', 'svg', 'eps')
            **kwargs: Additional save parameters
        """
        if not self._current_figures:
            warnings.warn("No active figures to save")
            return
        
        current_fig = self._current_figures[-1]
        current_fig.savefig(filename, dpi=dpi, format=format, 
                           bbox_inches='tight', **kwargs)
    
    def close_figures(self):
        """Close all figures created by this visualization instance."""
        for fig in self._current_figures:
            plt.close(fig)
        self._current_figures.clear()


class PlotManager:
    """
    Manager class for handling multiple plots and complex visualizations.
    
    Provides utilities for creating multi-panel figures, managing plot
    layouts, and coordinating between different visualization types.
    """
    
    def __init__(self, theme='default'):
        """
        Initialize plot manager.
        
        Args:
            theme: Default theme for all managed plots
        """
        self.theme = theme
        self.figures = {}
        self.current_figure = None
    
    def create_figure(self, name, figsize=(12, 8), **kwargs):
        """
        Create a new named figure.
        
        Args:
            name: Figure identifier
            figsize: Figure size tuple
            **kwargs: Additional figure parameters
            
        Returns:
            Figure object
        """
        fig = plt.figure(figsize=figsize, **kwargs)
        self.figures[name] = fig
        self.current_figure = name
        return fig
    
    def create_subplot_figure(self, name, layout, figsize=None, **kwargs):
        """
        Create a figure with subplot layout.
        
        Args:
            name: Figure identifier
            layout: Tuple of (nrows, ncols) or list of subplot specifications
            figsize: Figure size tuple
            **kwargs: Additional subplot parameters
            
        Returns:
            Tuple of (figure, axes)
        """
        if isinstance(layout, tuple):
            nrows, ncols = layout
            if figsize is None:
                figsize = (4 * ncols, 3 * nrows)
            
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, 
                                   figsize=figsize, **kwargs)
        else:
            # Complex layout using subplot_mosaic
            fig, axes = plt.subplot_mosaic(layout, figsize=figsize, **kwargs)
        
        self.figures[name] = fig
        self.current_figure = name
        return fig, axes
    
    def get_figure(self, name=None):
        """Get a figure by name or return current figure."""
        if name is None:
            name = self.current_figure
        return self.figures.get(name)
    
    def save_all_figures(self, directory, prefix='pampal', **kwargs):
        """
        Save all managed figures to a directory.
        
        Args:
            directory: Output directory
            prefix: Filename prefix
            **kwargs: Additional save parameters
        """
        os.makedirs(directory, exist_ok=True)
        
        for name, fig in self.figures.items():
            filename = os.path.join(directory, f"{prefix}_{name}.png")
            fig.savefig(filename, bbox_inches='tight', **kwargs)
    
    def close_all(self):
        """Close all managed figures."""
        for fig in self.figures.values():
            plt.close(fig)
        self.figures.clear()
        self.current_figure = None


# Module-level convenience functions
def set_style(style='default'):
    """
    Set the global PAMpal plotting style.
    
    Args:
        style: Style name ('default', 'publication', 'presentation', 'minimal')
    """
    theme = PampalTheme(style)
    theme.apply()


def reset_style():
    """Reset matplotlib to default styling."""
    plt.rcdefaults()


# Initialize color schemes on module import
_color_schemes = ColorSchemes()

# Set default style
set_style('default')