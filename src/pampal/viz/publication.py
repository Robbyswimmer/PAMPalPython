"""
Publication-quality plotting tools for PAMpal.

This module provides tools for creating publication-ready figures,
including custom themes, multi-panel layouts, statistical plots,
and figure export utilities.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
import warnings
from pathlib import Path

from .core import VisualizationBase, ColorSchemes


class PublicationTheme:
    """Publication-ready plotting theme manager."""
    
    def __init__(self, style: str = 'nature'):
        self.style = style
        self.setup_theme()
    
    def setup_theme(self):
        """Set up matplotlib parameters for publication quality."""
        # Base parameters for all styles
        base_params = {
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
            'font.size': 8,
            'axes.linewidth': 0.5,
            'axes.spines.left': True,
            'axes.spines.bottom': True,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'xtick.major.size': 3,
            'xtick.minor.size': 1.5,
            'ytick.major.size': 3,
            'ytick.minor.size': 1.5,
            'xtick.major.width': 0.5,
            'xtick.minor.width': 0.5,
            'ytick.major.width': 0.5,
            'ytick.minor.width': 0.5,
            'legend.frameon': False,
            'legend.fontsize': 7,
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.05
        }
        
        # Style-specific parameters
        if self.style == 'nature':
            style_params = {
                'figure.figsize': (3.5, 2.5),  # Single column
                'font.size': 7,
                'axes.labelsize': 7,
                'xtick.labelsize': 6,
                'ytick.labelsize': 6,
                'legend.fontsize': 6
            }
        elif self.style == 'science':
            style_params = {
                'figure.figsize': (3.3, 2.5),
                'font.size': 8,
                'axes.labelsize': 8,
                'xtick.labelsize': 7,
                'ytick.labelsize': 7,
                'legend.fontsize': 7
            }
        elif self.style == 'plos':
            style_params = {
                'figure.figsize': (4.0, 3.0),
                'font.size': 9,
                'axes.labelsize': 9,
                'xtick.labelsize': 8,
                'ytick.labelsize': 8,
                'legend.fontsize': 8
            }
        else:  # default
            style_params = {
                'figure.figsize': (4.0, 3.0),
                'font.size': 8
            }
        
        # Combine parameters
        all_params = {**base_params, **style_params}
        
        # Apply parameters
        plt.rcParams.update(all_params)
    
    def get_colors(self) -> Dict[str, str]:
        """Get publication-appropriate color palette."""
        if self.style in ['nature', 'science']:
            # Conservative scientific palette
            return {
                'primary': '#1f77b4',
                'secondary': '#ff7f0e', 
                'tertiary': '#2ca02c',
                'quaternary': '#d62728',
                'accent': '#9467bd'
            }
        else:
            # More colorful palette for other journals
            return {
                'primary': '#2E86C1',
                'secondary': '#E74C3C',
                'tertiary': '#28B463',
                'quaternary': '#F39C12',
                'accent': '#8E44AD'
            }


class MultipanelFigure:
    """Manager for creating complex multi-panel publication figures."""
    
    def __init__(self, layout: Union[Tuple[int, int], Dict[str, Any]], 
                 figsize: Tuple[float, float] = None,
                 theme: str = 'nature'):
        self.theme = PublicationTheme(theme)
        self.colors = self.theme.get_colors()
        
        if isinstance(layout, tuple):
            self.rows, self.cols = layout
            self.fig, self.axes = plt.subplots(self.rows, self.cols, figsize=figsize)
        else:
            # Custom layout using GridSpec
            self.fig = plt.figure(figsize=figsize)
            self.gs = GridSpec(**layout)
            self.axes = {}
        
        # Ensure axes is always 2D for consistent indexing
        if hasattr(self, 'axes') and isinstance(self.axes, np.ndarray):
            if self.axes.ndim == 1:
                if len(layout) == 2 and layout[0] == 1:
                    self.axes = self.axes.reshape(1, -1)
                else:
                    self.axes = self.axes.reshape(-1, 1)
    
    def add_subplot(self, position: Union[Tuple[int, int], str], **kwargs):
        """Add a subplot at specified position."""
        if isinstance(position, tuple):
            row, col = position
            if hasattr(self, 'axes'):
                return self.axes[row, col] if self.axes.ndim == 2 else self.axes[row]
        else:
            ax = self.fig.add_subplot(self.gs[position], **kwargs)
            self.axes[position] = ax
            return ax
    
    def add_panel_label(self, ax, label: str, position: str = 'top_left',
                       fontsize: int = 10, fontweight: str = 'bold'):
        """Add panel label (A, B, C, etc.) to subplot."""
        positions = {
            'top_left': (-0.15, 1.05),
            'top_right': (1.05, 1.05),
            'bottom_left': (-0.15, -0.15),
            'bottom_right': (1.05, -0.15)
        }
        
        x, y = positions.get(position, positions['top_left'])
        ax.text(x, y, label, transform=ax.transAxes, fontsize=fontsize,
                fontweight=fontweight, va='bottom', ha='left')
    
    def save_figure(self, filename: str, **kwargs):
        """Save the multi-panel figure."""
        plt.tight_layout()
        self.fig.savefig(filename, **kwargs)


def create_publication_spectrogram(waveform: np.ndarray, sample_rate: int,
                                  detections: pd.DataFrame = None,
                                  figsize: Tuple[float, float] = None,
                                  title: str = None,
                                  theme: str = 'nature') -> Tuple[plt.Figure, plt.Axes]:
    """
    Create a publication-quality spectrogram with optional detection overlays.
    
    Args:
        waveform: 1D array of audio samples
        sample_rate: Sample rate in Hz
        detections: Optional DataFrame with detection data
        figsize: Figure size tuple
        title: Plot title
        theme: Publication theme to use
        
    Returns:
        Tuple of (figure, axes) objects
    """
    pub_theme = PublicationTheme(theme)
    
    from .spectrograms import plot_spectrogram_with_detections
    
    fig, ax = plot_spectrogram_with_detections(
        waveform, detections or pd.DataFrame(), sample_rate,
        figsize=figsize, theme='publication'
    )
    
    # Apply publication formatting
    ax.tick_params(direction='in', which='both')
    
    if title:
        ax.set_title(title, fontsize=plt.rcParams['font.size'] + 1, pad=10)
    
    return fig, ax


def create_parameter_comparison_figure(data_dict: Dict[str, pd.DataFrame],
                                     parameters: List[str],
                                     figsize: Tuple[float, float] = None,
                                     theme: str = 'nature') -> Tuple[plt.Figure, np.ndarray]:
    """
    Create publication-quality parameter comparison figure.
    
    Args:
        data_dict: Dictionary mapping group names to DataFrames
        parameters: List of parameters to compare
        figsize: Figure size tuple
        theme: Publication theme to use
        
    Returns:
        Tuple of (figure, axes array)
    """
    pub_theme = PublicationTheme(theme)
    colors = pub_theme.get_colors()
    
    n_params = len(parameters)
    fig, axes = plt.subplots(1, n_params, figsize=figsize or (3 * n_params, 3))
    
    if n_params == 1:
        axes = [axes]
    
    color_list = list(colors.values())
    
    for i, param in enumerate(parameters):
        ax = axes[i]
        
        # Collect data for box plot
        data_list = []
        labels = []
        
        for j, (group_name, df) in enumerate(data_dict.items()):
            if param in df.columns:
                param_data = df[param].dropna()
                if len(param_data) > 0:
                    data_list.append(param_data)
                    labels.append(group_name)
        
        if data_list:
            # Create box plot
            bp = ax.boxplot(data_list, labels=labels, patch_artist=True,
                           boxprops=dict(linewidth=0.5),
                           whiskerprops=dict(linewidth=0.5),
                           capprops=dict(linewidth=0.5),
                           medianprops=dict(linewidth=0.5, color='black'))
            
            # Color the boxes
            for patch, j in zip(bp['boxes'], range(len(data_list))):
                patch.set_facecolor(color_list[j % len(color_list)])
                patch.set_alpha(0.7)
            
            # Statistical significance testing
            if len(data_list) == 2:
                from scipy import stats
                try:
                    stat, p_value = stats.mannwhitneyu(data_list[0], data_list[1])
                    if p_value < 0.001:
                        sig_text = '***'
                    elif p_value < 0.01:
                        sig_text = '**'
                    elif p_value < 0.05:
                        sig_text = '*'
                    else:
                        sig_text = 'ns'
                    
                    # Add significance annotation
                    y_max = max([max(d) for d in data_list])
                    y_pos = y_max * 1.1
                    ax.plot([1, 2], [y_pos, y_pos], 'k-', linewidth=0.5)
                    ax.text(1.5, y_pos * 1.05, sig_text, ha='center', va='bottom',
                           fontsize=plt.rcParams['font.size'])
                except Exception as e:
                    warnings.warn(f"Statistical test failed: {e}")
        
        ax.set_ylabel(param.replace('_', ' ').title())
        ax.tick_params(direction='in', which='both')
        ax.grid(True, alpha=0.3, linewidth=0.3)
        
        # Rotate x-axis labels if needed
        if max(len(label) for label in labels) > 8:
            ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    return fig, axes


def create_detection_summary_figure(detections: pd.DataFrame,
                                   waveform_examples: Dict[str, np.ndarray] = None,
                                   sample_rate: int = 192000,
                                   figsize: Tuple[float, float] = None,
                                   theme: str = 'nature') -> Tuple[plt.Figure, Dict[str, plt.Axes]]:
    """
    Create a comprehensive detection summary figure for publication.
    
    Args:
        detections: DataFrame with detection data
        waveform_examples: Dictionary of example waveforms
        sample_rate: Sample rate for waveforms
        figsize: Figure size tuple
        theme: Publication theme to use
        
    Returns:
        Tuple of (figure, axes dictionary)
    """
    pub_theme = PublicationTheme(theme)
    
    # Create custom layout
    layout = {
        'nrows': 3,
        'ncols': 4,
        'width_ratios': [1, 1, 1, 1],
        'height_ratios': [1, 1, 1],
        'hspace': 0.3,
        'wspace': 0.3
    }
    
    multi_fig = MultipanelFigure(layout, figsize=figsize)
    axes = {}
    
    # Panel A: Example waveforms
    if waveform_examples:
        ax_waves = multi_fig.fig.add_subplot(multi_fig.gs[0, :2])
        axes['waveforms'] = ax_waves
        
        for i, (example_id, waveform) in enumerate(waveform_examples.items()):
            time = np.arange(len(waveform)) / sample_rate
            offset = i * 0.5  # Vertical offset
            ax_waves.plot(time, waveform + offset, linewidth=0.5, 
                         color=list(pub_theme.get_colors().values())[i % 5],
                         label=f'Example {i+1}')
        
        ax_waves.set_xlabel('Time (s)')
        ax_waves.set_ylabel('Amplitude + Offset')
        ax_waves.legend(fontsize=6)
        ax_waves.tick_params(direction='in', which='both')
        multi_fig.add_panel_label(ax_waves, 'A')
    
    # Panel B: Frequency distribution
    if 'peak_freq' in detections.columns:
        ax_freq = multi_fig.fig.add_subplot(multi_fig.gs[0, 2:])
        axes['frequency'] = ax_freq
        
        freq_data = detections['peak_freq'].dropna() / 1000  # Convert to kHz
        ax_freq.hist(freq_data, bins=30, alpha=0.7, edgecolor='black', linewidth=0.3,
                    color=pub_theme.get_colors()['primary'])
        ax_freq.set_xlabel('Peak Frequency (kHz)')
        ax_freq.set_ylabel('Count')
        ax_freq.tick_params(direction='in', which='both')
        ax_freq.grid(True, alpha=0.3, linewidth=0.3)
        multi_fig.add_panel_label(ax_freq, 'B')
    
    # Panel C: Parameter relationships
    if 'peak_freq' in detections.columns and 'duration' in detections.columns:
        ax_scatter = multi_fig.fig.add_subplot(multi_fig.gs[1, :2])
        axes['scatter'] = ax_scatter
        
        freq_data = detections['peak_freq'].dropna()
        dur_data = detections['duration'].dropna()
        
        # Find common indices
        common_idx = detections['peak_freq'].notna() & detections['duration'].notna()
        
        if common_idx.sum() > 0:
            x_data = detections.loc[common_idx, 'peak_freq'] / 1000
            y_data = detections.loc[common_idx, 'duration'] * 1000
            
            ax_scatter.scatter(x_data, y_data, alpha=0.6, s=10,
                             color=pub_theme.get_colors()['secondary'])
            
            # Add trend line
            if len(x_data) > 2:
                z = np.polyfit(x_data, y_data, 1)
                p = np.poly1d(z)
                ax_scatter.plot(x_data.sort_values(), p(x_data.sort_values()), 
                              'k--', linewidth=0.5, alpha=0.8)
        
        ax_scatter.set_xlabel('Peak Frequency (kHz)')
        ax_scatter.set_ylabel('Duration (ms)')
        ax_scatter.tick_params(direction='in', which='both')
        ax_scatter.grid(True, alpha=0.3, linewidth=0.3)
        multi_fig.add_panel_label(ax_scatter, 'C')
    
    # Panel D: Temporal pattern
    if 'UTC' in detections.columns:
        ax_temporal = multi_fig.fig.add_subplot(multi_fig.gs[1, 2:])
        axes['temporal'] = ax_temporal
        
        times = pd.to_datetime(detections['UTC'])
        hourly_counts = times.dt.hour.value_counts().sort_index()
        
        ax_temporal.bar(hourly_counts.index, hourly_counts.values, 
                       alpha=0.7, color=pub_theme.get_colors()['tertiary'],
                       edgecolor='black', linewidth=0.3)
        ax_temporal.set_xlabel('Hour of Day')
        ax_temporal.set_ylabel('Detection Count')
        ax_temporal.set_xticks(range(0, 24, 4))
        ax_temporal.tick_params(direction='in', which='both')
        ax_temporal.grid(True, alpha=0.3, linewidth=0.3)
        multi_fig.add_panel_label(ax_temporal, 'D')
    
    # Panel E: Summary statistics table
    ax_stats = multi_fig.fig.add_subplot(multi_fig.gs[2, :])
    axes['statistics'] = ax_stats
    
    # Create summary statistics
    stats_data = []
    
    stats_data.append(['Total Detections', str(len(detections)), '', ''])
    
    if 'peak_freq' in detections.columns:
        freq_data = detections['peak_freq'].dropna()
        stats_data.append(['Peak Frequency (kHz)', 
                          f'{freq_data.mean()/1000:.1f} ± {freq_data.std()/1000:.1f}',
                          f'{freq_data.min()/1000:.1f}',
                          f'{freq_data.max()/1000:.1f}'])
    
    if 'duration' in detections.columns:
        dur_data = detections['duration'].dropna()
        stats_data.append(['Duration (ms)', 
                          f'{dur_data.mean()*1000:.1f} ± {dur_data.std()*1000:.1f}',
                          f'{dur_data.min()*1000:.1f}',
                          f'{dur_data.max()*1000:.1f}'])
    
    # Create table
    table = ax_stats.table(cellText=stats_data,
                          colLabels=['Parameter', 'Mean ± SD', 'Min', 'Max'],
                          cellLoc='center',
                          loc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 1.5)
    
    # Style the table
    for i in range(len(stats_data) + 1):
        for j in range(4):
            cell = table[(i, j)]
            if i == 0:  # Header
                cell.set_facecolor('#E6E6E6')
                cell.set_text_props(weight='bold')
            cell.set_linewidth(0.5)
    
    ax_stats.axis('off')
    multi_fig.add_panel_label(ax_stats, 'E')
    
    plt.tight_layout()
    return multi_fig.fig, axes


def add_scale_bar(ax: plt.Axes, length: float, units: str = 's',
                 location: str = 'lower right', color: str = 'black',
                 linewidth: float = 2, fontsize: int = 8):
    """
    Add a scale bar to a plot.
    
    Args:
        ax: Matplotlib axes object
        length: Length of scale bar in data units
        units: Units label
        location: Location on plot
        color: Scale bar color
        linewidth: Line width
        fontsize: Font size for label
    """
    # Get axis limits
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # Calculate position based on location
    locations = {
        'lower right': (0.85, 0.1),
        'lower left': (0.15, 0.1),
        'upper right': (0.85, 0.9),
        'upper left': (0.15, 0.9)
    }
    
    rel_x, rel_y = locations.get(location, locations['lower right'])
    
    # Convert to data coordinates
    x_pos = xlim[0] + rel_x * (xlim[1] - xlim[0])
    y_pos = ylim[0] + rel_y * (ylim[1] - ylim[0])
    
    # Draw scale bar
    ax.plot([x_pos - length/2, x_pos + length/2], [y_pos, y_pos],
           color=color, linewidth=linewidth, solid_capstyle='butt')
    
    # Add label
    ax.text(x_pos, y_pos - 0.03 * (ylim[1] - ylim[0]), 
           f'{length} {units}', ha='center', va='top',
           fontsize=fontsize, color=color)


def add_inset_axes(ax: plt.Axes, bounds: Tuple[float, float, float, float],
                  data_limits: Tuple[float, float, float, float] = None) -> plt.Axes:
    """
    Add an inset axes to a plot.
    
    Args:
        ax: Parent axes
        bounds: (x, y, width, height) in axes coordinates (0-1)
        data_limits: (xmin, xmax, ymin, ymax) for inset data
        
    Returns:
        Inset axes object
    """
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    
    inset_ax = inset_axes(ax, width="100%", height="100%", 
                         bbox_to_anchor=bounds, bbox_transform=ax.transAxes)
    
    if data_limits:
        inset_ax.set_xlim(data_limits[0], data_limits[1])
        inset_ax.set_ylim(data_limits[2], data_limits[3])
    
    return inset_ax


def export_figure_formats(fig: plt.Figure, base_filename: str,
                         formats: List[str] = ['png', 'pdf', 'svg'],
                         dpi: int = 300, **kwargs):
    """
    Export figure in multiple formats.
    
    Args:
        fig: Matplotlib figure
        base_filename: Base filename without extension
        formats: List of formats to export
        dpi: Resolution for raster formats
        **kwargs: Additional arguments for savefig
    """
    base_path = Path(base_filename)
    
    for fmt in formats:
        output_path = base_path.with_suffix(f'.{fmt}')
        
        save_kwargs = {'dpi': dpi, 'bbox_inches': 'tight', 'pad_inches': 0.05}
        save_kwargs.update(kwargs)
        
        if fmt == 'pdf':
            save_kwargs.pop('dpi', None)  # PDF doesn't use DPI
        
        fig.savefig(output_path, format=fmt, **save_kwargs)
        print(f"Saved: {output_path}")


def create_journal_template(journal: str = 'nature') -> Dict[str, Any]:
    """
    Get figure specifications for specific journals.
    
    Args:
        journal: Journal name ('nature', 'science', 'plos', etc.)
        
    Returns:
        Dictionary with figure specifications
    """
    templates = {
        'nature': {
            'single_column': (3.5, 2.625),  # 89mm width
            'double_column': (7.0, 5.25),   # 178mm width
            'max_height': 9.5,              # 240mm
            'font_size': 7,
            'line_width': 0.5,
            'dpi': 300,
            'formats': ['pdf', 'png']
        },
        'science': {
            'single_column': (3.3, 2.475),  # 84mm width
            'double_column': (6.9, 5.175),  # 175mm width
            'max_height': 9.3,              # 235mm
            'font_size': 8,
            'line_width': 0.5,
            'dpi': 300,
            'formats': ['pdf', 'png']
        },
        'plos': {
            'single_column': (3.27, 2.45),  # 83mm width
            'double_column': (6.83, 5.12),  # 173.5mm width
            'max_height': 8.75,             # 222mm
            'font_size': 9,
            'line_width': 0.6,
            'dpi': 300,
            'formats': ['png', 'tiff']
        }
    }
    
    return templates.get(journal, templates['nature'])