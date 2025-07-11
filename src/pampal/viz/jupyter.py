"""
Jupyter notebook integration module for PAMpal visualizations.

This module provides widgets, dashboards, and live plotting capabilities
specifically designed for interactive analysis in Jupyter notebooks.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
import warnings

try:
    import ipywidgets as widgets
    from IPython.display import display, HTML, clear_output
    JUPYTER_AVAILABLE = True
except ImportError:
    JUPYTER_AVAILABLE = False
    widgets = None

from .core import VisualizationBase, ColorSchemes
from .waveforms import plot_waveform
from .spectrograms import plot_spectrogram
from .detections import plot_detection_overview


class JupyterVizWidget:
    """Base class for Jupyter visualization widgets."""
    
    def __init__(self):
        if not JUPYTER_AVAILABLE:
            raise ImportError("Jupyter widgets not available. Install with: pip install ipywidgets")
        
        self.colors = ColorSchemes()
        self.output = widgets.Output()
        
    def display(self):
        """Display the widget."""
        display(self.widget)


class DetectionExplorerWidget(JupyterVizWidget):
    """Interactive detection exploration widget for Jupyter notebooks."""
    
    def __init__(self, detections: pd.DataFrame, waveforms: Dict[str, np.ndarray] = None,
                 sample_rate: int = 192000):
        super().__init__()
        
        self.detections = detections
        self.waveforms = waveforms or {}
        self.sample_rate = sample_rate
        
        # Create widgets
        self._create_widgets()
        self._create_layout()
        
    def _create_widgets(self):
        """Create UI widgets."""
        # Detection selector
        detection_options = [(f"Detection {i}: {row.get('detection_type', 'Unknown')}", i) 
                           for i, row in self.detections.iterrows()]
        
        self.detection_selector = widgets.Dropdown(
            options=detection_options,
            description='Detection:',
            style={'description_width': 'initial'}
        )
        
        # Parameter selectors
        numeric_cols = self.detections.select_dtypes(include=[np.number]).columns.tolist()
        
        self.x_param = widgets.Dropdown(
            options=numeric_cols,
            value=numeric_cols[0] if numeric_cols else None,
            description='X Parameter:',
            style={'description_width': 'initial'}
        )
        
        self.y_param = widgets.Dropdown(
            options=numeric_cols,
            value=numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0],
            description='Y Parameter:',
            style={'description_width': 'initial'}
        )
        
        # Filter widgets
        if 'detection_type' in self.detections.columns:
            type_options = ['All'] + list(self.detections['detection_type'].unique())
            self.type_filter = widgets.SelectMultiple(
                options=type_options,
                value=['All'],
                description='Types:',
                style={'description_width': 'initial'}
            )
        else:
            self.type_filter = None
        
        # Frequency range filter
        if 'peak_freq' in self.detections.columns:
            freq_data = self.detections['peak_freq'].dropna()
            if len(freq_data) > 0:
                self.freq_range = widgets.FloatRangeSlider(
                    value=[freq_data.min(), freq_data.max()],
                    min=freq_data.min(),
                    max=freq_data.max(),
                    step=(freq_data.max() - freq_data.min()) / 100,
                    description='Freq Range (Hz):',
                    style={'description_width': 'initial'}
                )
            else:
                self.freq_range = None
        else:
            self.freq_range = None
        
        # Plot type selector
        self.plot_type = widgets.RadioButtons(
            options=['Scatter Plot', 'Detection Overview', 'Waveform', 'Spectrogram'],
            value='Scatter Plot',
            description='Plot Type:'
        )
        
        # Update button
        self.update_button = widgets.Button(
            description='Update Plot',
            button_style='primary'
        )
        
        # Set up callbacks
        self.update_button.on_click(self._update_plot)
        self.detection_selector.observe(self._on_detection_change, names='value')
        
    def _create_layout(self):
        """Create widget layout."""
        controls = [
            widgets.HBox([self.x_param, self.y_param]),
            self.plot_type
        ]
        
        if self.type_filter:
            controls.append(self.type_filter)
            
        if self.freq_range:
            controls.append(self.freq_range)
            
        controls.extend([
            self.detection_selector,
            self.update_button
        ])
        
        self.widget = widgets.VBox([
            widgets.HTML("<h3>Detection Explorer</h3>"),
            widgets.VBox(controls),
            self.output
        ])
        
        # Initial plot
        self._update_plot(None)
        
    def _filter_detections(self) -> pd.DataFrame:
        """Apply current filters to detections."""
        filtered = self.detections.copy()
        
        # Type filter
        if self.type_filter and 'All' not in self.type_filter.value:
            filtered = filtered[filtered['detection_type'].isin(self.type_filter.value)]
        
        # Frequency filter
        if self.freq_range and 'peak_freq' in filtered.columns:
            freq_min, freq_max = self.freq_range.value
            filtered = filtered[
                (filtered['peak_freq'] >= freq_min) & 
                (filtered['peak_freq'] <= freq_max)
            ]
        
        return filtered
        
    def _update_plot(self, button):
        """Update the plot based on current settings."""
        with self.output:
            clear_output(wait=True)
            
            try:
                filtered_detections = self._filter_detections()
                
                if filtered_detections.empty:
                    print("No detections match current filters")
                    return
                
                plt.figure(figsize=(10, 6))
                
                if self.plot_type.value == 'Scatter Plot':
                    self._plot_scatter(filtered_detections)
                elif self.plot_type.value == 'Detection Overview':
                    self._plot_overview(filtered_detections)
                elif self.plot_type.value == 'Waveform':
                    self._plot_waveform()
                elif self.plot_type.value == 'Spectrogram':
                    self._plot_spectrogram()
                
                plt.tight_layout()
                plt.show()
                
            except Exception as e:
                print(f"Error creating plot: {str(e)}")
    
    def _plot_scatter(self, detections: pd.DataFrame):
        """Create scatter plot."""
        if self.x_param.value and self.y_param.value:
            x_data = detections[self.x_param.value].dropna()
            y_data = detections[self.y_param.value].dropna()
            
            # Find common indices
            common_idx = detections[self.x_param.value].notna() & detections[self.y_param.value].notna()
            
            if common_idx.sum() > 0:
                x_vals = detections.loc[common_idx, self.x_param.value]
                y_vals = detections.loc[common_idx, self.y_param.value]
                
                if 'detection_type' in detections.columns:
                    for det_type in detections['detection_type'].unique():
                        type_mask = detections.loc[common_idx, 'detection_type'] == det_type
                        if type_mask.sum() > 0:
                            color = self.colors.detection_colors().get(det_type, '#1f77b4')
                            plt.scatter(x_vals[type_mask], y_vals[type_mask], 
                                      label=det_type, alpha=0.7, color=color)
                    plt.legend()
                else:
                    plt.scatter(x_vals, y_vals, alpha=0.7)
                
                plt.xlabel(self.x_param.value.replace('_', ' ').title())
                plt.ylabel(self.y_param.value.replace('_', ' ').title())
                plt.title(f'{self.x_param.value} vs {self.y_param.value}')
                plt.grid(True, alpha=0.3)
            else:
                plt.text(0.5, 0.5, 'No valid data for selected parameters', 
                        ha='center', va='center', transform=plt.gca().transAxes)
    
    def _plot_overview(self, detections: pd.DataFrame):
        """Create detection overview plot."""
        fig, axes = plot_detection_overview(detections)
        plt.show()
    
    def _plot_waveform(self):
        """Plot waveform for selected detection."""
        detection_idx = self.detection_selector.value
        detection_id = str(detection_idx)
        
        if detection_id in self.waveforms:
            waveform = self.waveforms[detection_id]
            fig, ax = plot_waveform(waveform, self.sample_rate, 
                                  title=f'Detection {detection_idx} Waveform')
        else:
            plt.text(0.5, 0.5, 'Waveform not available for this detection', 
                    ha='center', va='center', transform=plt.gca().transAxes)
    
    def _plot_spectrogram(self):
        """Plot spectrogram for selected detection."""
        detection_idx = self.detection_selector.value
        detection_id = str(detection_idx)
        
        if detection_id in self.waveforms:
            waveform = self.waveforms[detection_id]
            fig, ax = plot_spectrogram(waveform, self.sample_rate, 
                                     title=f'Detection {detection_idx} Spectrogram')
        else:
            plt.text(0.5, 0.5, 'Waveform not available for this detection', 
                    ha='center', va='center', transform=plt.gca().transAxes)
    
    def _on_detection_change(self, change):
        """Handle detection selection change."""
        if self.plot_type.value in ['Waveform', 'Spectrogram']:
            self._update_plot(None)


class StudyDashboard(JupyterVizWidget):
    """Interactive study-level dashboard widget."""
    
    def __init__(self, study_data: Dict[str, Any]):
        super().__init__()
        
        self.study_data = study_data
        self.detections = study_data.get('detections', pd.DataFrame())
        
        if self.detections.empty:
            raise ValueError("No detection data provided")
        
        self._create_widgets()
        self._create_layout()
    
    def _create_widgets(self):
        """Create dashboard widgets."""
        # Time range selector
        if 'UTC' in self.detections.columns:
            times = pd.to_datetime(self.detections['UTC'])
            self.time_range = widgets.SelectionRangeSlider(
                options=[(t.strftime('%Y-%m-%d %H:%M'), t) for t in times.sort_values()],
                index=(0, len(times) - 1),
                description='Time Range:',
                style={'description_width': 'initial'}
            )
        else:
            self.time_range = None
        
        # Detection type filter
        if 'detection_type' in self.detections.columns:
            type_options = list(self.detections['detection_type'].unique())
            self.type_selector = widgets.SelectMultiple(
                options=type_options,
                value=type_options,
                description='Detection Types:',
                style={'description_width': 'initial'}
            )
        else:
            self.type_selector = None
        
        # Visualization tabs
        self.viz_tabs = widgets.Tab()
        
        # Tab content outputs
        self.overview_output = widgets.Output()
        self.temporal_output = widgets.Output()
        self.spatial_output = widgets.Output()
        
        self.viz_tabs.children = [self.overview_output, self.temporal_output, self.spatial_output]
        self.viz_tabs.set_title(0, 'Overview')
        self.viz_tabs.set_title(1, 'Temporal')
        self.viz_tabs.set_title(2, 'Spatial')
        
        # Update button
        self.update_button = widgets.Button(
            description='Update Dashboard',
            button_style='primary'
        )
        
        self.update_button.on_click(self._update_dashboard)
    
    def _create_layout(self):
        """Create dashboard layout."""
        controls = []
        
        if self.time_range:
            controls.append(self.time_range)
        
        if self.type_selector:
            controls.append(self.type_selector)
        
        controls.append(self.update_button)
        
        self.widget = widgets.VBox([
            widgets.HTML("<h2>PAMpal Study Dashboard</h2>"),
            widgets.VBox(controls),
            self.viz_tabs
        ])
        
        # Initial update
        self._update_dashboard(None)
    
    def _filter_data(self) -> pd.DataFrame:
        """Apply current filters to data."""
        filtered = self.detections.copy()
        
        # Time filter
        if self.time_range:
            start_time, end_time = self.time_range.value
            times = pd.to_datetime(filtered['UTC'])
            filtered = filtered[(times >= start_time) & (times <= end_time)]
        
        # Type filter
        if self.type_selector:
            filtered = filtered[filtered['detection_type'].isin(self.type_selector.value)]
        
        return filtered
    
    def _update_dashboard(self, button):
        """Update all dashboard visualizations."""
        filtered_data = self._filter_data()
        
        if filtered_data.empty:
            for output in [self.overview_output, self.temporal_output, self.spatial_output]:
                with output:
                    clear_output(wait=True)
                    print("No data matches current filters")
            return
        
        # Update overview tab
        with self.overview_output:
            clear_output(wait=True)
            self._create_overview_plots(filtered_data)
        
        # Update temporal tab
        with self.temporal_output:
            clear_output(wait=True)
            self._create_temporal_plots(filtered_data)
        
        # Update spatial tab
        with self.spatial_output:
            clear_output(wait=True)
            self._create_spatial_plots(filtered_data)
    
    def _create_overview_plots(self, data: pd.DataFrame):
        """Create overview visualizations."""
        try:
            from .study import plot_study_overview
            
            study_data_filtered = {
                'detections': data,
                'survey_info': self.study_data.get('survey_info', {}),
                'effort_data': self.study_data.get('effort_data', {})
            }
            
            fig, axes = plot_study_overview(study_data_filtered, figsize=(12, 8))
            plt.show()
            
        except Exception as e:
            print(f"Error creating overview: {str(e)}")
    
    def _create_temporal_plots(self, data: pd.DataFrame):
        """Create temporal visualizations."""
        try:
            from .study import plot_temporal_patterns
            
            fig, axes = plot_temporal_patterns(data, figsize=(12, 8))
            plt.show()
            
        except Exception as e:
            print(f"Error creating temporal plots: {str(e)}")
    
    def _create_spatial_plots(self, data: pd.DataFrame):
        """Create spatial visualizations."""
        try:
            if 'latitude' in data.columns and 'longitude' in data.columns:
                from .study import plot_spatial_distribution
                
                fig, ax = plot_spatial_distribution(data, figsize=(10, 8))
                plt.show()
            else:
                print("No spatial data available (latitude/longitude columns missing)")
                
        except Exception as e:
            print(f"Error creating spatial plots: {str(e)}")


class LivePlotter:
    """Live plotting utility for real-time data visualization in Jupyter."""
    
    def __init__(self, plot_function: Callable, update_interval: float = 1.0):
        if not JUPYTER_AVAILABLE:
            raise ImportError("Jupyter widgets not available. Install with: pip install ipywidgets")
        
        self.plot_function = plot_function
        self.update_interval = update_interval
        self.output = widgets.Output()
        self.is_running = False
        
        # Control widgets
        self.start_button = widgets.Button(description='Start', button_style='success')
        self.stop_button = widgets.Button(description='Stop', button_style='danger')
        self.interval_slider = widgets.FloatSlider(
            value=update_interval,
            min=0.1,
            max=10.0,
            step=0.1,
            description='Update Interval (s):'
        )
        
        # Set up callbacks
        self.start_button.on_click(self._start_plotting)
        self.stop_button.on_click(self._stop_plotting)
        self.interval_slider.observe(self._update_interval, names='value')
        
        # Create layout
        self.widget = widgets.VBox([
            widgets.HBox([self.start_button, self.stop_button, self.interval_slider]),
            self.output
        ])
    
    def _start_plotting(self, button):
        """Start live plotting."""
        self.is_running = True
        self.start_button.disabled = True
        self.stop_button.disabled = False
        self._plot_loop()
    
    def _stop_plotting(self, button):
        """Stop live plotting."""
        self.is_running = False
        self.start_button.disabled = False
        self.stop_button.disabled = True
    
    def _update_interval(self, change):
        """Update plotting interval."""
        self.update_interval = change['new']
    
    def _plot_loop(self):
        """Main plotting loop."""
        import time
        
        while self.is_running:
            with self.output:
                clear_output(wait=True)
                try:
                    self.plot_function()
                    plt.show()
                except Exception as e:
                    print(f"Error in plot function: {str(e)}")
            
            time.sleep(self.update_interval)
    
    def display(self):
        """Display the live plotter widget."""
        display(self.widget)


def create_parameter_selector(detections: pd.DataFrame):
    """
    Create an interactive parameter selector widget.
    
    Args:
        detections: DataFrame with detection parameters
        
    Returns:
        Interactive widget for parameter selection and filtering
    """
    if not JUPYTER_AVAILABLE:
        raise ImportError("Jupyter widgets not available. Install with: pip install ipywidgets")
    
    import ipywidgets as widgets  # Import locally when needed
    
    # Get numeric columns
    numeric_cols = detections.select_dtypes(include=[np.number]).columns.tolist()
    
    # Parameter selection widgets
    param_widgets = {}
    
    for col in numeric_cols:
        data = detections[col].dropna()
        if len(data) > 0:
            param_widgets[col] = widgets.FloatRangeSlider(
                value=[data.min(), data.max()],
                min=data.min(),
                max=data.max(),
                step=(data.max() - data.min()) / 100,
                description=f'{col}:',
                style={'description_width': 'initial'}
            )
    
    # Create widget layout
    widget_list = [widgets.HTML("<h4>Parameter Filters</h4>")]
    widget_list.extend(param_widgets.values())
    
    return widgets.VBox(widget_list)


def display_detection_table(detections: pd.DataFrame, 
                           columns: List[str] = None,
                           max_rows: int = 100) -> None:
    """
    Display detection data as an interactive table.
    
    Args:
        detections: DataFrame with detection data
        columns: Columns to display (None for all)
        max_rows: Maximum number of rows to display
    """
    if not JUPYTER_AVAILABLE:
        raise ImportError("Jupyter widgets not available. Install with: pip install ipywidgets")
    
    from IPython.display import display, HTML  # Import locally when needed
    
    display_data = detections.copy()
    
    if columns:
        display_data = display_data[columns]
    
    if len(display_data) > max_rows:
        display_data = display_data.head(max_rows)
        print(f"Showing first {max_rows} of {len(detections)} detections")
    
    # Format numeric columns
    numeric_cols = display_data.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if display_data[col].dtype == float:
            display_data[col] = display_data[col].round(3)
    
    # Display as HTML table
    html_table = display_data.to_html(classes='table table-striped', table_id='detection-table')
    
    # Add some styling
    styled_html = f"""
    <style>
    #detection-table {{
        font-size: 12px;
        border-collapse: collapse;
        width: 100%;
    }}
    #detection-table th, #detection-table td {{
        border: 1px solid #ddd;
        padding: 8px;
        text-align: left;
    }}
    #detection-table th {{
        background-color: #f2f2f2;
        font-weight: bold;
    }}
    #detection-table tr:nth-child(even) {{
        background-color: #f9f9f9;
    }}
    </style>
    {html_table}
    """
    
    display(HTML(styled_html))


def export_notebook_plots(notebook_path: str, output_dir: str = 'plots') -> None:
    """
    Export all plots from a Jupyter notebook to individual files.
    
    Args:
        notebook_path: Path to the Jupyter notebook file
        output_dir: Directory to save exported plots
    """
    try:
        import nbformat
        from nbconvert import PythonExporter
        import os
        
        # Read notebook
        with open(notebook_path, 'r') as f:
            notebook = nbformat.read(f, as_version=4)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Export plots (this would need more sophisticated implementation)
        print(f"Notebook plot export functionality not yet implemented")
        print(f"Would export plots from {notebook_path} to {output_dir}")
        
    except ImportError:
        print("nbformat and nbconvert required for notebook export")
        print("Install with: pip install nbformat nbconvert")