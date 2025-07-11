"""
Interactive visualization module for PAMpal.

This module provides interactive plotting capabilities using Plotly,
including detection browsers, parameter selectors, and dashboard-style
visualizations for exploratory data analysis.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
import warnings

from .core import ColorSchemes


class InteractivePlotter:
    """Interactive plotter using Plotly for exploratory data analysis."""
    
    def __init__(self):
        self.colors = ColorSchemes()
        self.detection_colors = self.colors.detection_colors()
        
    def _get_detection_color(self, detection_type: str) -> str:
        """Get color for detection type."""
        return self.detection_colors.get(detection_type, self.detection_colors['unknown'])


def plot_interactive_spectrogram(waveform: np.ndarray, sample_rate: int = 192000,
                                window_size: int = 512, overlap: float = 0.75,
                                freq_range: Tuple[float, float] = None,
                                detections: pd.DataFrame = None,
                                title: str = None) -> go.Figure:
    """
    Create an interactive spectrogram with detection overlays.
    
    Args:
        waveform: 1D array of audio samples
        sample_rate: Sample rate in Hz
        window_size: FFT window size
        overlap: Overlap between windows (0-1)
        freq_range: Frequency range to display (min_freq, max_freq) in Hz
        detections: DataFrame with detection information
        title: Plot title
        
    Returns:
        Plotly Figure object
    """
    from ..signal_processing import calculate_spectrogram
    
    # Calculate spectrogram
    Sxx_db, frequencies, times = calculate_spectrogram(
        waveform, sample_rate, window_size=window_size, overlap=overlap
    )
    
    # Apply frequency range filter
    if freq_range is not None:
        freq_mask = (frequencies >= freq_range[0]) & (frequencies <= freq_range[1])
        frequencies = frequencies[freq_mask]
        Sxx_db = Sxx_db[freq_mask, :]
    
    # Create interactive heatmap
    fig = go.Figure(data=go.Heatmap(
        z=Sxx_db,
        x=times,
        y=frequencies,
        colorscale='Viridis',
        colorbar=dict(title='Power (dB)'),
        hovertemplate='Time: %{x:.3f}s<br>Frequency: %{y:.0f}Hz<br>Power: %{z:.1f}dB<extra></extra>'
    ))
    
    # Add detection overlays if provided
    if detections is not None and not detections.empty:
        plotter = InteractivePlotter()
        
        if 'time' in detections.columns:
            for _, detection in detections.iterrows():
                det_time = detection['time']
                det_type = detection.get('detection_type', 'unknown')
                color = plotter._get_detection_color(det_type)
                
                if 'frequency' in detection and pd.notna(detection['frequency']):
                    # Point detection
                    freq = detection['frequency']
                    fig.add_scatter(
                        x=[det_time], y=[freq],
                        mode='markers',
                        marker=dict(size=8, color=color, symbol='circle', 
                                  line=dict(width=2, color='white')),
                        name=f'{det_type.title()}',
                        showlegend=True,
                        hovertemplate=f'Detection: {det_type}<br>Time: %{{x:.3f}}s<br>Frequency: %{{y:.0f}}Hz<extra></extra>'
                    )
                else:
                    # Vertical line for time-only detection
                    fig.add_vline(
                        x=det_time,
                        line=dict(color=color, width=2, dash='dash'),
                        annotation_text=det_type,
                        annotation_position="top"
                    )
    
    # Update layout
    fig.update_layout(
        title=title or f'Interactive Spectrogram (SR: {sample_rate}Hz)',
        xaxis_title='Time (s)',
        yaxis_title='Frequency (Hz)',
        hovermode='closest',
        width=800,
        height=500
    )
    
    return fig


def plot_interactive_waveform(waveform: np.ndarray, sample_rate: int = 192000,
                             detections: pd.DataFrame = None,
                             title: str = None) -> go.Figure:
    """
    Create an interactive waveform plot with detection markers.
    
    Args:
        waveform: 1D array of audio samples
        sample_rate: Sample rate in Hz
        detections: DataFrame with detection information
        title: Plot title
        
    Returns:
        Plotly Figure object
    """
    time = np.arange(len(waveform)) / sample_rate
    
    # Create waveform trace
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=time,
        y=waveform,
        mode='lines',
        name='Waveform',
        line=dict(color='blue', width=1),
        hovertemplate='Time: %{x:.3f}s<br>Amplitude: %{y:.3f}<extra></extra>'
    ))
    
    # Add detection markers if provided
    if detections is not None and not detections.empty:
        plotter = InteractivePlotter()
        
        if 'time' in detections.columns:
            for det_type in detections.get('detection_type', ['unknown']).unique():
                type_detections = detections[detections.get('detection_type', 'unknown') == det_type] \
                    if 'detection_type' in detections.columns else detections
                
                color = plotter._get_detection_color(det_type)
                
                for _, detection in type_detections.iterrows():
                    det_time = detection['time']
                    
                    # Find closest sample
                    sample_idx = int(det_time * sample_rate)
                    if 0 <= sample_idx < len(waveform):
                        amplitude = waveform[sample_idx]
                        
                        fig.add_trace(go.Scatter(
                            x=[det_time],
                            y=[amplitude],
                            mode='markers',
                            marker=dict(size=8, color=color, symbol='diamond'),
                            name=f'{det_type.title()} Detection',
                            showlegend=True,
                            hovertemplate=f'Detection: {det_type}<br>Time: %{{x:.3f}}s<br>Amplitude: %{{y:.3f}}<extra></extra>'
                        ))
    
    # Update layout
    fig.update_layout(
        title=title or f'Interactive Waveform (SR: {sample_rate}Hz)',
        xaxis_title='Time (s)',
        yaxis_title='Amplitude',
        hovermode='x unified',
        width=800,
        height=400
    )
    
    return fig


def plot_detection_browser(detections: pd.DataFrame,
                          parameters: List[str] = None,
                          color_by: str = 'detection_type') -> go.Figure:
    """
    Create an interactive detection parameter browser.
    
    Args:
        detections: DataFrame with detection parameters
        parameters: List of parameters to include in browser
        color_by: Column to use for color coding
        
    Returns:
        Plotly Figure object with interactive parameter plots
    """
    if parameters is None:
        available_params = ['peak_freq', 'duration', 'amplitude', 'bandwidth', 'centroid_freq']
        parameters = [p for p in available_params if p in detections.columns]
    
    if len(parameters) < 2:
        raise ValueError("Need at least 2 parameters for browser")
    
    # Create subplot figure
    n_params = len(parameters)
    subplot_titles = [param.replace('_', ' ').title() for param in parameters]
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Parameter Distributions', 'Parameter Relationships', 
                       'Time Series', 'Summary Statistics'],
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"type": "table"}]]
    )
    
    plotter = InteractivePlotter()
    
    # 1. Parameter distributions (histograms)
    for i, param in enumerate(parameters[:3]):  # Show first 3 parameters
        data = detections[param].dropna()
        if len(data) > 0:
            fig.add_trace(
                go.Histogram(
                    x=data,
                    name=param.replace('_', ' ').title(),
                    opacity=0.7,
                    nbinsx=30
                ),
                row=1, col=1
            )
    
    # 2. Parameter relationships (scatter plot)
    if len(parameters) >= 2:
        x_param, y_param = parameters[0], parameters[1]
        
        if color_by in detections.columns:
            unique_values = detections[color_by].unique()
            for value in unique_values:
                mask = detections[color_by] == value
                subset = detections[mask]
                
                color = plotter._get_detection_color(str(value)) if color_by == 'detection_type' else None
                
                fig.add_trace(
                    go.Scatter(
                        x=subset[x_param],
                        y=subset[y_param],
                        mode='markers',
                        name=str(value),
                        marker=dict(color=color, size=6),
                        hovertemplate=f'{x_param}: %{{x}}<br>{y_param}: %{{y}}<br>{color_by}: {value}<extra></extra>'
                    ),
                    row=1, col=2
                )
        else:
            fig.add_trace(
                go.Scatter(
                    x=detections[x_param],
                    y=detections[y_param],
                    mode='markers',
                    name='Detections',
                    hovertemplate=f'{x_param}: %{{x}}<br>{y_param}: %{{y}}<extra></extra>'
                ),
                row=1, col=2
            )
    
    # 3. Time series (if UTC column available)
    if 'UTC' in detections.columns:
        times = pd.to_datetime(detections['UTC'])
        
        # Plot detection count over time
        hourly_counts = times.dt.floor('H').value_counts().sort_index()
        
        fig.add_trace(
            go.Scatter(
                x=hourly_counts.index,
                y=hourly_counts.values,
                mode='lines+markers',
                name='Detections/Hour',
                hovertemplate='Time: %{x}<br>Count: %{y}<extra></extra>'
            ),
            row=2, col=1
        )
    
    # 4. Summary statistics table
    stats_data = []
    for param in parameters:
        if param in detections.columns:
            data = detections[param].dropna()
            if len(data) > 0:
                stats_data.append([
                    param.replace('_', ' ').title(),
                    f"{len(data)}",
                    f"{data.mean():.3f}",
                    f"{data.std():.3f}",
                    f"{data.min():.3f}",
                    f"{data.max():.3f}"
                ])
    
    if stats_data:
        fig.add_trace(
            go.Table(
                header=dict(values=['Parameter', 'Count', 'Mean', 'Std', 'Min', 'Max'],
                           fill_color='lightblue'),
                cells=dict(values=list(zip(*stats_data)),
                          fill_color='white')
            ),
            row=2, col=2
        )
    
    # Update layout
    fig.update_layout(
        title=f'Interactive Detection Browser (n={len(detections)} detections)',
        height=800,
        showlegend=True
    )
    
    # Update axes labels
    fig.update_xaxes(title_text=x_param.replace('_', ' ').title(), row=1, col=2)
    fig.update_yaxes(title_text=y_param.replace('_', ' ').title(), row=1, col=2)
    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_yaxes(title_text="Detections/Hour", row=2, col=1)
    
    return fig


def create_detection_dashboard(study_data: Dict[str, Any]) -> go.Figure:
    """
    Create an interactive dashboard for study-level data exploration.
    
    Args:
        study_data: Dictionary containing study information
        
    Returns:
        Plotly Figure object with dashboard layout
    """
    detections = study_data.get('detections', pd.DataFrame())
    survey_info = study_data.get('survey_info', {})
    
    if detections.empty:
        raise ValueError("No detection data provided")
    
    # Create subplot dashboard
    fig = make_subplots(
        rows=3, cols=3,
        subplot_titles=[
            'Detection Types', 'Frequency Distribution', 'Temporal Pattern',
            'Spatial Distribution', 'Detection Timeline', 'Parameter Correlation',
            'Daily Activity', 'Effort Summary', 'Detection Statistics'
        ],
        specs=[
            [{"type": "pie"}, {"type": "histogram"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "scatter"}, {"type": "heatmap"}],
            [{"type": "bar"}, {"type": "indicator"}, {"type": "table"}]
        ]
    )
    
    plotter = InteractivePlotter()
    
    # 1. Detection type pie chart
    if 'detection_type' in detections.columns:
        type_counts = detections['detection_type'].value_counts()
        colors = [plotter._get_detection_color(t) for t in type_counts.index]
        
        fig.add_trace(
            go.Pie(
                labels=type_counts.index,
                values=type_counts.values,
                marker=dict(colors=colors),
                hovertemplate='%{label}: %{value} (%{percent})<extra></extra>'
            ),
            row=1, col=1
        )
    
    # 2. Frequency histogram
    if 'peak_freq' in detections.columns:
        freq_data = detections['peak_freq'].dropna() / 1000  # Convert to kHz
        
        fig.add_trace(
            go.Histogram(
                x=freq_data,
                nbinsx=30,
                name='Frequency',
                hovertemplate='Frequency: %{x}kHz<br>Count: %{y}<extra></extra>'
            ),
            row=1, col=2
        )
    
    # 3. Temporal pattern
    if 'UTC' in detections.columns:
        times = pd.to_datetime(detections['UTC'])
        hourly_counts = times.dt.floor('H').value_counts().sort_index()
        
        fig.add_trace(
            go.Scatter(
                x=hourly_counts.index,
                y=hourly_counts.values,
                mode='lines+markers',
                name='Timeline',
                hovertemplate='Time: %{x}<br>Detections: %{y}<extra></extra>'
            ),
            row=1, col=3
        )
    
    # 4. Spatial distribution
    if 'latitude' in detections.columns and 'longitude' in detections.columns:
        valid_coords = detections.dropna(subset=['latitude', 'longitude'])
        
        if not valid_coords.empty:
            fig.add_trace(
                go.Scatter(
                    x=valid_coords['longitude'],
                    y=valid_coords['latitude'],
                    mode='markers',
                    name='Locations',
                    hovertemplate='Lat: %{y:.3f}<br>Lon: %{x:.3f}<extra></extra>'
                ),
                row=2, col=1
            )
    
    # 5. Detection timeline (cumulative)
    if 'UTC' in detections.columns:
        times = pd.to_datetime(detections['UTC']).sort_values()
        cumulative = np.arange(1, len(times) + 1)
        
        fig.add_trace(
            go.Scatter(
                x=times,
                y=cumulative,
                mode='lines',
                name='Cumulative',
                hovertemplate='Time: %{x}<br>Total: %{y}<extra></extra>'
            ),
            row=2, col=2
        )
    
    # 6. Parameter correlation heatmap
    numeric_cols = detections.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        corr_matrix = detections[numeric_cols].corr()
        
        fig.add_trace(
            go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.index,
                colorscale='RdBu',
                zmid=0,
                hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>'
            ),
            row=2, col=3
        )
    
    # 7. Daily activity pattern
    if 'UTC' in detections.columns:
        times = pd.to_datetime(detections['UTC'])
        hour_counts = times.dt.hour.value_counts().sort_index()
        
        fig.add_trace(
            go.Bar(
                x=hour_counts.index,
                y=hour_counts.values,
                name='Hourly',
                hovertemplate='Hour: %{x}:00<br>Count: %{y}<extra></extra>'
            ),
            row=3, col=1
        )
    
    # 8. Effort indicator
    effort_data = study_data.get('effort_data', {})
    total_detections = len(detections)
    
    fig.add_trace(
        go.Indicator(
            mode="number+delta",
            value=total_detections,
            title={"text": "Total Detections"},
            delta={'reference': effort_data.get('expected_detections', total_detections)}
        ),
        row=3, col=2
    )
    
    # 9. Summary statistics table
    summary_stats = []
    if 'peak_freq' in detections.columns:
        freq_data = detections['peak_freq'].dropna()
        summary_stats.append(['Peak Frequency (kHz)', f"{freq_data.mean()/1000:.1f}", 
                             f"{freq_data.std()/1000:.1f}", f"{len(freq_data)}"])
    
    if 'duration' in detections.columns:
        dur_data = detections['duration'].dropna()
        summary_stats.append(['Duration (ms)', f"{dur_data.mean()*1000:.1f}", 
                             f"{dur_data.std()*1000:.1f}", f"{len(dur_data)}"])
    
    if summary_stats:
        fig.add_trace(
            go.Table(
                header=dict(values=['Parameter', 'Mean', 'Std', 'Count']),
                cells=dict(values=list(zip(*summary_stats)))
            ),
            row=3, col=3
        )
    
    # Update layout
    fig.update_layout(
        title=f"PAMpal Study Dashboard: {survey_info.get('name', 'Unknown Study')}",
        height=900,
        showlegend=False
    )
    
    return fig


def launch_detection_explorer(detections: pd.DataFrame,
                             waveforms: Dict[str, np.ndarray] = None,
                             sample_rate: int = 192000,
                             port: int = 8050) -> None:
    """
    Launch an interactive web-based detection explorer.
    
    Args:
        detections: DataFrame with detection data
        waveforms: Optional dictionary of waveforms for each detection
        sample_rate: Sample rate for waveforms
        port: Port for web server
    """
    try:
        import dash
        from dash import dcc, html, Input, Output, callback
    except ImportError:
        raise ImportError("Dash is required for the detection explorer. Install with: pip install dash")
    
    app = dash.Dash(__name__)
    
    # Create detection browser figure
    browser_fig = plot_detection_browser(detections)
    
    # App layout
    app.layout = html.Div([
        html.H1("PAMpal Detection Explorer"),
        
        html.Div([
            dcc.Graph(
                id='detection-browser',
                figure=browser_fig
            )
        ]),
        
        html.Div([
            html.H3("Detection Details"),
            html.Div(id='detection-details')
        ])
    ])
    
    @app.callback(
        Output('detection-details', 'children'),
        [Input('detection-browser', 'clickData')]
    )
    def display_detection_details(clickData):
        if clickData is None:
            return "Click on a detection point to see details"
        
        # Extract clicked point information
        point = clickData['points'][0]
        return html.Div([
            html.P(f"X: {point.get('x', 'N/A')}"),
            html.P(f"Y: {point.get('y', 'N/A')}"),
            html.P(f"Curve: {point.get('curveNumber', 'N/A')}")
        ])
    
    print(f"Starting detection explorer at http://localhost:{port}")
    app.run_server(debug=True, port=port)


def save_interactive_plot(fig: go.Figure, filename: str, 
                         format: str = 'html') -> None:
    """
    Save interactive plot to file.
    
    Args:
        fig: Plotly figure object
        filename: Output filename
        format: Output format ('html', 'png', 'pdf', 'svg')
    """
    if format == 'html':
        fig.write_html(filename)
    elif format == 'png':
        fig.write_image(filename, format='png')
    elif format == 'pdf':
        fig.write_image(filename, format='pdf')
    elif format == 'svg':
        fig.write_image(filename, format='svg')
    else:
        raise ValueError(f"Unsupported format: {format}")