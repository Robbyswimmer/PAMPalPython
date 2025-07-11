"""
Unit tests for PAMpal visualization module.

This module contains comprehensive tests for all visualization components,
including plotting functions, interactive tools, and optimization utilities.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pytest
import tempfile
import shutil
import os
from datetime import datetime, timedelta

# Import PAMpal visualization modules
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from pathlib import Path
except ImportError:
    # Python 2.7 compatibility
    class Path(object):
        def __init__(self, path):
            self.path = str(path)
        
        def exists(self):
            return os.path.exists(self.path)
        
        def with_suffix(self, suffix):
            base = os.path.splitext(self.path)[0]
            return Path(base + suffix)

# Import visualization components
from pampal.viz import (
    VisualizationBase, ColorSchemes, PampalTheme,
    plot_waveform, plot_spectrogram, plot_detection_overview,
    plot_study_overview, plot_temporal_patterns
)
from pampal.viz.waveforms import plot_multi_waveform
from pampal.viz.spectrograms import plot_spectrogram_with_detections
from pampal.viz.optimization import VisualizationCache, MemoryManager, DataDownsampler


class TestVisualizationBase:
    """Test core visualization infrastructure."""
    
    def test_visualization_base_init(self):
        """Test VisualizationBase initialization."""
        viz = VisualizationBase()
        assert viz.theme is not None
        assert viz.colors is not None
    
    def test_color_schemes(self):
        """Test ColorSchemes functionality."""
        colors = ColorSchemes()
        
        # Test detection colors
        det_colors = colors.detection_colors()
        assert isinstance(det_colors, dict)
        assert 'click' in det_colors
        assert 'whistle' in det_colors
        
        # Test species colors
        species_colors = colors.species_colors()
        assert isinstance(species_colors, dict)
        
        # Test scientific palettes
        viridis = colors.scientific_palette('viridis')
        assert len(viridis) > 0
    
    def test_pampal_theme(self):
        """Test PampalTheme functionality."""
        theme = PampalTheme('default')
        assert theme.name == 'default'
        
        # Test theme parameters
        params = theme.get_params()
        assert isinstance(params, dict)
        assert 'font.size' in params


class TestWaveformPlots:
    """Test waveform plotting functions."""
    
    @pytest.fixture
    def sample_waveform(self):
        """Create sample waveform data."""
        sample_rate = 192000
        duration = 0.1  # 100ms
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Create a simple chirp signal
        f0, f1 = 10000, 50000
        waveform = np.sin(2 * np.pi * (f0 + (f1 - f0) * t / duration) * t)
        
        return waveform, sample_rate
    
    def test_plot_waveform_basic(self, sample_waveform):
        """Test basic waveform plotting."""
        waveform, sample_rate = sample_waveform
        
        fig, ax = plot_waveform(waveform, sample_rate)
        
        assert fig is not None
        assert ax is not None
        assert len(ax.lines) > 0
        
        plt.close(fig)
    
    def test_plot_waveform_with_envelope(self, sample_waveform):
        """Test waveform plotting with envelope."""
        waveform, sample_rate = sample_waveform
        
        fig, ax = plot_waveform(waveform, sample_rate, show_envelope=True)
        
        assert fig is not None
        assert ax is not None
        assert len(ax.lines) >= 2  # Waveform + envelope
        
        plt.close(fig)
    
    def test_plot_multi_waveform(self, sample_waveform):
        """Test multi-waveform plotting."""
        waveform, sample_rate = sample_waveform
        
        # Create multiple waveforms
        waveforms = {
            'Signal 1': waveform,
            'Signal 2': waveform * 0.5,
            'Signal 3': waveform * 2.0
        }
        
        fig, axes = plot_multi_waveform(waveforms, sample_rate)
        
        assert fig is not None
        assert len(axes) == len(waveforms)
        
        plt.close(fig)


class TestSpectrogramPlots:
    """Test spectrogram plotting functions."""
    
    @pytest.fixture
    def sample_spectrogram_data(self):
        """Create sample data for spectrogram tests."""
        sample_rate = 192000
        duration = 0.1  # Shorter duration for faster tests
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Create a frequency sweep
        f0, f1 = 5000, 50000
        waveform = np.sin(2 * np.pi * (f0 + (f1 - f0) * t / duration) * t)
        
        # Add some noise
        waveform += 0.1 * np.random.randn(len(waveform))
        
        return waveform, sample_rate
    
    def test_plot_spectrogram_basic(self, sample_spectrogram_data):
        """Test basic spectrogram plotting."""
        waveform, sample_rate = sample_spectrogram_data
        
        fig, ax = plot_spectrogram(waveform, sample_rate)
        
        assert fig is not None
        assert ax is not None
        # Check for spectrogram display - either images or collections
        has_plot_data = len(ax.images) > 0 or len(ax.collections) > 0
        assert has_plot_data, "Spectrogram should have plot data"
        
        plt.close(fig)
    
    def test_plot_spectrogram_with_detections(self, sample_spectrogram_data):
        """Test spectrogram with detection overlays."""
        waveform, sample_rate = sample_spectrogram_data
        
        # Create sample detections
        detections = pd.DataFrame({
            'time': [0.2, 0.5, 0.8],
            'frequency': [15000, 25000, 35000],
            'detection_type': ['click', 'whistle', 'click']
        })
        
        fig, ax = plot_spectrogram_with_detections(waveform, detections, sample_rate)
        
        assert fig is not None
        assert ax is not None
        
        plt.close(fig)


class TestDetectionPlots:
    """Test detection analysis plotting functions."""
    
    @pytest.fixture
    def sample_detections(self):
        """Create sample detection data."""
        n_detections = 100
        
        # Create synthetic detection data
        np.random.seed(42)
        
        base_time = datetime.now()
        times = [base_time + timedelta(seconds=i * 0.5 + np.random.random() * 0.1) 
                for i in range(n_detections)]
        
        detections = pd.DataFrame({
            'UTC': times,
            'detection_type': np.random.choice(['click', 'whistle', 'moan'], n_detections),
            'peak_freq': 20000 + 30000 * np.random.random(n_detections),
            'duration': 0.001 + 0.004 * np.random.random(n_detections),
            'amplitude': 0.5 + 0.5 * np.random.random(n_detections),
            'bandwidth': 5000 + 15000 * np.random.random(n_detections)
        })
        
        return detections
    
    def test_plot_detection_overview(self, sample_detections):
        """Test detection overview plotting."""
        fig, axes = plot_detection_overview(sample_detections)
        
        assert fig is not None
        assert axes is not None
        assert isinstance(axes, np.ndarray)
        
        plt.close(fig)
    
    def test_plot_click_parameters(self, sample_detections):
        """Test click parameter analysis plotting."""
        from pampal.viz.detections import plot_click_parameters
        
        click_detections = sample_detections[sample_detections['detection_type'] == 'click']
        
        if len(click_detections) > 0:
            fig, axes = plot_click_parameters(click_detections, 
                                            parameters=['peak_freq', 'duration', 'amplitude'])
            
            assert fig is not None
            assert axes is not None
            
            plt.close(fig)
    
    def test_plot_ici_analysis(self, sample_detections):
        """Test inter-click interval analysis."""
        from pampal.viz.detections import plot_ici_analysis
        
        click_detections = sample_detections[sample_detections['detection_type'] == 'click']
        
        if len(click_detections) >= 2:
            fig, axes = plot_ici_analysis(click_detections)
            
            assert fig is not None
            assert axes is not None
            
            plt.close(fig)


class TestStudyLevelPlots:
    """Test study-level visualization functions."""
    
    @pytest.fixture
    def sample_study_data(self):
        """Create sample study data."""
        n_detections = 500
        np.random.seed(42)
        
        # Create detection data with location
        base_time = datetime.now()
        times = [base_time + timedelta(hours=i * 0.1 + np.random.random() * 0.05) 
                for i in range(n_detections)]
        
        detections = pd.DataFrame({
            'UTC': times,
            'detection_type': np.random.choice(['click', 'whistle', 'moan'], n_detections),
            'peak_freq': 15000 + 40000 * np.random.random(n_detections),
            'duration': 0.001 + 0.009 * np.random.random(n_detections),
            'amplitude': 0.3 + 0.7 * np.random.random(n_detections),
            'latitude': 36.0 + 0.5 * np.random.randn(n_detections),
            'longitude': -121.0 + 0.5 * np.random.randn(n_detections)
        })
        
        study_data = {
            'detections': detections,
            'survey_info': {
                'name': 'Test Survey',
                'location': 'Test Location',
                'start_date': '2024-01-01',
                'end_date': '2024-01-10'
            },
            'effort_data': {
                'total_hours': 240,
                'active_hours': 180
            }
        }
        
        return study_data
    
    def test_plot_study_overview(self, sample_study_data):
        """Test study overview plotting."""
        fig, axes = plot_study_overview(sample_study_data)
        
        assert fig is not None
        assert axes is not None
        assert isinstance(axes, np.ndarray)
        
        plt.close(fig)
    
    def test_plot_temporal_patterns(self, sample_study_data):
        """Test temporal patterns plotting."""
        detections = sample_study_data['detections']
        
        fig, axes = plot_temporal_patterns(detections)
        
        assert fig is not None
        assert axes is not None
        
        plt.close(fig)
    
    def test_plot_spatial_distribution(self, sample_study_data):
        """Test spatial distribution plotting."""
        from pampal.viz.study import plot_spatial_distribution
        
        detections = sample_study_data['detections']
        
        fig, ax = plot_spatial_distribution(detections)
        
        assert fig is not None
        assert ax is not None
        
        plt.close(fig)


class TestInteractivePlots:
    """Test interactive visualization functions."""
    
    @pytest.fixture
    def sample_interactive_data(self):
        """Create sample data for interactive tests."""
        n_detections = 50
        np.random.seed(42)
        
        detections = pd.DataFrame({
            'time': np.sort(np.random.random(n_detections) * 10),
            'frequency': 20000 + 30000 * np.random.random(n_detections),
            'amplitude': 0.5 + 0.5 * np.random.random(n_detections),
            'detection_type': np.random.choice(['click', 'whistle'], n_detections)
        })
        
        # Create sample waveform
        sample_rate = 192000
        duration = 0.1
        t = np.linspace(0, duration, int(sample_rate * duration))
        waveform = np.sin(2 * np.pi * 25000 * t) * np.exp(-t * 20)
        
        return detections, waveform, sample_rate
    
    def test_interactive_spectrogram(self, sample_interactive_data):
        """Test interactive spectrogram creation."""
        try:
            from pampal.viz.interactive import plot_interactive_spectrogram
            
            detections, waveform, sample_rate = sample_interactive_data
            
            fig = plot_interactive_spectrogram(waveform, sample_rate, detections=detections)
            
            assert fig is not None
            assert hasattr(fig, 'data')  # Plotly figure
            
        except ImportError:
            pytest.skip("Plotly not available for interactive tests")
    
    def test_detection_browser(self, sample_interactive_data):
        """Test detection browser creation."""
        try:
            from pampal.viz.interactive import plot_detection_browser
            
            detections, _, _ = sample_interactive_data
            
            fig = plot_detection_browser(detections, parameters=['frequency', 'amplitude'])
            
            assert fig is not None
            assert hasattr(fig, 'data')  # Plotly figure
            
        except ImportError:
            pytest.skip("Plotly not available for interactive tests")


class TestOptimization:
    """Test optimization and caching utilities."""
    
    def test_visualization_cache(self):
        """Test visualization caching system."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = VisualizationCache(cache_dir=temp_dir)
            
            # Test cache miss
            data = np.random.random(100)
            params = {'param1': 1, 'param2': 'test'}
            
            result = cache.get(data, params)
            assert result is None
            
            # Test cache set and hit
            test_result = {'test': 'value'}
            cache.set(data, params, test_result)
            
            cached_result = cache.get(data, params)
            assert cached_result == test_result
            
            # Test cache stats
            stats = cache.get_stats()
            assert stats['num_entries'] >= 1
    
    def test_memory_manager(self):
        """Test memory management utilities."""
        # Test memory usage reporting
        memory_info = MemoryManager.get_memory_usage()
        assert 'rss_mb' in memory_info
        assert 'available_mb' in memory_info
        
        # Test array memory estimation
        memory_mb = MemoryManager.estimate_array_memory((1000, 1000), np.float64)
        assert memory_mb > 0
        
        # Test chunked processing
        large_array = np.random.random(10000)
        chunks = MemoryManager.chunked_processing(large_array, chunk_size=1000)
        assert len(chunks) == 10
    
    def test_data_downsampler(self):
        """Test data downsampling utilities."""
        # Test waveform downsampling
        long_waveform = np.random.random(100000)
        downsampled = DataDownsampler.downsample_waveform(long_waveform, target_samples=1000)
        assert len(downsampled) <= 1000
        
        # Test spectrogram downsampling
        large_spectrogram = np.random.random((1000, 2000))
        downsampled_spec = DataDownsampler.downsample_spectrogram(
            large_spectrogram, target_shape=(100, 200)
        )
        assert downsampled_spec.shape == (100, 200)
        
        # Test detection sampling
        large_detections = pd.DataFrame({
            'peak_freq': np.random.random(5000),
            'amplitude': np.random.random(5000)
        })
        sampled = DataDownsampler.adaptive_detection_sampling(large_detections, max_detections=100)
        assert len(sampled) <= 100


class TestPublicationTools:
    """Test publication-quality plotting tools."""
    
    def test_publication_theme(self):
        """Test publication theme setup."""
        from pampal.viz.publication import PublicationTheme
        
        theme = PublicationTheme('nature')
        assert theme.style == 'nature'
        
        colors = theme.get_colors()
        assert isinstance(colors, dict)
        assert 'primary' in colors
    
    def test_multipanel_figure(self):
        """Test multi-panel figure creation."""
        from pampal.viz.publication import MultipanelFigure
        
        multi_fig = MultipanelFigure((2, 2), figsize=(8, 6))
        assert multi_fig.fig is not None
        
        # Test adding panel labels
        if hasattr(multi_fig, 'axes') and multi_fig.axes.size > 0:
            ax = multi_fig.axes.flat[0] if multi_fig.axes.ndim > 1 else multi_fig.axes[0]
            multi_fig.add_panel_label(ax, 'A')
        
        plt.close(multi_fig.fig)
    
    def test_journal_templates(self):
        """Test journal-specific templates."""
        from pampal.viz.publication import create_journal_template
        
        nature_template = create_journal_template('nature')
        assert 'single_column' in nature_template
        assert 'font_size' in nature_template
        
        science_template = create_journal_template('science')
        assert 'single_column' in science_template


class TestExportUtilities:
    """Test export and file handling utilities."""
    
    def test_plot_exporter(self):
        """Test plot export functionality."""
        from pampal.viz.optimization import PlotExporter
        
        # Create a simple test figure
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot([1, 2, 3], [1, 4, 2])
        ax.set_title('Test Plot')
        
        with tempfile.TemporaryDirectory() as temp_dir:
            filename = os.path.join(temp_dir, 'test_plot')
            
            # Test export
            PlotExporter.export_high_res_figure(fig, filename, formats=['png'])
            
            # Check if file was created
            assert Path(filename + ".png").exists()
        
        plt.close(fig)


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_empty_data_handling(self):
        """Test handling of empty datasets."""
        empty_df = pd.DataFrame()
        
        # Should handle empty data gracefully
        try:
            fig, axes = plot_detection_overview(empty_df)
            assert fig is not None
            plt.close(fig)
        except ValueError:
            # Expected for some functions that require data
            pass
    
    def test_missing_columns(self):
        """Test handling of missing required columns."""
        incomplete_df = pd.DataFrame({
            'time': [1, 2, 3],
            'amplitude': [0.5, 0.7, 0.3]
        })
        
        # Should handle missing columns gracefully
        try:
            fig, axes = plot_detection_overview(incomplete_df)
            assert fig is not None
            plt.close(fig)
        except (ValueError, KeyError):
            # Expected for functions that require specific columns
            pass
    
    def test_invalid_parameters(self):
        """Test handling of invalid parameters."""
        sample_data = np.random.random(1000)
        
        # Test invalid sample rate
        with pytest.raises((ValueError, TypeError)):
            plot_waveform(sample_data, sample_rate=-1)
        
        # Test invalid frequency range
        try:
            fig, ax = plot_spectrogram(sample_data, freq_range=(100000, 10))  # Invalid range
            plt.close(fig)
        except ValueError:
            # Expected for invalid ranges
            pass


# Integration tests
class TestIntegration:
    """Integration tests for complete workflows."""
    
    def test_complete_analysis_workflow(self):
        """Test a complete analysis workflow."""
        # Create synthetic dataset
        n_detections = 200
        np.random.seed(42)
        
        base_time = datetime.now()
        times = [base_time + timedelta(seconds=i * 2 + np.random.random()) 
                for i in range(n_detections)]
        
        detections = pd.DataFrame({
            'UTC': times,
            'detection_type': np.random.choice(['click', 'whistle'], n_detections),
            'peak_freq': 20000 + 25000 * np.random.random(n_detections),
            'duration': 0.002 + 0.008 * np.random.random(n_detections),
            'amplitude': 0.4 + 0.6 * np.random.random(n_detections),
            'latitude': 36.0 + 0.1 * np.random.randn(n_detections),
            'longitude': -121.0 + 0.1 * np.random.randn(n_detections)
        })
        
        study_data = {
            'detections': detections,
            'survey_info': {'name': 'Integration Test Survey'},
            'effort_data': {'total_hours': 100}
        }
        
        # Test various visualization functions
        figures_created = []
        
        try:
            # Detection overview
            fig1, _ = plot_detection_overview(detections)
            figures_created.append(fig1)
            
            # Study overview
            fig2, _ = plot_study_overview(study_data)
            figures_created.append(fig2)
            
            # Temporal patterns
            fig3, _ = plot_temporal_patterns(detections)
            figures_created.append(fig3)
            
            # All figures should be created successfully
            assert len(figures_created) == 3
            
        finally:
            # Clean up
            for fig in figures_created:
                plt.close(fig)


if __name__ == '__main__':
    # Run tests with pytest
    pytest.main([__file__, '-v'])