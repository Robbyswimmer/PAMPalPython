#!/usr/bin/env python3
"""
Integration tests using real example data.

This test suite validates the complete PAMpal Python system using the
converted real data examples. It tests data loading, visualization,
signal processing, and analysis workflows end-to-end.
"""

import unittest
import sys
import os
import tempfile
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt

# Add the package to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pampal.data import (
    load_test_click, load_test_whistle, load_test_cepstrum,
    load_test_gpl, load_example_study, load_all_test_data,
    list_available_datasets, get_dataset_info, create_sample_analysis_data,
    get_click_waveform, get_whistle_contour, get_cepstrum_data,
    get_gpl_detection_data, get_study_detections, DataLoadError
)

from pampal.viz import (
    plot_waveform, plot_spectrogram, plot_detection_overview,
    plot_study_overview, set_style, reset_style,
    MultipanelFigure, PublicationTheme
)

from pampal.signal_processing import calculate_spectrogram


class TestRealDataLoading(unittest.TestCase):
    """Test real data loading functionality."""
    
    def test_individual_dataset_loading(self):
        """Test loading each dataset individually."""
        # Test click data
        click_data = load_test_click()
        self.assertIn('wave', click_data)
        self.assertIn('sr', click_data)
        
        wave = np.array(click_data['wave'])
        self.assertEqual(wave.shape[1], 2)  # Two channels
        self.assertEqual(click_data['sr'], 500000)  # 500 kHz
        
        # Test whistle data
        whistle_data = load_test_whistle()
        self.assertIn('freq', whistle_data)
        self.assertIn('time', whistle_data)
        
        freq = np.array(whistle_data['freq'])
        time = np.array(whistle_data['time'])
        self.assertEqual(len(freq), len(time))
        
        # Test cepstrum data
        ceps_data = load_test_cepstrum()
        self.assertIn('cepstrum', ceps_data)
        self.assertIn('quefrency', ceps_data)
        self.assertIn('time', ceps_data)
        
        # Test GPL data
        gpl_data = load_test_gpl()
        self.assertIn('energy', gpl_data)
        self.assertIn('freq', gpl_data)
        self.assertIn('time', gpl_data)
        self.assertIn('points', gpl_data)
        
        # Test study data
        study_data = load_example_study()
        self.assertIn('events', study_data)
        self.assertIn('study', study_data)
    
    def test_bulk_data_loading(self):
        """Test loading all datasets at once."""
        all_data = load_all_test_data()
        
        expected_keys = {'click', 'whistle', 'cepstrum', 'gpl', 'study'}
        self.assertEqual(set(all_data.keys()), expected_keys)
        
        for key, data in all_data.items():
            self.assertIsInstance(data, dict)
            self.assertGreater(len(data), 0)
    
    def test_dataset_info_functions(self):
        """Test dataset information and listing functions."""
        # Test dataset listing
        datasets = list_available_datasets()
        self.assertIsInstance(datasets, dict)
        self.assertGreater(len(datasets), 0)
        
        # Test dataset info for each available dataset
        for dataset_name in datasets.keys():
            info = get_dataset_info(dataset_name)
            self.assertIsInstance(info, dict)
            self.assertIn('description', info)
    
    def test_convenience_functions(self):
        """Test convenience data access functions."""
        # Test click convenience function
        wave, sr = get_click_waveform()
        self.assertIsInstance(wave, np.ndarray)
        self.assertEqual(wave.shape[1], 2)
        self.assertEqual(sr, 500000)
        
        # Test whistle convenience function
        freq, time = get_whistle_contour()
        self.assertIsInstance(freq, np.ndarray)
        self.assertIsInstance(time, np.ndarray)
        self.assertEqual(len(freq), len(time))
        
        # Test cepstrum convenience function
        cepstrum, quefrency, time = get_cepstrum_data()
        self.assertIsInstance(cepstrum, np.ndarray)
        self.assertIsInstance(quefrency, np.ndarray)
        self.assertIsInstance(time, np.ndarray)
        
        # Test GPL convenience function
        energy, freq, time, points = get_gpl_detection_data()
        self.assertIsInstance(energy, np.ndarray)
        self.assertIsInstance(freq, np.ndarray)
        self.assertIsInstance(time, np.ndarray)
        
        # Test study detections function
        detections = get_study_detections()
        self.assertIsInstance(detections, pd.DataFrame)
    
    def test_sample_analysis_data_creation(self):
        """Test creation of sample analysis data structure."""
        analysis_data = create_sample_analysis_data()
        
        expected_keys = {'waveforms', 'contours', 'spectral_analysis', 'study_data', 'metadata'}
        self.assertEqual(set(analysis_data.keys()), expected_keys)
        
        # Check waveforms structure
        self.assertIn('click_example', analysis_data['waveforms'])
        click_example = analysis_data['waveforms']['click_example']
        self.assertIn('data', click_example)
        self.assertIn('sr', click_example)
        self.assertIn('type', click_example)
        
        # Check contours structure
        self.assertIn('whistle_example', analysis_data['contours'])
        whistle_example = analysis_data['contours']['whistle_example']
        self.assertIn('freq', whistle_example)
        self.assertIn('time', whistle_example)
        self.assertIn('type', whistle_example)


class TestRealDataVisualization(unittest.TestCase):
    """Test visualization with real data."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
        
        # Load test data
        self.click_data = load_test_click()
        self.whistle_data = load_test_whistle()
        self.ceps_data = load_test_cepstrum()
    
    def tearDown(self):
        """Clean up test environment."""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.temp_dir)
        plt.close('all')
        reset_style()
    
    def test_waveform_plotting(self):
        """Test waveform plotting with real click data."""
        wave = np.array(self.click_data['wave'])
        sr = self.click_data['sr']
        
        # Test basic waveform plot
        fig, ax = plot_waveform(wave[:, 0], sample_rate=sr, title="Test Click")
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)
        
        # Save and verify file creation
        plt.savefig('test_waveform.png')
        self.assertTrue(os.path.exists('test_waveform.png'))
        
        plt.close(fig)
    
    def test_spectrogram_plotting(self):
        """Test spectrogram plotting with real click data."""
        wave = np.array(self.click_data['wave'])
        sr = self.click_data['sr']
        
        # Test spectrogram plot
        fig, ax = plot_spectrogram(wave[:, 0], sample_rate=sr, title="Test Spectrogram")
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)
        
        # Save and verify file creation
        plt.savefig('test_spectrogram.png')
        self.assertTrue(os.path.exists('test_spectrogram.png'))
        
        plt.close(fig)
    
    def test_multipanel_figure(self):
        """Test multipanel figure creation."""
        # Create a 2x2 multipanel figure
        mp_fig = MultipanelFigure((2, 2), figsize=(10, 8))
        self.assertIsNotNone(mp_fig)
        self.assertEqual(mp_fig.axes.shape, (2, 2))
        
        # Add some test plots
        wave = np.array(self.click_data['wave'])
        time = np.linspace(0, len(wave)/self.click_data['sr'], len(wave))
        
        ax1 = mp_fig.axes[0, 0]
        ax1.plot(time, wave[:, 0])
        ax1.set_title('Channel 1')
        mp_fig.add_panel_label(ax1, 'A')
        
        ax2 = mp_fig.axes[0, 1]
        ax2.plot(time, wave[:, 1])
        ax2.set_title('Channel 2')
        mp_fig.add_panel_label(ax2, 'B')
        
        plt.savefig('test_multipanel.png')
        self.assertTrue(os.path.exists('test_multipanel.png'))
        
        plt.close('all')
    
    def test_publication_theme(self):
        """Test publication theme functionality."""
        pub_theme = PublicationTheme('nature')
        
        # Test that theme was created successfully
        self.assertEqual(pub_theme.style, 'nature')
        
        # Test color scheme
        colors = pub_theme.get_colors()
        self.assertIsInstance(colors, dict)
        self.assertIn('primary', colors)
        
        # Test creating a plot with publication theme
        wave = np.array(self.click_data['wave'])
        sr = self.click_data['sr']
        
        fig, ax = plot_waveform(wave[:, 0], sample_rate=sr, title="Publication Plot")
        
        plt.savefig('test_publication.png', dpi=300)
        self.assertTrue(os.path.exists('test_publication.png'))
        
        plt.close(fig)


class TestRealDataSignalProcessing(unittest.TestCase):
    """Test signal processing with real data."""
    
    def test_spectrogram_calculation(self):
        """Test spectrogram calculation with real click data."""
        click_data = load_test_click()
        wave = np.array(click_data['wave'])
        sr = click_data['sr']
        
        # Calculate spectrogram
        Sxx_db, frequencies, times = calculate_spectrogram(
            wave[:, 0], sr, window_size=256, overlap=0.75
        )
        
        # Verify output shapes and ranges
        self.assertIsInstance(Sxx_db, np.ndarray)
        self.assertIsInstance(frequencies, np.ndarray)
        self.assertIsInstance(times, np.ndarray)
        
        self.assertEqual(len(frequencies), Sxx_db.shape[0])
        self.assertEqual(len(times), Sxx_db.shape[1])
        
        # Verify frequency range is reasonable
        self.assertGreater(frequencies.max(), 0)
        self.assertLessEqual(frequencies.max(), sr/2)  # Nyquist limit
        
        # Verify time range matches signal duration
        expected_duration = len(wave) / sr
        self.assertLessEqual(times.max(), expected_duration)


class TestRealDataWorkflows(unittest.TestCase):
    """Test complete analysis workflows with real data."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
    
    def tearDown(self):
        """Clean up test environment."""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.temp_dir)
        plt.close('all')
        reset_style()
    
    def test_click_analysis_workflow(self):
        """Test complete click analysis workflow."""
        # Load click data
        click_data = load_test_click()
        wave = np.array(click_data['wave'])
        sr = click_data['sr']
        
        # Step 1: Basic waveform plot
        fig1, ax1 = plot_waveform(wave[:, 0], sample_rate=sr, title="Click Waveform")
        plt.savefig('workflow_waveform.png')
        self.assertTrue(os.path.exists('workflow_waveform.png'))
        plt.close(fig1)
        
        # Step 2: Spectrogram
        fig2, ax2 = plot_spectrogram(wave[:, 0], sample_rate=sr, title="Click Spectrogram")
        plt.savefig('workflow_spectrogram.png')
        self.assertTrue(os.path.exists('workflow_spectrogram.png'))
        plt.close(fig2)
        
        # Step 3: Multi-channel comparison
        fig3, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        time_ms = np.linspace(0, len(wave)/sr*1000, len(wave))
        
        for i, (ax, label) in enumerate(zip(axes, ['Channel 1', 'Channel 2'])):
            ax.plot(time_ms, wave[:, i], 'b-', linewidth=1)
            ax.set_ylabel('Amplitude')
            ax.set_title(f'{label}')
            ax.grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Time (ms)')
        plt.tight_layout()
        plt.savefig('workflow_multichannel.png')
        self.assertTrue(os.path.exists('workflow_multichannel.png'))
        plt.close(fig3)
    
    def test_whistle_analysis_workflow(self):
        """Test complete whistle analysis workflow."""
        # Load whistle data
        whistle_data = load_test_whistle()
        freq = np.array(whistle_data['freq'])
        time = np.array(whistle_data['time'])
        
        # Basic contour plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(time, freq/1000, 'r-', linewidth=2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (kHz)')
        ax.set_title('Whistle Frequency Contour')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('workflow_whistle.png')
        self.assertTrue(os.path.exists('workflow_whistle.png'))
        plt.close(fig)
    
    def test_study_analysis_workflow(self):
        """Test complete study analysis workflow."""
        # Load study data
        study_data = load_example_study()
        detections = get_study_detections()
        
        if not detections.empty:
            # Create study overview plot
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot detection timeline if we have UTC data
            if 'UTC' in detections.columns:
                detection_times = pd.to_datetime(detections['UTC'])
                detection_types = detections['detector_type']
                
                for i, det_type in enumerate(detection_types.unique()):
                    mask = detection_types == det_type
                    times = detection_times[mask]
                    y_pos = [i] * len(times)
                    ax.scatter(times, y_pos, label=det_type, alpha=0.7)
                
                ax.set_ylabel('Detection Type')
                ax.set_xlabel('Time')
                ax.set_title('Study Detection Timeline')
                ax.legend()
                ax.grid(True, alpha=0.3)
            else:
                # Fallback: simple detection count plot
                detection_counts = detections['detector_type'].value_counts()
                ax.bar(range(len(detection_counts)), detection_counts.values)
                ax.set_xticks(range(len(detection_counts)))
                ax.set_xticklabels(detection_counts.index)
                ax.set_ylabel('Count')
                ax.set_title('Detection Type Summary')
            
            plt.tight_layout()
            plt.savefig('workflow_study.png')
            self.assertTrue(os.path.exists('workflow_study.png'))
            plt.close(fig)
    
    def test_complete_workflow_integration(self):
        """Test the complete workflow using all datasets."""
        # Load all data
        all_data = load_all_test_data()
        analysis_data = create_sample_analysis_data()
        
        # Verify we can access all components
        self.assertIn('click', all_data)
        self.assertIn('whistle', all_data)
        self.assertIn('cepstrum', all_data)
        self.assertIn('gpl', all_data)
        self.assertIn('study', all_data)
        
        # Verify analysis data structure
        self.assertIn('waveforms', analysis_data)
        self.assertIn('contours', analysis_data)
        self.assertIn('spectral_analysis', analysis_data)
        self.assertIn('study_data', analysis_data)
        
        # Create a summary plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        # Click waveform
        click_wave = np.array(all_data['click']['wave'])
        click_sr = all_data['click']['sr']
        time_ms = np.linspace(0, len(click_wave)/click_sr*1000, len(click_wave))
        ax1.plot(time_ms, click_wave[:, 0], 'b-', linewidth=1)
        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('Amplitude')
        ax1.set_title('Click Waveform')
        ax1.grid(True, alpha=0.3)
        
        # Whistle contour
        whistle_freq = np.array(all_data['whistle']['freq'])
        whistle_time = np.array(all_data['whistle']['time'])
        ax2.plot(whistle_time, whistle_freq/1000, 'r-', linewidth=2)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Frequency (kHz)')
        ax2.set_title('Whistle Contour')
        ax2.grid(True, alpha=0.3)
        
        # Cepstrogram
        ceps_data = np.array(all_data['cepstrum']['cepstrum'])
        ceps_time = np.array(all_data['cepstrum']['time'])
        ceps_quefrency = np.array(all_data['cepstrum']['quefrency'])
        im = ax3.imshow(ceps_data, aspect='auto', origin='lower',
                       extent=[ceps_time[0]*1000, ceps_time[-1]*1000,
                              ceps_quefrency[0]*1e6, ceps_quefrency[-1]*1e6],
                       cmap='hot')
        ax3.set_xlabel('Time (ms)')
        ax3.set_ylabel('Quefrency (μs)')
        ax3.set_title('Cepstrogram')
        ax3.set_ylim(0, 100)
        
        # Study detections summary
        detections = get_study_detections()
        if not detections.empty:
            detection_counts = detections['detector_type'].value_counts()
            bars = ax4.bar(range(len(detection_counts)), detection_counts.values,
                          color=['blue', 'red'], alpha=0.7)
            ax4.set_xticks(range(len(detection_counts)))
            ax4.set_xticklabels(detection_counts.index, rotation=45)
            ax4.set_ylabel('Count')
            ax4.set_title('Detection Summary')
        else:
            ax4.text(0.5, 0.5, 'No detections found', ha='center', va='center',
                    transform=ax4.transAxes)
            ax4.set_title('Detection Summary')
        
        plt.suptitle('PAMpal Python - Real Data Integration Test', fontsize=14)
        plt.tight_layout()
        plt.savefig('workflow_complete_integration.png', dpi=300, bbox_inches='tight')
        self.assertTrue(os.path.exists('workflow_complete_integration.png'))
        plt.close(fig)


class TestErrorHandling(unittest.TestCase):
    """Test error handling in data loading."""
    
    def test_nonexistent_dataset(self):
        """Test handling of requests for non-existent datasets."""
        with self.assertRaises(ValueError):
            get_dataset_info('nonexistent_dataset')
    
    def test_data_loading_error_handling(self):
        """Test graceful handling of data loading errors."""
        # This test would need to simulate file system errors
        # For now, we just verify the exception type is properly defined
        self.assertTrue(issubclass(DataLoadError, Exception))


if __name__ == '__main__':
    # Run the integration tests
    unittest.main(verbosity=2)