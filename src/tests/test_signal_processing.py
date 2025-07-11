"""
Unit tests for the signal_processing module.

This module tests all signal processing functionality including waveform analysis,
spectrogram generation, acoustic parameter calculations, and ICI analysis.
"""
import unittest
import numpy as np
import pandas as pd
import tempfile
import os
from unittest.mock import patch, MagicMock

from pampal.signal_processing import (
    AcousticParameters,
    extract_waveform_data,
    calculate_spectrogram,
    calculate_click_parameters,
    extract_whistle_contour,
    calculate_cepstrum,
    calculate_inter_click_intervals,
    analyze_detection_sequence
)


class TestAcousticParameters(unittest.TestCase):
    """Test the AcousticParameters container class."""
    
    def test_initialization(self):
        """Test AcousticParameters initialization."""
        params = AcousticParameters()
        
        # Check that all parameters are initialized to None
        self.assertIsNone(params.peak_frequency)
        self.assertIsNone(params.bandwidth)
        self.assertIsNone(params.duration)
        self.assertIsNone(params.amplitude)
        self.assertIsNone(params.rms_amplitude)
        self.assertIsNone(params.peak_amplitude)
        self.assertIsNone(params.centroid_frequency)
        self.assertIsNone(params.q_factor)
        self.assertIsNone(params.snr)


class TestWaveformExtraction(unittest.TestCase):
    """Test waveform extraction from binary files."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock detection data
        self.detection_data = pd.DataFrame({
            'UID': [1001, 1002, 1003],
            'UTC': pd.to_datetime(['2022-01-01 12:00:00', 
                                  '2022-01-01 12:00:01', 
                                  '2022-01-01 12:00:02']),
            'peak_freq': [45000, 50000, 48000],
            'amplitude': [0.8, 1.0, 0.9],
            'duration': [0.001, 0.0015, 0.0012]
        })
    
    @patch('pampal.signal_processing.read_pgdf_header')
    def test_extract_waveform_data(self, mock_read_header):
        """Test waveform extraction from binary files."""
        # Mock header data
        mock_read_header.return_value = {
            'data_start': 100,
            'sample_rate': 192000
        }
        
        # Create a temporary binary file
        with tempfile.NamedTemporaryFile(suffix='.pgdf', delete=False) as tmp_file:
            # Write some dummy binary data
            tmp_file.write(b'\x00' * 200)  # Header space
            tmp_file.write(b'\x01\x02\x03\x04' * 1000)  # Dummy waveform data
            tmp_file_path = tmp_file.name
        
        try:
            # Test waveform extraction
            waveforms = extract_waveform_data(tmp_file_path, self.detection_data)
            
            # Check that we got waveforms for each detection
            self.assertEqual(len(waveforms), 3)
            self.assertIn('1001', waveforms)
            self.assertIn('1002', waveforms)
            self.assertIn('1003', waveforms)
            
            # Check that waveforms are numpy arrays
            for uid, waveform in waveforms.items():
                self.assertIsInstance(waveform, np.ndarray)
                self.assertGreater(len(waveform), 0)
                
        finally:
            os.unlink(tmp_file_path)


class TestSpectrogramCalculation(unittest.TestCase):
    """Test spectrogram calculation functionality."""
    
    def test_calculate_spectrogram(self):
        """Test spectrogram calculation from waveform."""
        # Create a test signal: sine wave at 1000 Hz
        sample_rate = 8000
        duration = 0.1  # 100ms
        frequency = 1000
        
        t = np.linspace(0, duration, int(sample_rate * duration))
        waveform = np.sin(2 * np.pi * frequency * t)
        
        # Calculate spectrogram
        Sxx, frequencies, times = calculate_spectrogram(waveform, sample_rate)
        
        # Check output shapes and types
        self.assertIsInstance(Sxx, np.ndarray)
        self.assertIsInstance(frequencies, np.ndarray)
        self.assertIsInstance(times, np.ndarray)
        
        # Check that dimensions make sense
        self.assertEqual(Sxx.shape[0], len(frequencies))
        self.assertEqual(Sxx.shape[1], len(times))
        
        # Check frequency range
        self.assertGreaterEqual(frequencies[0], 0)
        self.assertLessEqual(frequencies[-1], sample_rate / 2)
        
        # Check that peak frequency is near our test frequency
        freq_peak_idx = np.unravel_index(np.argmax(Sxx), Sxx.shape)[0]
        peak_frequency = frequencies[freq_peak_idx]
        self.assertLess(abs(peak_frequency - frequency), 100)  # Within 100 Hz
    
    def test_calculate_spectrogram_parameters(self):
        """Test spectrogram with different parameters."""
        # Create test signal
        sample_rate = 16000
        t = np.linspace(0, 0.05, int(sample_rate * 0.05))
        waveform = np.sin(2 * np.pi * 2000 * t)
        
        # Test different window sizes
        for window_size in [128, 256, 512]:
            Sxx, frequencies, times = calculate_spectrogram(
                waveform, sample_rate, window_size=window_size
            )
            self.assertGreater(len(frequencies), 0)
            self.assertGreater(len(times), 0)
        
        # Test different overlap values
        for overlap in [0.5, 0.75, 0.9]:
            Sxx, frequencies, times = calculate_spectrogram(
                waveform, sample_rate, overlap=overlap
            )
            self.assertGreater(len(times), 0)


class TestClickParameters(unittest.TestCase):
    """Test click parameter calculations."""
    
    def test_calculate_click_parameters(self):
        """Test calculation of click acoustic parameters."""
        # Create a synthetic click: damped sinusoid
        sample_rate = 192000
        duration = 0.001  # 1ms
        frequency = 50000  # 50kHz
        
        t = np.linspace(0, duration, int(sample_rate * duration))
        envelope = np.exp(-t * 5000)  # Exponential decay
        waveform = envelope * np.sin(2 * np.pi * frequency * t)
        
        # Calculate parameters
        params = calculate_click_parameters(waveform, sample_rate)
        
        # Check that parameters are calculated
        self.assertIsNotNone(params.duration)
        self.assertIsNotNone(params.peak_amplitude)
        self.assertIsNotNone(params.rms_amplitude)
        self.assertIsNotNone(params.peak_frequency)
        self.assertIsNotNone(params.centroid_frequency)
        
        # Check reasonable values
        self.assertAlmostEqual(params.duration, duration, places=4)
        self.assertGreater(params.peak_amplitude, 0)
        self.assertGreater(params.rms_amplitude, 0)
        self.assertLess(params.rms_amplitude, params.peak_amplitude)
        
        # Peak frequency should be near our test frequency
        self.assertLess(abs(params.peak_frequency - frequency), 5000)
    
    def test_calculate_click_parameters_with_freq_range(self):
        """Test click parameters with frequency range restriction."""
        # Create test signal with multiple frequency components
        sample_rate = 192000
        duration = 0.001
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Signal with 20kHz and 60kHz components
        waveform = (np.sin(2 * np.pi * 20000 * t) + 
                   2 * np.sin(2 * np.pi * 60000 * t))  # 60kHz is stronger
        
        # Test with frequency range that excludes 60kHz
        params = calculate_click_parameters(waveform, sample_rate, 
                                          freq_range=(10000, 30000))
        
        # Peak frequency should be around 20kHz, not 60kHz
        self.assertLess(params.peak_frequency, 30000)
        self.assertGreater(params.peak_frequency, 15000)


class TestWhistleContour(unittest.TestCase):
    """Test whistle contour extraction."""
    
    def test_extract_whistle_contour(self):
        """Test extraction of frequency contour from whistle."""
        # Create a synthetic whistle with frequency sweep
        sample_rate = 48000
        duration = 0.5  # 500ms
        
        t = np.linspace(0, duration, int(sample_rate * duration))
        # Frequency sweep from 5kHz to 15kHz
        freq_sweep = 5000 + 10000 * t / duration
        waveform = np.sin(2 * np.pi * freq_sweep * t)
        
        # Extract contour
        contour = extract_whistle_contour(waveform, sample_rate)
        
        # Check output format
        self.assertIsInstance(contour, pd.DataFrame)
        self.assertIn('time', contour.columns)
        self.assertIn('frequency', contour.columns)
        self.assertIn('amplitude', contour.columns)
        
        # Check that we got reasonable number of points
        self.assertGreater(len(contour), 10)
        
        # Check that frequencies are in reasonable range
        self.assertGreater(contour['frequency'].min(), 3000)
        self.assertLess(contour['frequency'].max(), 25000)  # Allow for some spectral leakage
        
        # Check that time increases monotonically
        self.assertTrue(contour['time'].is_monotonic_increasing)


class TestCepstrum(unittest.TestCase):
    """Test cepstrum calculation."""
    
    def test_calculate_cepstrum(self):
        """Test cepstrum calculation."""
        # Create a signal with periodic structure
        sample_rate = 48000
        duration = 0.01  # 10ms
        
        t = np.linspace(0, duration, int(sample_rate * duration))
        # Signal with fundamental at 1000Hz and harmonics
        waveform = (np.sin(2 * np.pi * 1000 * t) + 
                   0.5 * np.sin(2 * np.pi * 2000 * t) +
                   0.25 * np.sin(2 * np.pi * 3000 * t))
        
        # Calculate cepstrum
        cepstrum, quefrency = calculate_cepstrum(waveform, sample_rate)
        
        # Check output format
        self.assertIsInstance(cepstrum, np.ndarray)
        self.assertIsInstance(quefrency, np.ndarray)
        self.assertEqual(len(cepstrum), len(quefrency))
        
        # Check that quefrency values are reasonable
        self.assertGreaterEqual(quefrency[0], 0)
        self.assertGreater(quefrency[-1], quefrency[0])


class TestInterClickIntervals(unittest.TestCase):
    """Test inter-click interval calculations."""
    
    def test_calculate_inter_click_intervals(self):
        """Test ICI calculation with regular clicks."""
        # Create regular click times (every 50ms)
        click_times = np.array([0.0, 0.05, 0.10, 0.15, 0.20, 0.25])
        
        # Calculate ICI
        ici_results = calculate_inter_click_intervals(click_times)
        
        # Check output format
        self.assertIn('intervals', ici_results)
        self.assertIn('mean_ici', ici_results)
        self.assertIn('std_ici', ici_results)
        self.assertIn('median_ici', ici_results)
        self.assertIn('ici_cv', ici_results)
        self.assertIn('regular_clicks', ici_results)
        
        # Check calculated values
        self.assertAlmostEqual(ici_results['mean_ici'], 0.05, places=3)
        self.assertAlmostEqual(ici_results['median_ici'], 0.05, places=3)
        self.assertLess(ici_results['std_ici'], 0.01)  # Should be very small for regular clicks
        self.assertTrue(ici_results['regular_clicks'])  # Should be detected as regular
    
    def test_calculate_inter_click_intervals_irregular(self):
        """Test ICI calculation with irregular clicks."""
        # Create irregular click times
        click_times = np.array([0.0, 0.03, 0.08, 0.15, 0.18, 0.35])
        
        # Calculate ICI
        ici_results = calculate_inter_click_intervals(click_times)
        
        # Check that irregularity is detected
        self.assertGreater(ici_results['ici_cv'], 0.3)  # High coefficient of variation
        self.assertFalse(ici_results['regular_clicks'])  # Should not be regular
    
    def test_calculate_inter_click_intervals_insufficient_data(self):
        """Test ICI calculation with insufficient data."""
        # Test with single click
        single_click = np.array([0.0])
        ici_results = calculate_inter_click_intervals(single_click)
        
        self.assertEqual(len(ici_results['intervals']), 0)
        self.assertTrue(np.isnan(ici_results['mean_ici']))
        self.assertFalse(ici_results['regular_clicks'])


class TestDetectionSequenceAnalysis(unittest.TestCase):
    """Test comprehensive detection sequence analysis."""
    
    def test_analyze_detection_sequence_clicks(self):
        """Test analysis of click detection sequence."""
        # Create mock click detection data
        detections = pd.DataFrame({
            'UTC': pd.to_datetime([
                '2022-01-01 12:00:00.000',
                '2022-01-01 12:00:00.050',
                '2022-01-01 12:00:00.100',
                '2022-01-01 12:00:00.150',
                '2022-01-01 12:00:00.200'
            ]),
            'peak_freq': [45000, 47000, 46000, 48000, 46500],
            'amplitude': [0.8, 0.9, 0.7, 1.0, 0.85]
        })
        
        # Analyze sequence
        results = analyze_detection_sequence(detections, detector_type='click')
        
        # Check basic statistics
        self.assertEqual(results['n_detections'], 5)
        self.assertEqual(results['detector_type'], 'click')
        self.assertAlmostEqual(results['duration'], 0.2, places=3)
        self.assertAlmostEqual(results['detection_rate'], 25.0, places=1)  # 5 clicks in 0.2s
        
        # Check ICI analysis
        self.assertIn('mean_ici', results)
        self.assertIn('regular_clicks', results)
        self.assertAlmostEqual(results['mean_ici'], 0.05, places=3)
        
        # Check frequency statistics
        self.assertIn('freq_stats', results)
        freq_stats = results['freq_stats']
        self.assertAlmostEqual(freq_stats['mean_freq'], 46500, places=0)
        self.assertEqual(freq_stats['min_freq'], 45000)
        self.assertEqual(freq_stats['max_freq'], 48000)
    
    def test_analyze_detection_sequence_empty(self):
        """Test analysis with empty detection sequence."""
        empty_detections = pd.DataFrame()
        
        results = analyze_detection_sequence(empty_detections)
        
        self.assertEqual(results['n_detections'], 0)
        self.assertEqual(results['duration'], 0)
        self.assertEqual(results['detection_rate'], 0)


if __name__ == '__main__':
    unittest.main()
