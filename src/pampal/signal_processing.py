"""
Signal processing module for acoustic analysis in PAMpal.

This module provides core signal processing functionality for analyzing acoustic data
from marine mammal detections, including waveform analysis, spectrogram generation,
and acoustic parameter calculations.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
import warnings
from scipy import signal
from scipy.fft import fft, fftfreq
import struct

from .binary_parser import read_pgdf_header, PamBinaryError
from .calibration import CalibrationFunction, apply_calibration_to_spectrum


class AcousticParameters:
    """Container class for acoustic parameters calculated from detections."""
    
    def __init__(self):
        self.peak_frequency = None
        self.bandwidth = None
        self.duration = None
        self.amplitude = None
        self.rms_amplitude = None
        self.peak_amplitude = None
        self.centroid_frequency = None
        self.q_factor = None
        self.snr = None


def extract_waveform_data(binary_file_path: str, detection_data: pd.DataFrame, 
                         calibration_function: Optional[CalibrationFunction] = None) -> Dict[str, np.ndarray]:
    """
    Extract raw waveform data from binary files for specific detections.
    
    Args:
        binary_file_path: Path to the binary file
        detection_data: DataFrame containing detection information with timestamps
        calibration_function: Optional calibration function to apply to waveforms
        
    Returns:
        Dictionary mapping detection UIDs to waveform arrays
        
    Raises:
        PamBinaryError: If unable to read binary file or extract waveforms
    """
    try:
        # Read binary file header to get basic info
        header = read_pgdf_header(binary_file_path)
        
        waveforms = {}
        
        with open(binary_file_path, 'rb') as f:
            # Skip to data section
            f.seek(header['data_start'])
            
            for idx, detection in detection_data.iterrows():
                try:
                    # Extract waveform for this detection
                    # This is a simplified implementation - real implementation would
                    # need to parse the specific binary format for waveform data
                    waveform = _extract_single_waveform(f, detection, header)
                    
                    if waveform is not None:
                        # Apply calibration if provided
                        if calibration_function is not None:
                            try:
                                from .calibration import apply_calibration_to_waveform
                                waveform = apply_calibration_to_waveform(
                                    waveform, header.get('sample_rate', 192000), calibration_function
                                )
                            except Exception as e:
                                warnings.warn(f"Failed to apply calibration to waveform: {str(e)}")
                        
                        uid = detection.get('UID', f'detection_{idx}')
                        waveforms[str(uid)] = waveform
                        
                except Exception as e:
                    warnings.warn(f"Failed to extract waveform for detection {idx}: {str(e)}")
                    continue
                    
        return waveforms
        
    except Exception as e:
        raise PamBinaryError(f"Error extracting waveforms from {binary_file_path}: {str(e)}")


def _extract_single_waveform(file_handle, detection: pd.Series, header: Dict) -> Optional[np.ndarray]:
    """
    Extract a single waveform from the binary file.
    
    This is a placeholder implementation. The actual implementation would need
    to understand the specific binary format used by each detector type.
    """
    # For now, generate a synthetic waveform based on detection parameters
    # In real implementation, this would read actual waveform data from the binary file
    
    sample_rate = header.get('sample_rate', 192000)  # Default sample rate
    duration = detection.get('duration', 0.001)  # Default 1ms duration
    
    if duration <= 0:
        duration = 0.001
        
    # Generate time vector
    n_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, n_samples)
    
    # Generate a synthetic click-like waveform
    # This would be replaced with actual binary data reading
    frequency = detection.get('peak_freq', 50000)  # Default 50kHz
    amplitude = detection.get('amplitude', 1.0)
    
    # Create a damped sinusoid (typical of echolocation clicks)
    envelope = np.exp(-t * 5000)  # Exponential decay
    waveform = amplitude * envelope * np.sin(2 * np.pi * frequency * t)
    
    # Add some noise
    noise_level = 0.1
    noise = noise_level * np.random.randn(len(waveform))
    waveform += noise
    
    return waveform.astype(np.float32)


def calculate_spectrogram(waveform: np.ndarray, sample_rate: int, 
                         window_size: int = 512, overlap: float = 0.75,
                         window_type: str = 'hann', 
                         calibration_function: Optional[CalibrationFunction] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate spectrogram from a waveform.
    
    Args:
        waveform: 1D numpy array containing audio samples
        sample_rate: Sample rate in Hz
        window_size: FFT window size (power of 2 recommended)
        overlap: Overlap between windows (0-1)
        window_type: Type of window function ('hann', 'hamming', 'blackman', etc.)
        calibration_function: Optional calibration function to apply to spectrum
        
    Returns:
        Tuple of (spectrogram_db, frequency_bins, time_bins)
        - spectrogram_db: 2D array of power spectral density in dB
        - frequency_bins: 1D array of frequency values
        - time_bins: 1D array of time values
    """
    # Adjust window size for short signals
    actual_window_size = min(window_size, len(waveform))
    actual_overlap = min(overlap, 0.9)  # Cap overlap to avoid issues
    
    # Ensure noverlap is less than nperseg
    noverlap = int(actual_window_size * actual_overlap)
    if noverlap >= actual_window_size:
        noverlap = max(0, actual_window_size - 1)
    
    # Calculate spectrogram using scipy
    frequencies, times, Sxx = signal.spectrogram(
        waveform, 
        fs=sample_rate,
        window=window_type,
        nperseg=actual_window_size,
        noverlap=noverlap,
        scaling='density'
    )
    
    # Convert to dB scale
    Sxx_db = 10 * np.log10(np.maximum(Sxx, 1e-12))  # Avoid log(0)
    
    # Apply calibration if provided
    if calibration_function is not None:
        try:
            # Apply calibration to each time slice
            for i in range(Sxx_db.shape[1]):
                Sxx_db[:, i] = apply_calibration_to_spectrum(frequencies, Sxx_db[:, i], calibration_function)
        except Exception as e:
            warnings.warn(f"Failed to apply calibration to spectrogram: {str(e)}")
    
    return Sxx_db, frequencies, times


def calculate_click_parameters(waveform: np.ndarray, sample_rate: int, 
                              freq_range: Tuple[float, float] = None,
                              calibration_function: Optional[CalibrationFunction] = None) -> AcousticParameters:
    """
    Calculate acoustic parameters for click-type detections.
    
    Args:
        waveform: 1D numpy array containing the click waveform
        sample_rate: Sample rate in Hz
        freq_range: Optional frequency range (min_freq, max_freq) for analysis
        calibration_function: Optional calibration function to apply to spectrum
        
    Returns:
        AcousticParameters object containing calculated parameters
    """
    params = AcousticParameters()
    
    # Time domain parameters
    params.duration = len(waveform) / sample_rate
    params.peak_amplitude = np.max(np.abs(waveform))
    params.rms_amplitude = np.sqrt(np.mean(waveform**2))
    
    # Frequency domain analysis
    # Calculate power spectral density
    frequencies, psd = signal.welch(waveform, fs=sample_rate, nperseg=min(len(waveform), 512))
    
    # Convert PSD to dB scale
    psd_db = 10 * np.log10(np.maximum(psd, 1e-12))
    
    # Apply calibration if provided
    if calibration_function is not None:
        try:
            psd_db = apply_calibration_to_spectrum(frequencies, psd_db, calibration_function)
        except Exception as e:
            warnings.warn(f"Failed to apply calibration to spectrum: {str(e)}")
    
    # Convert back to linear scale for analysis
    psd = 10**(psd_db / 10)
    
    # Apply frequency range filter if specified
    if freq_range is not None:
        freq_mask = (frequencies >= freq_range[0]) & (frequencies <= freq_range[1])
        frequencies = frequencies[freq_mask]
        psd = psd[freq_mask]
    
    if len(psd) > 0:
        # Peak frequency
        peak_idx = np.argmax(psd)
        params.peak_frequency = frequencies[peak_idx]
        
        # Centroid frequency (weighted average)
        params.centroid_frequency = np.sum(frequencies * psd) / np.sum(psd)
        
        # Bandwidth calculations
        # Find -3dB bandwidth
        peak_power = psd[peak_idx]
        half_power = peak_power / 2
        
        # Find frequencies where power drops to half maximum
        above_half_power = psd >= half_power
        if np.any(above_half_power):
            freq_indices = np.where(above_half_power)[0]
            if len(freq_indices) > 1:
                params.bandwidth = frequencies[freq_indices[-1]] - frequencies[freq_indices[0]]
            else:
                params.bandwidth = 0
        
        # Q factor (quality factor)
        if params.bandwidth > 0:
            params.q_factor = params.peak_frequency / params.bandwidth
    
    # Signal-to-noise ratio estimation
    # Simple approach: compare peak to background level
    if len(waveform) > 100:
        # Use first and last 10% as noise estimate
        noise_samples = int(len(waveform) * 0.1)
        noise_start = waveform[:noise_samples]
        noise_end = waveform[-noise_samples:]
        noise_level = np.sqrt(np.mean(np.concatenate([noise_start, noise_end])**2))
        
        if noise_level > 0:
            params.snr = 20 * np.log10(params.rms_amplitude / noise_level)
    
    return params


def extract_whistle_contour(waveform: np.ndarray, sample_rate: int,
                           min_freq: float = 1000, max_freq: float = 50000,
                           time_resolution: float = 0.001) -> pd.DataFrame:
    """
    Extract frequency contour from whistle-type detections.
    
    Args:
        waveform: 1D numpy array containing the whistle waveform
        sample_rate: Sample rate in Hz
        min_freq: Minimum frequency for contour extraction
        max_freq: Maximum frequency for contour extraction
        time_resolution: Time resolution for contour points (seconds)
        
    Returns:
        DataFrame with columns ['time', 'frequency', 'amplitude'] representing the contour
    """
    # Calculate spectrogram with high time resolution
    window_size = int(sample_rate * time_resolution * 2)  # Window size for desired resolution
    window_size = max(64, min(window_size, len(waveform) // 4))  # Reasonable bounds
    
    Sxx, frequencies, times = calculate_spectrogram(
        waveform, sample_rate, 
        window_size=window_size, 
        overlap=0.9  # High overlap for smooth contour
    )
    
    # Filter frequency range
    freq_mask = (frequencies >= min_freq) & (frequencies <= max_freq)
    frequencies_filtered = frequencies[freq_mask]
    Sxx_filtered = Sxx[freq_mask, :]
    
    contour_data = []
    
    # Extract peak frequency at each time step
    for t_idx, time_val in enumerate(times):
        if t_idx < Sxx_filtered.shape[1]:
            # Find peak frequency at this time
            power_spectrum = Sxx_filtered[:, t_idx]
            
            if len(power_spectrum) > 0:
                peak_idx = np.argmax(power_spectrum)
                peak_freq = frequencies_filtered[peak_idx]
                peak_amplitude = power_spectrum[peak_idx]
                
                # Only include if above threshold
                if peak_amplitude > np.max(power_spectrum) - 20:  # Within 20dB of peak
                    contour_data.append({
                        'time': time_val,
                        'frequency': peak_freq,
                        'amplitude': peak_amplitude
                    })
    
    return pd.DataFrame(contour_data)


def calculate_cepstrum(waveform: np.ndarray, sample_rate: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate cepstrum for echolocation click analysis.
    
    The cepstrum is useful for detecting periodic components in signals,
    particularly for analyzing echolocation clicks with multiple peaks.
    
    Args:
        waveform: 1D numpy array containing the signal
        sample_rate: Sample rate in Hz
        
    Returns:
        Tuple of (cepstrum, quefrency_bins)
        - cepstrum: Real cepstrum values
        - quefrency_bins: Quefrency values (time-like domain)
    """
    # Calculate power spectrum
    fft_signal = fft(waveform)
    power_spectrum = np.abs(fft_signal)**2
    
    # Avoid log(0) by adding small epsilon
    log_power_spectrum = np.log(np.maximum(power_spectrum, 1e-12))
    
    # Calculate cepstrum (IFFT of log power spectrum)
    cepstrum = np.real(fft(log_power_spectrum))
    
    # Generate quefrency bins
    n_samples = len(cepstrum)
    quefrency_bins = np.arange(n_samples) / sample_rate
    
    # Return only the first half (positive quefrencies)
    half_length = n_samples // 2
    return cepstrum[:half_length], quefrency_bins[:half_length]


def calculate_inter_click_intervals(detection_times: np.ndarray) -> Dict[str, Any]:
    """
    Calculate inter-click intervals (ICI) from a series of click detection times.
    
    ICI analysis is critical for species identification as different species
    have characteristic click patterns.
    
    Args:
        detection_times: Array of detection timestamps (in seconds)
        
    Returns:
        Dictionary containing ICI statistics:
        - 'intervals': Array of inter-click intervals
        - 'mean_ici': Mean ICI
        - 'std_ici': Standard deviation of ICI
        - 'median_ici': Median ICI
        - 'ici_cv': Coefficient of variation (std/mean)
        - 'regular_clicks': Boolean indicating if clicks are regular
    """
    if len(detection_times) < 2:
        return {
            'intervals': np.array([]),
            'mean_ici': np.nan,
            'std_ici': np.nan,
            'median_ici': np.nan,
            'ici_cv': np.nan,
            'regular_clicks': False
        }
    
    # Sort times to ensure proper order
    sorted_times = np.sort(detection_times)
    
    # Calculate intervals
    intervals = np.diff(sorted_times)
    
    # Calculate statistics
    mean_ici = np.mean(intervals)
    std_ici = np.std(intervals)
    median_ici = np.median(intervals)
    
    # Coefficient of variation
    ici_cv = std_ici / mean_ici if mean_ici > 0 else np.inf
    
    # Determine if clicks are regular (CV < 0.3 is often considered regular)
    regular_clicks = ici_cv < 0.3 and len(intervals) >= 3
    
    return {
        'intervals': intervals,
        'mean_ici': mean_ici,
        'std_ici': std_ici,
        'median_ici': median_ici,
        'ici_cv': ici_cv,
        'regular_clicks': regular_clicks
    }


def analyze_detection_sequence(detections: pd.DataFrame, 
                              detector_type: str = 'click') -> Dict[str, Any]:
    """
    Comprehensive analysis of a sequence of detections.
    
    Args:
        detections: DataFrame containing detection data with timestamps
        detector_type: Type of detector ('click', 'whistle', 'cepstrum')
        
    Returns:
        Dictionary containing comprehensive analysis results
    """
    results = {
        'detector_type': detector_type,
        'n_detections': len(detections),
        'duration': 0,
        'detection_rate': 0
    }
    
    if len(detections) == 0:
        return results
    
    # Time-based analysis
    if 'UTC' in detections.columns:
        times = pd.to_datetime(detections['UTC'])
        if len(times) > 1:
            duration = (times.max() - times.min()).total_seconds()
            results['duration'] = duration
            results['detection_rate'] = len(detections) / duration if duration > 0 else 0
            
            # Convert to seconds since start for ICI analysis
            time_seconds = (times - times.min()).dt.total_seconds().values
            
            if detector_type.lower() == 'click':
                # ICI analysis for clicks
                ici_results = calculate_inter_click_intervals(time_seconds)
                results.update(ici_results)
    
    # Frequency-based analysis if frequency data available
    if 'peak_freq' in detections.columns:
        freq_data = detections['peak_freq'].dropna()
        if len(freq_data) > 0:
            results['freq_stats'] = {
                'mean_freq': np.mean(freq_data),
                'std_freq': np.std(freq_data),
                'min_freq': np.min(freq_data),
                'max_freq': np.max(freq_data),
                'freq_range': np.max(freq_data) - np.min(freq_data)
            }
    
    # Amplitude-based analysis if amplitude data available
    if 'amplitude' in detections.columns:
        amp_data = detections['amplitude'].dropna()
        if len(amp_data) > 0:
            results['amplitude_stats'] = {
                'mean_amplitude': np.mean(amp_data),
                'std_amplitude': np.std(amp_data),
                'min_amplitude': np.min(amp_data),
                'max_amplitude': np.max(amp_data)
            }
    
    return results
