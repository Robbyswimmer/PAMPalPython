"""
Export and performance optimization utilities for PAMpal visualizations.

This module provides caching mechanisms, memory management tools,
and optimized export functions for handling large acoustic datasets.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
import warnings
import hashlib
import pickle
import os
import gc
import psutil
from functools import wraps
import time

try:
    from pathlib import Path
except ImportError:
    # Python 2.7 compatibility
    class Path(object):
        def __init__(self, path):
            self.path = str(path)
        
        def exists(self):
            return os.path.exists(self.path)
        
        def mkdir(self, exist_ok=False):
            try:
                if not exist_ok or not os.path.exists(self.path):
                    os.makedirs(self.path)
            except OSError:
                if not exist_ok:
                    raise
        
        def __truediv__(self, other):
            return Path(os.path.join(self.path, str(other)))
        
        def __div__(self, other):  # Python 2.7 compatibility
            return self.__truediv__(other)
        
        def with_suffix(self, suffix):
            base = os.path.splitext(self.path)[0]
            return Path(base + suffix)
        
        def glob(self, pattern):
            import glob as glob_module
            pattern_path = os.path.join(self.path, pattern)
            for match in glob_module.glob(pattern_path):
                yield Path(match)
        
        @property
        def name(self):
            return os.path.basename(self.path)
        
        @property
        def suffix(self):
            return os.path.splitext(self.path)[1]
        
        def is_file(self):
            return os.path.isfile(self.path)
        
        def unlink(self):
            if os.path.exists(self.path):
                os.remove(self.path)
        
        def stat(self):
            class StatResult:
                def __init__(self, stat_result):
                    self.st_size = stat_result.st_size
            return StatResult(os.stat(self.path))
        
        def __str__(self):
            return self.path


class VisualizationCache:
    """Caching system for computed visualization data."""
    
    def __init__(self, cache_dir: str = '.pampal_cache', max_size_mb: int = 500):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_size_mb = max_size_mb
        self.cache_info = self._load_cache_info()
    
    def _load_cache_info(self) -> Dict[str, Dict[str, Any]]:
        """Load cache metadata."""
        info_file = self.cache_dir / 'cache_info.pkl'
        if info_file.exists():
            try:
                with open(info_file, 'rb') as f:
                    return pickle.load(f)
            except Exception:
                return {}
        return {}
    
    def _save_cache_info(self):
        """Save cache metadata."""
        info_file = self.cache_dir / 'cache_info.pkl'
        with open(info_file, 'wb') as f:
            pickle.dump(self.cache_info, f)
    
    def _get_cache_key(self, data: Any, params: Dict[str, Any]) -> str:
        """Generate cache key from data and parameters."""
        # Create a hash from data shape/type and parameters
        if isinstance(data, np.ndarray):
            data_repr = "array_{}_{}_{}" .format(data.shape, data.dtype, data.size)
            if data.size < 1000:  # For small arrays, include actual data
                data_repr += "_{}".format(hashlib.md5(data.tobytes()).hexdigest()[:8])
        elif isinstance(data, pd.DataFrame):
            data_repr = "df_{}_{}_{}" .format(len(data), list(data.columns), data.dtypes.to_dict())
        else:
            data_repr = str(type(data))
        
        params_repr = str(sorted(params.items()))
        combined = "{}_{}".format(data_repr, params_repr)
        
        return hashlib.md5(combined.encode()).hexdigest()
    
    def get(self, data: Any, params: Dict[str, Any]) -> Optional[Any]:
        """Retrieve cached result."""
        cache_key = self._get_cache_key(data, params)
        cache_file = self.cache_dir / "{}.pkl".format(cache_key)
        
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    result = pickle.load(f)
                
                # Update access time
                self.cache_info[cache_key]['last_accessed'] = time.time()
                self._save_cache_info()
                
                return result
            except Exception as e:
                warnings.warn("Cache read error: {}".format(e))
                # Remove corrupted cache file
                try:
                    cache_file.unlink()
                except OSError:
                    pass  # File might not exist
                self.cache_info.pop(cache_key, None)
        
        return None
    
    def set(self, data: Any, params: Dict[str, Any], result: Any):
        """Store result in cache."""
        cache_key = self._get_cache_key(data, params)
        cache_file = self.cache_dir / "{}.pkl".format(cache_key)
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
            
            # Update cache info
            file_size = cache_file.stat().st_size
            self.cache_info[cache_key] = {
                'size_bytes': file_size,
                'created': time.time(),
                'last_accessed': time.time(),
                'params': params
            }
            
            self._save_cache_info()
            self._cleanup_cache()
            
        except Exception as e:
            warnings.warn("Cache write error: {}".format(e))
    
    def _cleanup_cache(self):
        """Remove old cache entries if size limit exceeded."""
        total_size = sum(info['size_bytes'] for info in self.cache_info.values())
        max_size_bytes = self.max_size_mb * 1024 * 1024
        
        if total_size > max_size_bytes:
            # Sort by last accessed time (oldest first)
            sorted_keys = sorted(self.cache_info.keys(), 
                               key=lambda k: self.cache_info[k]['last_accessed'])
            
            for cache_key in sorted_keys:
                cache_file = self.cache_dir / "{}.pkl".format(cache_key)
                if cache_file.exists():
                    cache_file.unlink()
                
                removed_size = self.cache_info[cache_key]['size_bytes']
                del self.cache_info[cache_key]
                total_size -= removed_size
                
                if total_size <= max_size_bytes * 0.8:  # Remove extra 20%
                    break
            
            self._save_cache_info()
    
    def clear(self):
        """Clear all cached data."""
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()
        self.cache_info = {}
        self._save_cache_info()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_size = sum(info['size_bytes'] for info in self.cache_info.values())
        return {
            'num_entries': len(self.cache_info),
            'total_size_mb': total_size / (1024 * 1024),
            'max_size_mb': self.max_size_mb,
            'cache_dir': str(self.cache_dir)
        }


# Global cache instance
_global_cache = VisualizationCache()


def cached_computation(cache_enabled: bool = True):
    """Decorator for caching expensive computations."""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not cache_enabled:
                return func(*args, **kwargs)
            
            # Extract cacheable parameters
            cache_params = {k: v for k, v in kwargs.items() 
                          if isinstance(v, (int, float, str, bool, tuple))}
            cache_params['func_name'] = func.__name__
            
            # Try to get from cache
            if len(args) > 0:
                cached_result = _global_cache.get(args[0], cache_params)
                if cached_result is not None:
                    return cached_result
            
            # Compute and cache
            result = func(*args, **kwargs)
            
            if len(args) > 0:
                _global_cache.set(args[0], cache_params, result)
            
            return result
        
        return wrapper
    return decorator


class MemoryManager:
    """Memory management utilities for large dataset processing."""
    
    @staticmethod
    def get_memory_usage() -> Dict[str, float]:
        """Get current memory usage statistics."""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / (1024 * 1024),  # Resident Set Size
            'vms_mb': memory_info.vms / (1024 * 1024),  # Virtual Memory Size
            'percent': process.memory_percent(),
            'available_mb': psutil.virtual_memory().available / (1024 * 1024)
        }
    
    @staticmethod
    def estimate_array_memory(shape: Tuple[int, ...], dtype: np.dtype) -> float:
        """Estimate memory usage for numpy array in MB."""
        total_elements = np.prod(shape)
        bytes_per_element = np.dtype(dtype).itemsize
        return (total_elements * bytes_per_element) / (1024 * 1024)
    
    @staticmethod
    def chunked_processing(data: np.ndarray, chunk_size: int = None,
                          func: Callable = None, **kwargs) -> List[Any]:
        """Process large arrays in chunks to manage memory."""
        if chunk_size is None:
            # Auto-determine chunk size based on available memory
            available_mb = MemoryManager.get_memory_usage()['available_mb']
            target_chunk_mb = min(100, available_mb * 0.1)  # Use 10% of available memory
            
            # Estimate chunk size
            element_mb = data.nbytes / (data.size * 1024 * 1024)
            chunk_size = int(target_chunk_mb / element_mb) if element_mb > 0 else 1000
            chunk_size = max(100, min(chunk_size, len(data)))
        
        results = []
        
        for i in range(0, len(data), chunk_size):
            chunk = data[i:i + chunk_size]
            
            if func:
                result = func(chunk, **kwargs)
                results.append(result)
            else:
                results.append(chunk)
            
            # Force garbage collection
            gc.collect()
        
        return results
    
    @staticmethod
    def memory_limit_decorator(max_memory_mb: float = 1000):
        """Decorator to limit memory usage of functions."""
        def decorator(func: Callable):
            @wraps(func)
            def wrapper(*args, **kwargs):
                initial_memory = MemoryManager.get_memory_usage()['rss_mb']
                
                result = func(*args, **kwargs)
                
                final_memory = MemoryManager.get_memory_usage()['rss_mb']
                memory_used = final_memory - initial_memory
                
                if memory_used > max_memory_mb:
                    warnings.warn("Function {} used {:.1f}MB (limit: {}MB)".format(
                        func.__name__, memory_used, max_memory_mb))
                
                return result
            return wrapper
        return decorator


class DataDownsampler:
    """Intelligent data downsampling for visualization."""
    
    @staticmethod
    def downsample_waveform(waveform: np.ndarray, target_samples: int = 10000,
                           method: str = 'decimate') -> np.ndarray:
        """
        Downsample waveform for visualization while preserving important features.
        
        Args:
            waveform: Input waveform
            target_samples: Target number of samples
            method: Downsampling method ('decimate', 'subsample', 'envelope')
            
        Returns:
            Downsampled waveform
        """
        if len(waveform) <= target_samples:
            return waveform
        
        if method == 'decimate':
            from scipy.signal import decimate
            decimation_factor = len(waveform) // target_samples
            if decimation_factor > 1:
                # Apply anti-aliasing filter and decimate
                return decimate(waveform, decimation_factor, zero_phase=True)
        
        elif method == 'subsample':
            # Simple subsampling
            step = len(waveform) // target_samples
            return waveform[::step]
        
        elif method == 'envelope':
            # Downsample using envelope detection
            chunk_size = len(waveform) // target_samples
            downsampled = np.zeros(target_samples)
            
            for i in range(target_samples):
                start_idx = i * chunk_size
                end_idx = min(start_idx + chunk_size, len(waveform))
                chunk = waveform[start_idx:end_idx]
                
                # Use max amplitude in chunk
                downsampled[i] = np.max(np.abs(chunk)) * np.sign(chunk[np.argmax(np.abs(chunk))])
            
            return downsampled
        
        return waveform[:target_samples]
    
    @staticmethod
    def downsample_spectrogram(spectrogram: np.ndarray, target_shape: Tuple[int, int] = (500, 500),
                              method: str = 'interpolate') -> np.ndarray:
        """
        Downsample spectrogram for visualization.
        
        Args:
            spectrogram: Input spectrogram (freq x time)
            target_shape: Target shape (freq_bins, time_bins)
            method: Downsampling method ('interpolate', 'bin_average')
            
        Returns:
            Downsampled spectrogram
        """
        if spectrogram.shape[0] <= target_shape[0] and spectrogram.shape[1] <= target_shape[1]:
            return spectrogram
        
        if method == 'interpolate':
            from scipy.interpolate import RegularGridInterpolator
            
            # Create interpolator
            freq_coords = np.arange(spectrogram.shape[0])
            time_coords = np.arange(spectrogram.shape[1])
            interpolator = RegularGridInterpolator((freq_coords, time_coords), spectrogram)
            
            # Create new coordinate grid
            new_freq = np.linspace(0, spectrogram.shape[0] - 1, target_shape[0])
            new_time = np.linspace(0, spectrogram.shape[1] - 1, target_shape[1])
            
            new_coords = np.meshgrid(new_freq, new_time, indexing='ij')
            points = np.column_stack([coord.ravel() for coord in new_coords])
            
            # Interpolate
            downsampled = interpolator(points).reshape(target_shape)
            return downsampled
        
        elif method == 'bin_average':
            # Average over bins
            freq_bins = np.linspace(0, spectrogram.shape[0], target_shape[0] + 1, dtype=int)
            time_bins = np.linspace(0, spectrogram.shape[1], target_shape[1] + 1, dtype=int)
            
            downsampled = np.zeros(target_shape)
            
            for i in range(target_shape[0]):
                for j in range(target_shape[1]):
                    freq_slice = slice(freq_bins[i], freq_bins[i + 1])
                    time_slice = slice(time_bins[j], time_bins[j + 1])
                    downsampled[i, j] = np.mean(spectrogram[freq_slice, time_slice])
            
            return downsampled
        
        return spectrogram
    
    @staticmethod
    def adaptive_detection_sampling(detections: pd.DataFrame, max_detections: int = 1000,
                                   preserve_extremes: bool = True) -> pd.DataFrame:
        """
        Intelligently sample detections for visualization.
        
        Args:
            detections: DataFrame with detection data
            max_detections: Maximum number of detections to keep
            preserve_extremes: Whether to preserve extreme values
            
        Returns:
            Sampled detections DataFrame
        """
        if len(detections) <= max_detections:
            return detections
        
        if preserve_extremes and len(detections) > max_detections:
            # Always keep extreme values for important parameters
            preserve_indices = set()
            
            for param in ['peak_freq', 'amplitude', 'duration']:
                if param in detections.columns:
                    data = detections[param].dropna()
                    if len(data) > 0:
                        # Keep min and max values
                        min_idx = data.idxmin()
                        max_idx = data.idxmax()
                        preserve_indices.update([min_idx, max_idx])
            
            # Random sample from remaining
            remaining_indices = set(detections.index) - preserve_indices
            n_random = max_detections - len(preserve_indices)
            
            if n_random > 0 and len(remaining_indices) > 0:
                random_indices = np.random.choice(list(remaining_indices), 
                                                size=min(n_random, len(remaining_indices)),
                                                replace=False)
                preserve_indices.update(random_indices)
            
            return detections.loc[list(preserve_indices)]
        
        else:
            # Simple random sampling
            sample_indices = np.random.choice(detections.index, size=max_detections, replace=False)
            return detections.loc[sample_indices]


class PlotExporter:
    """Advanced plot export utilities."""
    
    @staticmethod
    def export_high_res_figure(fig: plt.Figure, filename: str, 
                              dpi: int = 300, formats: List[str] = ['png', 'pdf'],
                              optimize: bool = True, **kwargs):
        """
        Export figure in high resolution with optimization.
        
        Args:
            fig: Matplotlib figure
            filename: Base filename (without extension)
            dpi: Resolution for raster formats
            formats: List of formats to export
            optimize: Whether to optimize output
            **kwargs: Additional savefig arguments
        """
        base_path = Path(filename)
        
        # Default save parameters
        save_params = {
            'bbox_inches': 'tight',
            'pad_inches': 0.05,
            'facecolor': 'white',
            'edgecolor': 'none'
        }
        save_params.update(kwargs)
        
        for fmt in formats:
            output_path = base_path.with_suffix(f'.{fmt}')
            
            fmt_params = save_params.copy()
            
            if fmt in ['png', 'jpg', 'tiff']:
                fmt_params['dpi'] = dpi
                
            elif fmt == 'pdf':
                # PDF-specific optimizations
                if optimize:
                    fmt_params['metadata'] = {
                        'Creator': 'PAMpal Visualization System',
                        'Producer': 'matplotlib'
                    }
            
            elif fmt == 'svg':
                # SVG-specific optimizations
                if optimize:
                    fmt_params['metadata'] = {
                        'Creator': 'PAMpal Visualization System'
                    }
            
            try:
                # Simplify the parameters for more reliable saving
                if fmt == 'png':
                    # Use only essential parameters for PNG to avoid compatibility issues
                    fig.savefig(str(output_path), format=fmt, dpi=dpi, bbox_inches='tight')
                else:
                    fig.savefig(str(output_path), format=fmt, **fmt_params)
                print("Exported: {}".format(output_path))
            except Exception as e:
                warnings.warn("Failed to export {}: {}".format(fmt, e))
    
    @staticmethod
    def batch_export_plots(plot_functions: List[Callable], output_dir: str,
                          prefix: str = 'plot', **export_kwargs):
        """
        Batch export multiple plots.
        
        Args:
            plot_functions: List of functions that return (fig, ax) tuples
            output_dir: Output directory
            prefix: Filename prefix
            **export_kwargs: Arguments for export_high_res_figure
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        for i, plot_func in enumerate(plot_functions):
            try:
                fig, ax = plot_func()
                filename = output_path / "{}_{:03d}".format(prefix, i+1)
                PlotExporter.export_high_res_figure(fig, str(filename), **export_kwargs)
                plt.close(fig)
            except Exception as e:
                warnings.warn("Failed to export plot {}: {}".format(i+1, e))
    
    @staticmethod
    def create_plot_archive(plot_dir: str, archive_name: str = 'plots.zip'):
        """
        Create a compressed archive of all plots in a directory.
        
        Args:
            plot_dir: Directory containing plots
            archive_name: Name of archive file
        """
        import zipfile
        
        plot_path = Path(plot_dir)
        archive_path = plot_path / archive_name
        
        with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for plot_file in plot_path.glob('*'):
                if plot_file.is_file() and plot_file.suffix in ['.png', '.pdf', '.svg', '.jpg']:
                    zipf.write(plot_file, plot_file.name)
        
        print("Created archive: {}".format(archive_path))
        return archive_path


def optimize_for_web(fig: plt.Figure) -> plt.Figure:
    """
    Optimize figure for web display.
    
    Args:
        fig: Matplotlib figure
        
    Returns:
        Optimized figure
    """
    # Reduce figure DPI for web
    fig.set_dpi(72)
    
    # Optimize line widths and marker sizes
    for ax in fig.get_axes():
        for line in ax.get_lines():
            if line.get_linewidth() > 1:
                line.set_linewidth(max(0.5, line.get_linewidth() * 0.7))
        
        for collection in ax.collections:
            if hasattr(collection, 'get_sizes'):
                sizes = collection.get_sizes()
                if len(sizes) > 0:
                    collection.set_sizes([max(1, s * 0.7) for s in sizes])
    
    return fig


def clear_cache():
    """Clear the global visualization cache."""
    _global_cache.clear()


def get_cache_stats() -> Dict[str, Any]:
    """Get global cache statistics."""
    return _global_cache.get_stats()