"""
Functions for retrieving binary data for specific detections.

This module provides functionality to retrieve binary data associated with 
specific detection UIDs from AcousticEvent or AcousticStudy objects.
"""
import os
import glob
import pandas as pd
import numpy as np
import re
from typing import Dict, List, Any, Optional, Union, Tuple
import warnings

from .acoustic_event import AcousticEvent
from .acoustic_study import AcousticStudy
from .binary_parser import read_binary_file, PamBinaryError


def get_binary_data(x: Union[AcousticEvent, AcousticStudy, List[AcousticEvent]], 
                   uid: List[str], 
                   detector_type: List[str] = None, 
                   quiet: bool = False) -> Dict[str, Any]:
    """
    Retrieve binary data for specific UIDs from AcousticEvent or AcousticStudy objects.
    
    This function matches binary files to detection UIDs and returns the raw binary data.
    It can handle single AcousticEvent objects, AcousticStudy objects, or lists of
    AcousticEvent objects.
    
    Args:
        x: AcousticEvent, AcousticStudy, or list of AcousticEvent objects
        uid: List of UIDs to retrieve data for
        detector_type: List of detector types to filter by (default: ['click', 'whistle', 'cepstrum', 'gpl'])
        quiet: Whether to suppress warnings about missing files or UIDs
        
    Returns:
        Dictionary mapping UIDs to binary data objects
        
    Raises:
        TypeError: If input is not an AcousticEvent, AcousticStudy, or list of AcousticEvent objects
    """
    # Default detector types if not specified
    if detector_type is None:
        detector_type = ['click', 'whistle', 'cepstrum', 'gpl']
    
    # Handle list of AcousticEvents
    if isinstance(x, list):
        is_acev = [isinstance(item, AcousticEvent) for item in x]
        
        if not any(is_acev):
            if not quiet:
                warnings.warn("No AcousticEvents found, check inputs.")
            return {}
            
        if not all(is_acev):
            if not quiet:
                warnings.warn("Not all inputs were AcousticEvent objects.")
                
        # Process each AcousticEvent and combine results
        result = {}
        for i, item in enumerate(x):
            if is_acev[i]:
                item_result = get_binary_data(item, uid, detector_type, quiet=True)
                # Add new items to result, avoiding duplicates
                for key, value in item_result.items():
                    if key not in result:
                        result[key] = value
                    
        return result
    
    # Check input type
    if not isinstance(x, (AcousticEvent, AcousticStudy)):
        if not quiet:
            warnings.warn("This is not an AcousticEvent or AcousticStudy object.")
        return {}
    
    # Get all binary files associated with the object
    all_binaries = get_all_binary_files(x)
    
    # Get detector data and filter by UID
    all_detector_data = get_detector_data(x)
    bins = filter_detections_by_uid(all_detector_data, uid)
    
    if bins is None or len(bins) == 0:
        if not quiet:
            warnings.warn(f"UID(s) {print_n(uid, 6)} were not found in data.")
        return {}
    
    # Filter by detector type
    bins = filter_by_detector_type(bins, detector_type)
    
    if bins is None or len(bins) == 0:
        if not quiet:
            warnings.warn(f"No matches found for detector types: {detector_type}")
        return {}
    
    # Process binary files
    bins['BinaryFile'] = bins['BinaryFile'].apply(os.path.basename)
    
    # Add sample rate if available
    bins = add_sample_rate(x, bins, detector_type)
    
    # Check for multiple matches
    uid_counts = bins['UID'].value_counts()
    multiple_matches = uid_counts[uid_counts > 1].index.tolist()
    
    if multiple_matches and not quiet:
        warnings.warn(f"Multiple matches found for UID(s): {print_n(multiple_matches, 6)}. Using first match.")
    
    # Remove duplicates, keeping first occurrence
    bins = bins.drop_duplicates(subset=['UID'], keep='first')
    
    # Check which UIDs were not found
    missing_uids = [u for u in uid if u not in bins['UID'].values]
    if missing_uids and not quiet:
        warnings.warn(f"No matches found for UID(s): {print_n(missing_uids, 6)}")
    
    # Load binary data for each UID
    result = {}
    for _, row in bins.iterrows():
        uid_value = row['UID']
        binary_file = row['BinaryFile']
        
        # Find the full path to the binary file
        binary_path = None
        for potential_path in all_binaries:
            if os.path.basename(potential_path) == binary_file:
                binary_path = potential_path
                break
        
        if binary_path:
            try:
                # Load binary data
                binary_data = read_binary_file(binary_path)
                
                # Filter to just this UID
                if 'UID' in binary_data.columns:
                    binary_data = binary_data[binary_data['UID'] == uid_value]
                    
                    # Add sample rate information if available
                    if 'sr' in row:
                        binary_data['sr'] = row['sr']
                        
                    result[uid_value] = binary_data
                    
            except Exception as e:
                if not quiet:
                    warnings.warn(f"Error reading binary file {binary_path} for UID {uid_value}: {str(e)}")
                    
        else:
            if not quiet:
                warnings.warn(f"Could not find binary file {binary_file} for UID {uid_value}")
    
    return result


def get_all_binary_files(x: Union[AcousticEvent, AcousticStudy]) -> List[str]:
    """
    Get all binary files associated with an AcousticEvent or AcousticStudy.
    
    Args:
        x: AcousticEvent or AcousticStudy object
        
    Returns:
        List of paths to binary files
    """
    if isinstance(x, AcousticEvent):
        # For AcousticEvent, return the binary files associated with the parent study
        if hasattr(x, 'study') and x.study is not None:
            return get_all_binary_files(x.study)
        return []
    
    elif isinstance(x, AcousticStudy):
        # For AcousticStudy, return all binary files in the binaries folder
        binary_folder = x.files.get('binaries', None)
        if binary_folder and os.path.isdir(binary_folder):
            # Find all .pgdf files in the folder
            binary_pattern = r'(Clicks|WhistlesMoans|GPL).*\.pgdf$'
            all_files = []
            for root, _, files in os.walk(binary_folder):
                for file in files:
                    if re.search(binary_pattern, file, re.IGNORECASE):
                        all_files.append(os.path.join(root, file))
            return all_files
        return []
    
    else:
        return []


def get_detector_data(x: Union[AcousticEvent, AcousticStudy]) -> List[pd.DataFrame]:
    """
    Get all detector data from an AcousticEvent or AcousticStudy.
    
    Args:
        x: AcousticEvent or AcousticStudy object
        
    Returns:
        List of DataFrames containing detector data
    """
    if isinstance(x, AcousticEvent):
        # Return detector data for a single event
        return [df.assign(event_id=x.id) for df in x.detectors.values()]
    
    elif isinstance(x, AcousticStudy):
        # Return detector data for all events in the study
        dfs = []
        for event in x.events:
            dfs.extend([df.assign(event_id=event.id) for df in event.detectors.values()])
        return dfs
    
    return []


def filter_detections_by_uid(detector_data: List[pd.DataFrame], uid: List[str]) -> pd.DataFrame:
    """
    Filter detector data to only include rows with matching UIDs.
    
    Args:
        detector_data: List of DataFrames containing detector data
        uid: List of UIDs to filter by
        
    Returns:
        DataFrame containing only rows with matching UIDs
    """
    filtered_dfs = []
    
    for df in detector_data:
        if 'UID' in df.columns:
            filtered_df = df[df['UID'].isin(uid)][['UTC', 'UID', 'BinaryFile', 'detectorName', 'db']]
            if not filtered_df.empty:
                filtered_dfs.append(filtered_df)
    
    if filtered_dfs:
        return pd.concat(filtered_dfs, ignore_index=True)
    else:
        return pd.DataFrame()


def filter_by_detector_type(bins: pd.DataFrame, detector_type: List[str]) -> pd.DataFrame:
    """
    Filter binary file matches by detector type.
    
    Args:
        bins: DataFrame containing binary file matches
        detector_type: List of detector types to filter by
        
    Returns:
        DataFrame filtered by detector type
    """
    if bins is None or len(bins) == 0:
        return bins
    
    # Create regex pattern for detector types
    type_patterns = []
    for t in detector_type:
        if t == 'click':
            type_patterns.append(r'^Click_Detector_|^SoundTrap_Click_Detector_')
        elif t == 'whistle':
            type_patterns.append(r'^WhistlesMoans_')
        elif t == 'cepstrum':
            type_patterns.append(r'^WhistlesMoans_.*([Cc]epstrum|[Bb]urst[Pp]ulse|[Bb]urst_[Pp]ulse).*')
        elif t == 'gpl':
            type_patterns.append(r'^GPL_Detector_')
    
    type_pattern = '|'.join(type_patterns)
    
    # Filter by detector type
    bins = bins[bins['BinaryFile'].str.contains(type_pattern, regex=True)]
    
    # Handle special case for cepstrum
    if 'cepstrum' not in detector_type:
        cepstrum_pattern = r'cepstrum|burstpulse|burst_pulse'
        no_ceps = ~bins['BinaryFile'].str.contains(cepstrum_pattern, case=False, regex=True)
        bins = bins[no_ceps]
    
    return bins


def add_sample_rate(x: Union[AcousticEvent, AcousticStudy], bins: pd.DataFrame, 
                   detector_type: List[str]) -> pd.DataFrame:
    """
    Add sample rate information to binary file matches.
    
    Args:
        x: AcousticEvent or AcousticStudy object
        bins: DataFrame containing binary file matches
        detector_type: List of detector types
        
    Returns:
        DataFrame with added sample rate information
    """
    if bins is None or len(bins) == 0:
        return bins
    
    # Clone the DataFrame to avoid modifying the original
    result = bins.copy()
    
    # Add sample rate based on detector type and settings
    if isinstance(x, AcousticEvent):
        # Use event settings
        if 'sr' in x.settings:
            result['sr'] = x.settings['sr']
    
    elif isinstance(x, AcousticStudy):
        # Map detector types to events and get sample rates
        result['sr'] = result.apply(lambda row: get_sr_for_detector(x, row, detector_type), axis=1)
    
    # Remove detector name column
    if 'detectorName' in result.columns:
        result = result.drop('detectorName', axis=1)
    
    return result.drop_duplicates()


def get_sr_for_detector(study: AcousticStudy, row: pd.Series, detector_type: List[str]) -> Optional[float]:
    """
    Get sample rate for a specific detector in a specific event.
    
    Args:
        study: AcousticStudy object
        row: Row from DataFrame containing binary file match
        detector_type: List of detector types
        
    Returns:
        Sample rate if available, None otherwise
    """
    # Default sample rate for different detector types
    default_sr = 192000  # Default for clicks
    
    if row['event_id'] in [event.id for event in study.events]:
        # Find the event
        for event in study.events:
            if event.id == row['event_id']:
                # Return sample rate from event settings
                return event.settings.get('sr', default_sr)
    
    return default_sr


def print_n(x: List[Any], n: int) -> str:
    """
    Format a list for display, showing at most n items.
    
    Args:
        x: List to format
        n: Maximum number of items to show
        
    Returns:
        Formatted string
    """
    if len(x) <= n:
        return ', '.join(str(i) for i in x)
    else:
        shown = ', '.join(str(i) for i in x[:n])
        return f"{shown}, ... ({len(x) - n} more)"
