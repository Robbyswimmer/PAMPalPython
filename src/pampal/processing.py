"""
Processing module containing functions for processing acoustic detections.

This module provides the core functionality for loading and processing data from 
PamGuard binary files and databases.
"""
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
import datetime
# import sqlite3  # Temporarily disabled for testing
try:
    import sqlite3
except ImportError:
    # Mock sqlite3 for environments where it's not available
    class MockSqlite3:
        def connect(self, *args, **kwargs):
            raise NotImplementedError("sqlite3 not available")
    sqlite3 = MockSqlite3()
import glob

from .settings import PAMpalSettings
from .acoustic_event import AcousticEvent
from .acoustic_study import AcousticStudy
from .binary_parser import load_binary_files, PamBinaryError


def process_detections(settings: PAMpalSettings) -> AcousticStudy:
    """
    Process detections using the provided settings.
    
    This is the main function for processing acoustic detections. It:
    1. Loads data from binary files and/or databases
    2. Applies processing functions specified in settings
    3. Creates AcousticEvent objects for each set of related detections
    4. Returns an AcousticStudy containing all the events and metadata
    
    Args:
        settings: PAMpalSettings object with configuration
        
    Returns:
        AcousticStudy object containing processed data
        
    Raises:
        ValueError: If no binary files or databases are specified in settings
    """
    # Check that we have either binary files or a database
    if not settings.db and not settings.binaries["list"]:
        raise ValueError("No binary files or database specified in settings")
    
    print(f"Processing detections with {len(settings.binaries['list'])} binary files")
    
    # Create a new AcousticStudy
    study = AcousticStudy(
        id=f"Study_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
        pps=settings,
        files={
            "db": settings.db,
            "binaries": settings.binaries["folder"]
        }
    )
    
    # Load binary data
    binary_data = {}
    if settings.binaries["list"]:
        binary_data = load_binaries(settings.binaries["list"])
    
    # Load database data
    db_data = {}
    if settings.db:
        db_data = load_database(settings.db)
    
    # Apply processing functions if any
    if settings.functions:
        for detector_type in binary_data:
            if detector_type in settings.functions:
                print(f"Applying functions to {detector_type} data...")
                binary_data[detector_type] = apply_functions(
                    {detector_type: binary_data[detector_type]}, 
                    {detector_type: settings.functions[detector_type]},
                    settings=settings
                )[detector_type]
    
    # Group data into events based on time
    events_data = group_detections_into_events(binary_data, db_data)
    
    # Create AcousticEvent objects
    for event_id, event_data in events_data.items():
        # Create an event with detectors from binary and database data
        event = AcousticEvent(
            id=event_id,
            detectors=event_data['detectors'],
            settings={"sr": event_data.get('sr', 192000)}  # Default sample rate if not found
        )
        
        # Add the event to the study
        study.add_event(event)
    
    print(f"Created study with {len(study.events)} events")
    return study


def group_detections_into_events(binary_data: Dict[str, pd.DataFrame], 
                                db_data: Dict[str, pd.DataFrame],
                                time_window: float = 60.0) -> Dict[str, Dict[str, Any]]:
    """
    Group detections into events based on time proximity.
    
    This is a simple implementation that groups detections that are within 
    time_window seconds of each other into the same event.
    
    Args:
        binary_data: Dictionary mapping detector types to DataFrames of detections
        db_data: Dictionary mapping database tables to DataFrames
        time_window: Time window in seconds to consider detections part of the same event
        
    Returns:
        Dictionary mapping event IDs to event data
    """
    events = {}
    detection_count = 0
    
    # Process binary data
    for detector_type, detections in binary_data.items():
        if len(detections) == 0:
            continue
            
        # Ensure data is sorted by time
        if 'UTC' in detections.columns:
            detections = detections.sort_values('UTC')
            
            # Initialize first event if we don't have any yet
            if len(events) == 0:
                first_event_id = f"Event_{detection_count:06d}"
                events[first_event_id] = {
                    'start_time': detections['UTC'].iloc[0],
                    'end_time': detections['UTC'].iloc[0],
                    'detectors': {},
                    'sr': 192000  # Default sample rate
                }
            
            current_event_id = list(events.keys())[-1]
            current_event = events[current_event_id]
            
            # Process each detection
            for _, row in detections.iterrows():
                detection_time = row['UTC']
                
                # Check if this detection belongs to the current event
                if (detection_time - current_event['end_time']).total_seconds() <= time_window:
                    # Update event end time
                    current_event['end_time'] = max(current_event['end_time'], detection_time)
                    
                    # Add detection to event
                    if detector_type not in current_event['detectors']:
                        current_event['detectors'][detector_type] = detections[detections['UTC'] == detection_time].copy()
                    else:
                        current_event['detectors'][detector_type] = pd.concat([
                            current_event['detectors'][detector_type],
                            detections[detections['UTC'] == detection_time]
                        ])
                else:
                    # Create a new event
                    detection_count += 1
                    new_event_id = f"Event_{detection_count:06d}"
                    events[new_event_id] = {
                        'start_time': detection_time,
                        'end_time': detection_time,
                        'detectors': {
                            detector_type: detections[detections['UTC'] == detection_time].copy()
                        },
                        'sr': 192000  # Default sample rate
                    }
                    current_event_id = new_event_id
                    current_event = events[current_event_id]
    
    # If no events were created, create one with whatever data we have
    if len(events) == 0:
        # Check if we have any data at all
        has_data = False
        for detector_type, detections in binary_data.items():
            if len(detections) > 0:
                has_data = True
                break
                
        if has_data:
            default_event_id = "Event_000000"
            events[default_event_id] = {
                'start_time': datetime.datetime.now(),
                'end_time': datetime.datetime.now(),
                'detectors': binary_data,
                'sr': 192000  # Default sample rate
            }
    
    return events


def load_binaries(file_paths: List[str]) -> Dict[str, pd.DataFrame]:
    """
    Load data from PamGuard binary files.
    
    Args:
        file_paths: List of paths to binary files
        
    Returns:
        Dictionary mapping detector types to DataFrames of detections
    """
    try:
        # Use the binary parser module to load files
        return load_binary_files(file_paths, verbose=True)
    except Exception as e:
        print(f"Error loading binary files: {str(e)}")
        # Return an empty result in case of error
        return {}


def load_database(db_path: str, detector_types: Optional[List[str]] = None, 
                 grouping_mode: str = 'event', legacy_format: bool = False) -> Union[Dict[str, pd.DataFrame], Dict[str, any]]:
    """
    Load data from a PAMGuard database file.
    
    Args:
        db_path: Path to the database file
        detector_types: List of detector types to load ('click', 'whistle', 'cepstrum', 'gpl')
                       If None, loads all available detector types
        grouping_mode: Event grouping mode ('event', 'detGroup', 'none')
        legacy_format: If True, returns data in legacy format for backward compatibility
    
    Returns:
        Dictionary with loaded data. If legacy_format=True, returns simple table mapping.
        Otherwise returns comprehensive data structure with detector data, events, and metadata.
    """
    from .database import load_database as enhanced_load_database, DatabaseError
    
    print(f"Loading database from {db_path}...")
    
    try:
        # Use enhanced database loader
        if legacy_format:
            # Legacy mode: return simple table mapping for backward compatibility
            result = enhanced_load_database(db_path, detector_types, 'none')
            detector_data = result.get('detector_data', {})
            
            # Flatten to legacy format
            legacy_result = {}
            for detector_type, df in detector_data.items():
                # Use original table names if available
                if 'detector_table' in df.columns:
                    for table_name in df['detector_table'].unique():
                        table_df = df[df['detector_table'] == table_name].copy()
                        table_df = table_df.drop('detector_table', axis=1, errors='ignore')
                        table_df = table_df.drop('detector_type', axis=1, errors='ignore')
                        legacy_result[table_name] = table_df
                else:
                    # Fallback to detector type name
                    legacy_result[f"{detector_type}_data"] = df
            
            return legacy_result
        else:
            # Enhanced mode: return full data structure
            return enhanced_load_database(db_path, detector_types, grouping_mode)
            
    except DatabaseError as e:
        print(f"Database error: {str(e)}")
        return {}
    except Exception as e:
        print(f"Error loading database: {str(e)}")
        return {}


def apply_functions(data: Dict[str, pd.DataFrame], 
                   functions: Dict[str, Dict[str, Any]], 
                   calibration: Dict[str, Dict[str, Any]] = None,
                   settings=None) -> Dict[str, pd.DataFrame]:
    """
    Apply processing functions to detection data.
    
    Args:
        data: Dictionary mapping detector types to DataFrames of detections
        functions: Dictionary mapping detector types to functions to apply
        calibration: Optional dictionary mapping detector types to calibration functions (legacy)
        settings: PAMpalSettings object containing calibration manager
        
    Returns:
        Dictionary mapping detector types to processed DataFrames
    """
    result = {}
    
    for detector_type, detector_data in data.items():
        # Skip if no functions for this detector type
        if detector_type not in functions or not functions[detector_type]:
            result[detector_type] = detector_data
            continue
            
        # Make a copy of the data to avoid modifying the original
        processed_data = detector_data.copy()
        
        # Get calibration function if available
        calibration_function = None
        if settings and hasattr(settings, 'calibration_manager'):
            calibration_function = settings.calibration_manager.get_calibration(detector_type)
        elif calibration and detector_type in calibration and calibration[detector_type]:
            # Legacy calibration support
            calibration_function = list(calibration[detector_type].values())[0]
        
        # Apply each function in the order they were added
        for func_name, func in functions[detector_type].items():
            print(f"Applying function '{func_name}' to {detector_type} data...")
            
            # Apply the function to the data
            try:
                # Pass calibration function to processing function if it supports it
                import inspect
                func_signature = inspect.signature(func)
                if 'calibration_function' in func_signature.parameters:
                    processed_data = func(processed_data, calibration_function=calibration_function)
                else:
                    processed_data = func(processed_data)
            except Exception as e:
                print(f"Error applying function '{func_name}': {str(e)}")
                
        result[detector_type] = processed_data
        
    return result
