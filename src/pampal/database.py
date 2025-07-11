"""
Database integration module for PAMpal.

This module provides comprehensive database integration capabilities following
the R PAMpal approach for loading and processing PAMGuard database files.
"""

import os
import sqlite3
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
import warnings
import re
from pathlib import Path


class DatabaseError(Exception):
    """Custom exception for database-related errors."""
    pass


class PAMGuardDatabase:
    """
    PAMGuard database handler for loading and processing detection data.
    
    This class provides comprehensive database integration following the R PAMpal
    approach, including schema discovery, event grouping, and data validation.
    """
    
    # Detector table patterns (following R PAMpal)
    DETECTOR_PATTERNS = {
        'click': [r'^Click_Detector_', r'^SoundTrap_Click_Detector_'],
        'whistle': [r'^WhistlesMoans_'],
        'cepstrum': [r'^WhistlesMoans_.*([Cc]epstrum|[Bb]urst[Pp]ulse|[Bb]urst_[Pp]ulse)'],
        'gpl': [r'^GPL_Detector_']
    }
    
    # Required columns for different table types
    REQUIRED_COLUMNS = {
        'detection': ['UID', 'UTC', 'BinaryFile'],
        'event': ['Id'],
        'offline_clicks': ['parentID'],
        'offline_events': ['Id']
    }
    
    def __init__(self, db_path: str):
        """
        Initialize database handler.
        
        Args:
            db_path: Path to PAMGuard SQLite database file
            
        Raises:
            FileNotFoundError: If database file doesn't exist
            DatabaseError: If database cannot be opened
        """
        self.db_path = Path(db_path).resolve()
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database file not found: {db_path}")
        
        self._connection = None
        self._tables = None
        self._schema_info = None
        
        # Test database connection
        try:
            with self._get_connection() as conn:
                conn.execute("SELECT 1").fetchone()
        except Exception as e:
            raise DatabaseError(f"Cannot open database {db_path}: {str(e)}")
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection."""
        return sqlite3.connect(str(self.db_path))
    
    def get_tables(self) -> List[str]:
        """
        Get list of all tables in the database.
        
        Returns:
            List of table names
        """
        if self._tables is None:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                )
                self._tables = [row[0] for row in cursor.fetchall()]
        return self._tables
    
    def discover_schema(self):
        """Discover database schema and categorize tables.
        
        Returns:
            dict: Schema information with categorized tables
        """
        tables = self.get_tables()
        
        schema = {
            'detectors': {
                'click': [],
                'whistle': [],
                'cepstrum': [],
                'gpl': []
            },
            'events': {
                'offline_events': [],
                'offline_clicks': [],
                'detection_groups': [],
                'detection_group_children': []
            },
            'other': []
        }
        
        # Event table patterns (check these first as they're more specific)
        event_patterns = {
            'offline_events': [r'.*OfflineEvents$'],
            'offline_clicks': [r'.*OfflineClicks$'],
            'detection_groups': [r'.*Detection.*Group.*Localiser.*Groups$'],
            'detection_group_children': [r'.*Detection.*Group.*Localiser.*Groups.*Children$', r'.*Children$']
        }
        
        # Detector table patterns
        detector_patterns = {
            'click': [r'.*Click.*Detector.*Clicks.*', r'.*Click.*Detector.*(?!OfflineEvents|OfflineClicks).*'],
            'whistle': [r'.*Whistle.*Moans.*(?!.*Cepstrum)'],
            'cepstrum': [r'.*Whistle.*Moans.*Cepstrum.*', r'.*Cepstrum.*'],
            'gpl': [r'.*GPL.*Detector.*']
        }
        
        for table in tables:
            categorized = False
            
            # Check event patterns first (more specific)
            for event_type, patterns in event_patterns.items():
                for pattern in patterns:
                    if re.match(pattern, table, re.IGNORECASE):
                        schema['events'][event_type].append(table)
                        categorized = True
                        break
                if categorized:
                    break
            
            # If not an event table, check detector patterns
            if not categorized:
                for detector_type, patterns in detector_patterns.items():
                    for pattern in patterns:
                        if re.match(pattern, table, re.IGNORECASE):
                            schema['detectors'][detector_type].append(table)
                            categorized = True
                            break
                    if categorized:
                        break
            
            # If still not categorized, add to other
            if not categorized:
                schema['other'].append(table)
        
        self._schema_info = schema
        return schema
    
    def get_table_columns(self, table_name: str) -> List[str]:
        """
        Get column names for a specific table.
        
        Args:
            table_name: Name of the table
            
        Returns:
            List of column names
        """
        with self._get_connection() as conn:
            cursor = conn.execute(f"PRAGMA table_info({table_name})")
            return [row[1] for row in cursor.fetchall()]
    
    def validate_table_schema(self, table_name: str, table_type: str) -> bool:
        """
        Validate that a table has required columns for its type.
        
        Args:
            table_name: Name of the table
            table_type: Type of table ('detection', 'event', etc.)
            
        Returns:
            True if table has required columns
        """
        if table_type not in self.REQUIRED_COLUMNS:
            return True  # No validation rules for this type
        
        columns = self.get_table_columns(table_name)
        required = self.REQUIRED_COLUMNS[table_type]
        
        missing = [col for col in required if col not in columns]
        if missing:
            warnings.warn(f"Table {table_name} missing required columns: {missing}")
            return False
        
        return True
    
    def load_detector_data(self, detector_types: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        Load detection data from detector tables.
        
        Args:
            detector_types: List of detector types to load ('click', 'whistle', etc.)
                          If None, loads all available detector types
        
        Returns:
            Dictionary mapping detector types to combined DataFrames
        """
        schema = self.discover_schema()
        
        if detector_types is None:
            detector_types = list(self.DETECTOR_PATTERNS.keys())
        
        result = {}
        
        with self._get_connection() as conn:
            for detector_type in detector_types:
                tables = schema['detectors'].get(detector_type, [])
                if not tables:
                    continue
                
                detector_data = []
                for table in tables:
                    if not self.validate_table_schema(table, 'detection'):
                        continue
                    
                    try:
                        df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
                        if len(df) == 0:
                            continue
                        
                        # Add metadata
                        df['detector_table'] = table
                        df['detector_type'] = detector_type
                        
                        # Convert UTC timestamps
                        if 'UTC' in df.columns:
                            df['UTC'] = self._convert_pamguard_time(df['UTC'])
                        
                        # Clean BinaryFile paths
                        if 'BinaryFile' in df.columns:
                            df['BinaryFile'] = df['BinaryFile'].apply(
                                lambda x: os.path.basename(x) if pd.notna(x) else x
                            )
                        
                        detector_data.append(df)
                        
                    except Exception as e:
                        warnings.warn(f"Error loading table {table}: {str(e)}")
                        continue
                
                if detector_data:
                    # Combine all tables for this detector type
                    combined_df = pd.concat(detector_data, ignore_index=True, sort=False)
                    result[detector_type] = combined_df
        
        return result
    
    def load_event_data(self, grouping_mode: str = 'event') -> Dict[str, pd.DataFrame]:
        """
        Load event grouping data.
        
        Args:
            grouping_mode: Event grouping mode ('event', 'detGroup')
        
        Returns:
            Dictionary with event and detection data
        """
        schema = self.discover_schema()
        result = {}
        
        with self._get_connection() as conn:
            if grouping_mode == 'event':
                # Load OfflineEvents and OfflineClicks tables
                event_tables = schema['events']['offline_events']
                click_tables = schema['events']['offline_clicks']
                
                if event_tables:
                    events_data = []
                    for table in event_tables:
                        try:
                            df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
                            if len(df) > 0:
                                df['event_table'] = table
                                events_data.append(df)
                        except Exception as e:
                            warnings.warn(f"Error loading event table {table}: {str(e)}")
                    
                    if events_data:
                        result['events'] = pd.concat(events_data, ignore_index=True, sort=False)
                
                if click_tables:
                    clicks_data = []
                    for table in click_tables:
                        try:
                            df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
                            if len(df) > 0:
                                df['click_table'] = table
                                clicks_data.append(df)
                        except Exception as e:
                            warnings.warn(f"Error loading click table {table}: {str(e)}")
                    
                    if clicks_data:
                        result['clicks'] = pd.concat(clicks_data, ignore_index=True, sort=False)
            
            elif grouping_mode == 'detGroup':
                # Load Detection Group Localiser tables
                group_tables = schema['events']['detection_groups']
                children_tables = schema['events']['detection_group_children']
                
                if group_tables:
                    groups_data = []
                    for table in group_tables:
                        try:
                            df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
                            if len(df) > 0:
                                df['group_table'] = table
                                groups_data.append(df)
                        except Exception as e:
                            warnings.warn(f"Error loading group table {table}: {str(e)}")
                    
                    if groups_data:
                        result['groups'] = pd.concat(groups_data, ignore_index=True, sort=False)
                
                if children_tables:
                    children_data = []
                    for table in children_tables:
                        try:
                            df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
                            if len(df) > 0:
                                df['children_table'] = table
                                children_data.append(df)
                        except Exception as e:
                            warnings.warn(f"Error loading children table {table}: {str(e)}")
                    
                    if children_data:
                        result['children'] = pd.concat(children_data, ignore_index=True, sort=False)
        
        return result
    
    def group_detections_by_events(self, detector_data: Dict[str, pd.DataFrame], 
                                  event_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Group detections by events using database relationships.
        
        Args:
            detector_data: Detection data from load_detector_data()
            event_data: Event data from load_event_data()
        
        Returns:
            Dictionary mapping event IDs to detector data
        """
        if 'events' not in event_data or 'clicks' not in event_data:
            warnings.warn("Event grouping requires both events and clicks tables")
            return {}
        
        events_df = event_data['events']
        clicks_df = event_data['clicks']
        
        # Validate required columns
        if 'Id' not in events_df.columns:
            raise DatabaseError("Events table missing 'Id' column")
        if 'parentID' not in clicks_df.columns:
            raise DatabaseError("Clicks table missing 'parentID' column")
        
        grouped_data = {}
        
        # For each event, find associated detections
        for _, event_row in events_df.iterrows():
            event_id = event_row['Id']
            
            # Find clicks associated with this event
            event_clicks = clicks_df[clicks_df['parentID'] == event_id]
            if len(event_clicks) == 0:
                continue
            
            event_detections = {}
            
            # For each detector type, find detections that match the event clicks
            for detector_type, detections_df in detector_data.items():
                if 'UID' not in detections_df.columns:
                    continue
                
                # Find detections that match UIDs in event clicks
                if 'UID' in event_clicks.columns:
                    matching_uids = event_clicks['UID'].tolist()
                    event_detector_data = detections_df[detections_df['UID'].isin(matching_uids)]
                    
                    if len(event_detector_data) > 0:
                        event_detections[detector_type] = event_detector_data
            
            if event_detections:
                grouped_data[str(event_id)] = event_detections
        
        return grouped_data
    
    def _convert_pamguard_time(self, time_series: pd.Series) -> pd.Series:
        """
        Convert PAMGuard time format to pandas datetime.
        
        Args:
            time_series: Series with PAMGuard time values
        
        Returns:
            Series with converted datetime values
        """
        def convert_single_time(time_val):
            if pd.isna(time_val):
                return pd.NaT
            
            try:
                # PAMGuard stores time as milliseconds since epoch or as string
                if isinstance(time_val, (int, float)):
                    # Assume milliseconds since epoch
                    return pd.to_datetime(time_val, unit='ms', utc=True)
                elif isinstance(time_val, str):
                    # Try to parse as ISO format
                    return pd.to_datetime(time_val, utc=True)
                else:
                    return pd.NaT
            except:
                return pd.NaT
        
        return time_series.apply(convert_single_time)
    
    def get_database_summary(self) -> Dict[str, any]:
        """
        Get summary information about the database.
        
        Returns:
            Dictionary with database summary information
        """
        schema = self.discover_schema()
        
        summary = {
            'database_path': str(self.db_path),
            'total_tables': len(self.get_tables()),
            'detector_tables': {
                detector_type: len(tables) 
                for detector_type, tables in schema['detectors'].items()
                if tables
            },
            'event_tables': {
                event_type: len(tables)
                for event_type, tables in schema['events'].items()
                if tables
            },
            'other_tables': len(schema['other'])
        }
        
        # Get detection counts
        detector_data = self.load_detector_data()
        summary['detection_counts'] = {
            detector_type: len(df) 
            for detector_type, df in detector_data.items()
        }
        
        return summary


def load_database(db_path: str, detector_types: Optional[List[str]] = None, 
                 grouping_mode: str = 'event') -> Dict[str, any]:
    """
    Load data from a PAMGuard database following R PAMpal approach.
    
    Args:
        db_path: Path to the database file
        detector_types: List of detector types to load
        grouping_mode: Event grouping mode ('event', 'detGroup', or 'none')
    
    Returns:
        Dictionary with loaded data
    """
    try:
        db = PAMGuardDatabase(db_path)
        
        # Load detector data
        detector_data = db.load_detector_data(detector_types)
        
        result = {
            'detector_data': detector_data,
            'database_path': db_path,
            'schema': db.discover_schema(),
            'summary': db.get_database_summary()
        }
        
        # Load and group by events if requested
        if grouping_mode in ['event', 'detGroup']:
            event_data = db.load_event_data(grouping_mode)
            result['event_data'] = event_data
            
            if grouping_mode == 'event' and detector_data and event_data:
                grouped_data = db.group_detections_by_events(detector_data, event_data)
                result['grouped_data'] = grouped_data
        
        return result
        
    except Exception as e:
        raise DatabaseError(f"Error loading database {db_path}: {str(e)}")


def validate_database_compatibility(db_path: str) -> Dict[str, any]:
    """
    Validate database compatibility with PAMpal.
    
    Args:
        db_path: Path to database file
    
    Returns:
        Dictionary with validation results
    """
    try:
        db = PAMGuardDatabase(db_path)
        schema = db.discover_schema()
        
        validation_results = {
            'compatible': True,
            'warnings': [],
            'errors': [],
            'pamguard_version': 'unknown',
            'detector_types_found': [],
            'event_tables_found': []
        }
        
        # Check for detector tables
        detector_types_found = [
            detector_type for detector_type, tables in schema['detectors'].items()
            if tables
        ]
        validation_results['detector_types_found'] = detector_types_found
        
        if not detector_types_found:
            validation_results['warnings'].append("No detector tables found")
        
        # Check for event tables
        event_types_found = [
            event_type for event_type, tables in schema['events'].items()
            if tables
        ]
        validation_results['event_tables_found'] = event_types_found
        
        # Check for PAMGuard 2.0+ compatibility (UID column)
        # Check detector tables directly for UID column
        for detector_type, tables in schema['detectors'].items():
            for table_name in tables:
                columns = db.get_table_columns(table_name)
                if 'UID' not in columns:
                    validation_results['errors'].append(
                        f"Detector table {table_name} missing UID column - "
                        "requires PAMGuard 2.0 or later"
                    )
                    validation_results['compatible'] = False
        
        if validation_results['errors']:
            validation_results['compatible'] = False
        
        return validation_results
        
    except Exception as e:
        return {
            'compatible': False,
            'errors': [f"Cannot validate database: {str(e)}"],
            'warnings': [],
            'pamguard_version': 'unknown',
            'detector_types_found': [],
            'event_tables_found': []
        }
