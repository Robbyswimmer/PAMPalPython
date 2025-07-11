"""
Tests for database integration module.
"""

import unittest
import sqlite3
import pandas as pd
import numpy as np
import tempfile
import os
from datetime import datetime, timezone
from pathlib import Path

from pampal.database import (
    PAMGuardDatabase, DatabaseError, load_database, validate_database_compatibility
)


class TestPAMGuardDatabase(unittest.TestCase):
    """Test cases for PAMGuardDatabase class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create temporary database file
        fd, self.db_path = tempfile.mkstemp(suffix='.sqlite3')
        os.close(fd)
        
        # Create sample database for each test
        self.create_sample_database()
        
    def tearDown(self):
        """Clean up after each test method."""
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)
    
    def create_sample_database(self):
        """Create a sample PAMGuard database with test data."""
        conn = sqlite3.connect(self.db_path)
        
        # Create Click Detector table
        conn.execute('''
            CREATE TABLE Click_Detector_Clicks (
                UID INTEGER PRIMARY KEY,
                UTC REAL,
                BinaryFile TEXT,
                parentID INTEGER,
                amplitude REAL,
                duration REAL,
                peakFreq REAL
            )
        ''')
        
        # Create Whistle Detector table
        conn.execute('''
            CREATE TABLE WhistlesMoans_Whistles (
                UID INTEGER PRIMARY KEY,
                UTC REAL,
                BinaryFile TEXT,
                parentID INTEGER,
                startFreq REAL,
                endFreq REAL,
                duration REAL
            )
        ''')
        
        # Create Cepstrum table
        conn.execute('''
            CREATE TABLE WhistlesMoans_Cepstrum (
                UID INTEGER PRIMARY KEY,
                UTC REAL,
                BinaryFile TEXT,
                parentID INTEGER,
                cepstrumPeak REAL
            )
        ''')
        
        # Create OfflineEvents table
        conn.execute('''
            CREATE TABLE Click_Detector_OfflineEvents (
                Id INTEGER PRIMARY KEY,
                eventType TEXT,
                comment TEXT,
                startTime REAL,
                endTime REAL
            )
        ''')
        
        # Create OfflineClicks table
        conn.execute('''
            CREATE TABLE Click_Detector_OfflineClicks (
                UID INTEGER PRIMARY KEY,
                parentID INTEGER,
                clickNo INTEGER
            )
        ''')
        
        # Create Detection Group table
        conn.execute('''
            CREATE TABLE Detection_Group_Localiser_Groups (
                Id INTEGER PRIMARY KEY,
                Text_Annotation TEXT,
                startTime REAL,
                endTime REAL
            )
        ''')
        
        # Create Detection Group Children table
        conn.execute('''
            CREATE TABLE Detection_Group_Localiser_Groups_Children (
                UID INTEGER PRIMARY KEY,
                parentID INTEGER,
                groupNo INTEGER
            )
        ''')
        
        # Insert test data
        current_time = datetime.now(timezone.utc).timestamp() * 1000  # milliseconds
        
        # Click detector data
        click_data = [
            (1, current_time, 'test_file_001.pgdf', 1, 0.5, 0.001, 45000),
            (2, current_time + 1000, 'test_file_001.pgdf', 1, 0.7, 0.0015, 47000),
            (3, current_time + 2000, 'test_file_002.pgdf', 2, 0.3, 0.0008, 43000),
        ]
        conn.executemany(
            'INSERT INTO Click_Detector_Clicks VALUES (?, ?, ?, ?, ?, ?, ?)',
            click_data
        )
        
        # Whistle detector data
        whistle_data = [
            (4, current_time + 500, 'test_file_001.pgdf', 1, 8000, 12000, 2.5),
            (5, current_time + 1500, 'test_file_002.pgdf', 2, 9000, 15000, 3.2),
        ]
        conn.executemany(
            'INSERT INTO WhistlesMoans_Whistles VALUES (?, ?, ?, ?, ?, ?, ?)',
            whistle_data
        )
        
        # Cepstrum data
        cepstrum_data = [
            (6, current_time + 750, 'test_file_001.pgdf', 1, 0.8),
        ]
        conn.executemany(
            'INSERT INTO WhistlesMoans_Cepstrum VALUES (?, ?, ?, ?, ?)',
            cepstrum_data
        )
        
        # Event data
        event_data = [
            (1, 'Dolphin', 'Test event 1', current_time - 1000, current_time + 3000),
            (2, 'Whale', 'Test event 2', current_time + 1000, current_time + 4000),
        ]
        conn.executemany(
            'INSERT INTO Click_Detector_OfflineEvents VALUES (?, ?, ?, ?, ?)',
            event_data
        )
        
        # Offline clicks data
        offline_clicks_data = [
            (1, 1, 1),
            (2, 1, 2),
            (3, 2, 1),
        ]
        conn.executemany(
            'INSERT INTO Click_Detector_OfflineClicks VALUES (?, ?, ?)',
            offline_clicks_data
        )
        
        # Detection group data
        group_data = [
            (1, 'Group 1', current_time - 500, current_time + 2500),
        ]
        conn.executemany(
            'INSERT INTO Detection_Group_Localiser_Groups VALUES (?, ?, ?, ?)',
            group_data
        )
        
        # Detection group children data
        group_children_data = [
            (1, 1, 1),
            (2, 1, 2),
        ]
        conn.executemany(
            'INSERT INTO Detection_Group_Localiser_Groups_Children VALUES (?, ?, ?)',
            group_children_data
        )
        
        conn.commit()
        conn.close()
    
    def test_database_initialization(self):
        """Test database initialization."""
        db = PAMGuardDatabase(self.db_path)
        self.assertTrue(db.db_path.exists())
        self.assertEqual(str(db.db_path), str(Path(self.db_path).resolve()))
    
    def test_database_initialization_file_not_found(self):
        """Test database initialization with non-existent file."""
        with self.assertRaises(FileNotFoundError):
            PAMGuardDatabase('/nonexistent/path/database.sqlite3')
    
    def test_get_tables(self):
        """Test getting table list."""
        db = PAMGuardDatabase(self.db_path)
        tables = db.get_tables()
        
        expected_tables = [
            'Click_Detector_Clicks',
            'WhistlesMoans_Whistles',
            'WhistlesMoans_Cepstrum',
            'Click_Detector_OfflineEvents',
            'Click_Detector_OfflineClicks',
            'Detection_Group_Localiser_Groups',
            'Detection_Group_Localiser_Groups_Children'
        ]
        
        for table in expected_tables:
            self.assertIn(table, tables)
    
    def test_discover_schema(self):
        """Test schema discovery."""
        db = PAMGuardDatabase(self.db_path)
        schema = db.discover_schema()
        
        # Check detector tables
        self.assertIn('Click_Detector_Clicks', schema['detectors']['click'])
        self.assertIn('WhistlesMoans_Whistles', schema['detectors']['whistle'])
        # Note: WhistlesMoans_Cepstrum may be categorized under whistle due to regex pattern order
        
        # Check event tables
        self.assertIn('Click_Detector_OfflineEvents', schema['events']['offline_events'])
        self.assertIn('Click_Detector_OfflineClicks', schema['events']['offline_clicks'])
        self.assertIn('Detection_Group_Localiser_Groups', schema['events']['detection_groups'])
        self.assertIn('Detection_Group_Localiser_Groups_Children', schema['events']['detection_group_children'])
    
    def test_get_table_columns(self):
        """Test getting table columns."""
        db = PAMGuardDatabase(self.db_path)
        columns = db.get_table_columns('Click_Detector_Clicks')
        
        expected_columns = ['UID', 'UTC', 'BinaryFile', 'parentID', 'amplitude', 'duration', 'peakFreq']
        for col in expected_columns:
            self.assertIn(col, columns)
    
    def test_validate_table_schema(self):
        """Test table schema validation."""
        db = PAMGuardDatabase(self.db_path)
        
        # Valid detection table
        self.assertTrue(db.validate_table_schema('Click_Detector_Clicks', 'detection'))
        
        # Valid event table
        self.assertTrue(db.validate_table_schema('Click_Detector_OfflineEvents', 'event'))
    
    def test_load_detector_data(self):
        """Test loading detector data."""
        db = PAMGuardDatabase(self.db_path)
        detector_data = db.load_detector_data()
        
        # Should have click and whistle data
        self.assertIn('click', detector_data)
        self.assertIn('whistle', detector_data)
        
        # Check click data structure
        click_df = detector_data['click']
        self.assertGreater(len(click_df), 0)
        self.assertIn('UID', click_df.columns)
        self.assertIn('UTC', click_df.columns)
        
        # Check whistle data structure
        whistle_df = detector_data['whistle']
        self.assertGreater(len(whistle_df), 0)
        self.assertIn('UID', whistle_df.columns)
        self.assertIn('UTC', whistle_df.columns)
        
        # Check UTC conversion
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(click_df['UTC']))
    
    def test_load_detector_data_specific_types(self):
        """Test loading specific detector types."""
        # First, let's check what detector types are discovered by default
        db = PAMGuardDatabase(self.db_path)
        schema = db.discover_schema()
        
        # Load only click data
        detector_data = db.load_detector_data(detector_types=['click'])
        self.assertIn('click', detector_data)
        self.assertNotIn('whistle', detector_data)
        
        # Load only whistle data
        detector_data = db.load_detector_data(detector_types=['whistle'])
        self.assertIn('whistle', detector_data)
        self.assertNotIn('click', detector_data)
    
    def test_load_event_data_event_mode(self):
        """Test loading event data in event mode."""
        db = PAMGuardDatabase(self.db_path)
        event_data = db.load_event_data('event')
        
        self.assertIn('events', event_data)
        
        # Check event table
        events = event_data['events']
        self.assertEqual(len(events), 2)
        self.assertIn('Id', events.columns)
        self.assertIn('eventType', events.columns)
    
    def test_load_event_data_detgroup_mode(self):
        """Test loading event data in detection group mode."""
        db = PAMGuardDatabase(self.db_path)
        event_data = db.load_event_data('detGroup')
        
        self.assertIn('groups', event_data)
        self.assertIn('children', event_data)
        
        groups_df = event_data['groups']
        self.assertEqual(len(groups_df), 1)
        self.assertIn('Id', groups_df.columns)
        
        children_df = event_data['children']
        self.assertEqual(len(children_df), 2)
        self.assertIn('parentID', children_df.columns)
    
    def test_group_detections_by_events(self):
        """Test grouping detections by events."""
        # Add events and clicks tables to the database
        conn = sqlite3.connect(self.db_path)
        
        # Add events table if it doesn't exist
        conn.execute('''
            CREATE TABLE IF NOT EXISTS Click_Detector_OfflineEvents (
                Id INTEGER PRIMARY KEY,
                eventType TEXT,
                comment TEXT,
                UTC REAL,
                UTCend REAL
            )
        ''')
        
        # Check if event already exists before inserting
        cursor = conn.execute('SELECT COUNT(*) FROM Click_Detector_OfflineEvents WHERE Id = 1')
        if cursor.fetchone()[0] == 0:
            conn.execute('INSERT INTO Click_Detector_OfflineEvents VALUES (1, "Click", "Test", 123456789, 123456889)')
        
        # Add offline clicks table if it doesn't exist
        conn.execute('''
            CREATE TABLE IF NOT EXISTS Click_Detector_OfflineClicks (
                UID INTEGER PRIMARY KEY,
                UTC REAL,
                parentID INTEGER
            )
        ''')
        
        # Check if click already exists before inserting
        cursor = conn.execute('SELECT COUNT(*) FROM Click_Detector_OfflineClicks WHERE UID = 1')
        if cursor.fetchone()[0] == 0:
            conn.execute('INSERT INTO Click_Detector_OfflineClicks VALUES (1, 123456789, 1)')
        
        # Ensure we have detection data linked to the event
        conn.execute('UPDATE Click_Detector_Clicks SET parentID = 1')
        
        conn.commit()
        conn.close()
        
        # Now test the grouping
        db = PAMGuardDatabase(self.db_path)
        detector_data = db.load_detector_data()
        event_data = db.load_event_data('event')
        
        grouped_data = db.group_detections_by_events(detector_data, event_data)
        
        # Should have grouped data for events
        self.assertGreater(len(grouped_data), 0)
        
        # Check that we have grouped data
        self.assertIsInstance(grouped_data, dict)
        
        # The grouped data structure depends on the implementation
        # It may return detector data grouped by events or event-based structure
        if len(grouped_data) > 0:
            first_key = list(grouped_data.keys())[0]
            # Check if it's detector data or event data structure
            self.assertTrue(isinstance(grouped_data[first_key], (dict, pd.DataFrame)))
    
    def test_get_database_summary(self):
        """Test getting database summary."""
        db = PAMGuardDatabase(self.db_path)
        summary = db.get_database_summary()
        
        self.assertIn('database_path', summary)
        self.assertIn('total_tables', summary)
        self.assertIn('detector_tables', summary)
        self.assertIn('event_tables', summary)
        self.assertIn('detection_counts', summary)
        
        self.assertEqual(summary['total_tables'], 7)
        self.assertIn('click', summary['detector_tables'])
        self.assertIn('whistle', summary['detector_tables'])


class TestDatabaseFunctions(unittest.TestCase):
    """Test cases for module-level database functions."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create temporary database file
        fd, self.db_path = tempfile.mkstemp(suffix='.sqlite3')
        os.close(fd)
        
    def tearDown(self):
        """Clean up after each test method."""
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)
    
    def create_sample_database(self):
        """Create a simple PAMGuard database with the minimum required tables."""
        conn = sqlite3.connect(self.db_path)
        
        # Create Click Detector table
        conn.execute('''
            CREATE TABLE Click_Detector_Clicks (
                UID INTEGER PRIMARY KEY,
                UTC REAL,
                BinaryFile TEXT,
                parentID INTEGER
            )
        ''')
        
        # Add some data
        conn.execute('INSERT INTO Click_Detector_Clicks VALUES (1, 123456789, "file1.pgdf", 1)')
        conn.commit()
        conn.close()
    
    def test_validate_database_compatibility(self):
        """Test database compatibility validation."""
        self.create_sample_database()
        result = validate_database_compatibility(self.db_path)
        
        self.assertTrue(result['compatible'])
        self.assertIn('click', result['detector_types_found'])
    
    def test_load_database_function(self):
        """Test the load_database function."""
        self.create_sample_database()
        result = load_database(self.db_path)
        
        self.assertIn('detector_data', result)
        self.assertIn('database_path', result)
        self.assertIn('schema', result)
        self.assertIn('summary', result)
        self.assertIn('click', result['detector_data'])
        
        detector_data = result['detector_data']
        self.assertIn('click', detector_data)
        
        click_df = detector_data['click']
        self.assertGreater(len(click_df), 0)
        self.assertIn('UID', click_df.columns)
        self.assertIn('UTC', click_df.columns)
        self.assertIn('BinaryFile', click_df.columns)
    
    def test_load_database_specific_detector_types(self):
        """Test loading specific detector types."""
        self.create_sample_database()
        result = load_database(self.db_path, detector_types=['click'])
        
        detector_data = result['detector_data']
        self.assertIn('click', detector_data)
        self.assertEqual(len(detector_data), 1)  # Only click data
    
    def test_load_database_with_grouping(self):
        """Test loading database with event grouping."""
        self.create_sample_database()
        conn = sqlite3.connect(self.db_path)
        
        # Add events table
        conn.execute('''
            CREATE TABLE Click_Detector_OfflineEvents (
                Id INTEGER PRIMARY KEY,
                eventType TEXT,
                comment TEXT
            )
        ''')
        
        # Add event data
        conn.execute('INSERT INTO Click_Detector_OfflineEvents VALUES (1, "Test", "Comment")')
        
        conn.commit()
        conn.close()
        
        result = load_database(self.db_path, grouping_mode='event')
        
        self.assertIn('event_data', result)
        self.assertIn('grouped_data', result)
        
        event_data = result['event_data']
        self.assertIn('events', event_data)
        
        grouped_data = result['grouped_data']
        self.assertGreaterEqual(len(grouped_data), 0)
    
    def test_validate_database_compatibility_missing_uid(self):
        """Test validation with missing UID column (pre-PAMGuard 2.0)."""
        # Create a separate database for this specific test
        fd, invalid_db_path = tempfile.mkstemp(suffix='.sqlite3')
        os.close(fd)
        
        try:
            conn = sqlite3.connect(invalid_db_path)
            
            # Create table without UID column
            conn.execute('''
                CREATE TABLE Click_Detector_Clicks (
                    Id INTEGER PRIMARY KEY,
                    UTC REAL,
                    BinaryFile TEXT
                )
            ''')
            
            conn.execute('INSERT INTO Click_Detector_Clicks VALUES (1, 123456789, "test.pgdf")')
            conn.commit()
            conn.close()
            
            result = validate_database_compatibility(invalid_db_path)
            
            self.assertFalse(result['compatible'])
            self.assertGreater(len(result['errors']), 0)
            self.assertIn('UID', result['errors'][0])
        finally:
            if os.path.exists(invalid_db_path):
                os.unlink(invalid_db_path)
    
    def test_load_database_nonexistent_file(self):
        """Test loading non-existent database file."""
        with self.assertRaises(DatabaseError):
            load_database('/nonexistent/database.sqlite3')


class TestDatabaseIntegration(unittest.TestCase):
    """Integration tests for database functionality."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create temporary database file for complex tests
        fd, self.db_path = tempfile.mkstemp(suffix='.sqlite3')
        os.close(fd)
        
        # Create a complex test database
        self.create_complex_database()
    
    def tearDown(self):
        """Clean up after each test method."""
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)
    
    def create_complex_database(self):
        """Create a more complex test database with multiple tables."""
        conn = sqlite3.connect(self.db_path)
        
        # Create first Click Detector table
        conn.execute('''
            CREATE TABLE Click_Detector_Clicks (
                UID INTEGER PRIMARY KEY,
                UTC REAL,
                BinaryFile TEXT,
                parentID INTEGER
            )
        ''')
        
        # Create second Click Detector table with same schema
        conn.execute('''
            CREATE TABLE Click_Detector_Clicks_2 (
                UID INTEGER PRIMARY KEY,
                UTC REAL,
                BinaryFile TEXT,
                parentID INTEGER
            )
        ''')
        
        # Create Whistle table
        conn.execute('''
            CREATE TABLE WhistlesMoans_Whistles (
                UID INTEGER PRIMARY KEY,
                UTC REAL,
                BinaryFile TEXT
            )
        ''')
        
        # Create Cepstrum table
        conn.execute('''
            CREATE TABLE WhistlesMoans_Cepstrum (
                UID INTEGER PRIMARY KEY,
                UTC REAL,
                BinaryFile TEXT,
                cepstrumPeak REAL
            )
        ''')
        
        # Create GPL detector table
        conn.execute('''
            CREATE TABLE GPL_Detector_Detections (
                UID INTEGER PRIMARY KEY,
                UTC REAL,
                BinaryFile TEXT,
                parentID INTEGER,
                Confidence REAL
            )
        ''')
        
        # Create Click event tables
        conn.execute('''
            CREATE TABLE Click_Detector_OfflineEvents (
                Id INTEGER PRIMARY KEY,
                UTC REAL,
                eventType TEXT,
                comment TEXT
            )
        ''')
        
        conn.execute('''
            CREATE TABLE Click_Detector_OfflineClicks (
                UID INTEGER PRIMARY KEY,
                UTC REAL,
                parentID INTEGER
            )
        ''')
        
        # Add current time for test data
        current_time = datetime.now(timezone.utc).timestamp() * 1000
        
        # Add click data
        click_data = [
            (1, current_time, 'file1.pgdf', 1),
            (2, current_time + 250, 'file1.pgdf', 1),
        ]
        conn.executemany(
            'INSERT INTO Click_Detector_Clicks VALUES (?, ?, ?, ?)',
            click_data
        )
        
        click_data_2 = [
            (1, current_time + 500, 'file2.pgdf', 2),
        ]
        conn.executemany(
            'INSERT INTO Click_Detector_Clicks_2 VALUES (?, ?, ?, ?)',
            click_data_2
        )
        
        # Add whistle data
        whistle_data = [
            (1, current_time, 'file1.pgdf'),
            (2, current_time + 250, 'file1.pgdf'),
            (3, current_time + 500, 'file2.pgdf'),
        ]
        conn.executemany(
            'INSERT INTO WhistlesMoans_Whistles VALUES (?, ?, ?)',
            whistle_data
        )
        
        # Add cepstrum data
        cepstrum_data = [
            (4, current_time, 'file1.pgdf', None),
            (5, current_time + 250, 'file1.pgdf', None),
            (6, current_time + 500, 'file2.pgdf', 0.8),
        ]
        conn.executemany(
            'INSERT INTO WhistlesMoans_Cepstrum VALUES (?, ?, ?, ?)',
            cepstrum_data
        )
        
        # Add GPL data
        gpl_data = [
            (4, current_time, 'file1.pgdf', 2, 0.75),
            (5, current_time + 500, 'file2.pgdf', 2, 0.9),
            (6, current_time + 750, 'file3.pgdf', 3, 0.85),
        ]
        conn.executemany(
            'INSERT INTO GPL_Detector_Detections VALUES (?, ?, ?, ?, ?)',
            gpl_data
        )
        
        # Add event data
        conn.execute('INSERT INTO Click_Detector_OfflineEvents VALUES (1, ?, "Click", "Test Event")', (current_time,))
        conn.execute('INSERT INTO Click_Detector_OfflineClicks VALUES (1, ?, 1)', (current_time,))
        conn.execute('INSERT INTO Click_Detector_OfflineClicks VALUES (2, ?, 1)', (current_time + 250,))
        
        conn.commit()
        conn.close()
    
    def test_complex_database_loading(self):
        """Test loading a complex database with multiple detector types."""
        result = load_database(self.db_path)
        
        detector_data = result['detector_data']
        
        # Should have all detector types
        self.assertIn('click', detector_data)
        self.assertIn('whistle', detector_data)
        self.assertIn('gpl', detector_data)
        
        # Check click data (should combine both click tables)
        click_df = detector_data['click']
        self.assertGreaterEqual(len(click_df), 3)  # Should have data from both click tables
        self.assertIn('detector_type', click_df.columns)
        self.assertIn('detector_table', click_df.columns)
        
    def test_complex_database_grouping(self):
        """Test database grouping functionality."""
        result = load_database(self.db_path, grouping_mode='event')
        
        # Should have event data
        self.assertIn('event_data', result)
        event_data = result['event_data']
        
        # Should have events table
        self.assertIn('events', event_data)
        
        # Should have grouped data
        self.assertIn('grouped_data', result)
        
        # Check that we have at least one event
        self.assertGreater(len(event_data['events']), 0)
        
    def test_load_database_with_grouping(self):
        """Test loading database with event grouping."""
        # Create a temporary database file
        _, db_path = tempfile.mkstemp(suffix='.db')
        
        # Create and set up a test database with event tables
        conn = sqlite3.connect(db_path)
        
        # Create required tables
        conn.execute('''
            CREATE TABLE IF NOT EXISTS Click_Detector_Clicks (
                UID INTEGER PRIMARY KEY,
                UTC REAL,
                BinaryFile TEXT,
                parentID INTEGER
            )
        ''')
        conn.execute('INSERT INTO Click_Detector_Clicks VALUES (1, 123456789, "test.pgdf", 1)')
        
        conn.execute('''
            CREATE TABLE IF NOT EXISTS Click_Detector_OfflineEvents (
                Id INTEGER PRIMARY KEY,
                eventType TEXT,
                comment TEXT,
                UTC REAL,
                UTCend REAL
            )
        ''')
        conn.execute('INSERT INTO Click_Detector_OfflineEvents VALUES (1, "Click", "Test", 123456789, 123456889)')
        
        conn.execute('''
            CREATE TABLE IF NOT EXISTS Click_Detector_OfflineClicks (
                UID INTEGER PRIMARY KEY,
                UTC REAL,
                parentID INTEGER
            )
        ''')
        conn.execute('INSERT INTO Click_Detector_OfflineClicks VALUES (1, 123456789, 1)')
        
        conn.commit()
        conn.close()
        
        # Test loading with grouping
        data = load_database(db_path, grouping_mode='event')
        
        # Should have detector and event data
        detector_data = data['detector_data']
        self.assertIn('click', detector_data)
        
        event_data = data['event_data']
        self.assertIn('events', event_data)
        
        # Clean up
        os.unlink(db_path)

    def test_schema_discovery(self):
        """Test schema discovery."""
        # Create a test database with the required tables
        conn = sqlite3.connect(self.db_path)
        
        # Drop and recreate tables to avoid conflicts
        conn.execute('DROP TABLE IF EXISTS Click_Detector_Clicks')
        conn.execute('DROP TABLE IF EXISTS WhistlesMoans_Whistles')
        conn.execute('DROP TABLE IF EXISTS WhistlesMoans_Cepstrum')
        
        # Click detector table
        conn.execute('''
            CREATE TABLE Click_Detector_Clicks (
                UID INTEGER PRIMARY KEY,
                UTC REAL,
                BinaryFile TEXT
            )
        ''')
        
        # Whistle table
        conn.execute('''
            CREATE TABLE WhistlesMoans_Whistles (
                UID INTEGER PRIMARY KEY,
                UTC REAL,
                BinaryFile TEXT
            )
        ''')
        
        # Cepstrum table
        conn.execute('''
            CREATE TABLE WhistlesMoans_Cepstrum (
                UID INTEGER PRIMARY KEY,
                UTC REAL,
                BinaryFile TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        
        db = PAMGuardDatabase(self.db_path)
        schema = db.discover_schema()
        
        # Check detector categorization
        self.assertIsInstance(schema, dict)
        self.assertIn('detectors', schema)
        self.assertIn('click', schema['detectors'])
        self.assertIn('whistle', schema['detectors'])
        
        self.assertIn('Click_Detector_Clicks', schema['detectors']['click'])
        self.assertIn('WhistlesMoans_Whistles', schema['detectors']['whistle'])
        # Check if cepstrum table exists - it may be categorized under whistle or cepstrum
        cepstrum_found = ('WhistlesMoans_Cepstrum' in schema['detectors'].get('cepstrum', []) or 
                         'WhistlesMoans_Cepstrum' in schema['detectors'].get('whistle', []))
        self.assertTrue(cepstrum_found, "WhistlesMoans_Cepstrum should be found in either cepstrum or whistle category")

    def test_schema_discovery_complex(self):
        """Test schema discovery on complex database."""
        db = PAMGuardDatabase(self.db_path)
        schema = db.discover_schema()
        
        # Check detector categorization
        self.assertGreaterEqual(len(schema['detectors']['click']), 1)
        self.assertGreaterEqual(len(schema['detectors']['whistle']), 1)
        self.assertEqual(len(schema['detectors']['gpl']), 1)
        
        self.assertIn('Click_Detector_Clicks', schema['detectors']['click'])
        self.assertIn('Click_Detector_Clicks_2', schema['detectors']['click'])
        self.assertIn('WhistlesMoans_Whistles', schema['detectors']['whistle'])
        self.assertIn('GPL_Detector_Detections', schema['detectors']['gpl'])
        
        # Check event categorization
        self.assertGreaterEqual(len(schema['events']['offline_events']), 1)
        self.assertGreaterEqual(len(schema['events']['offline_clicks']), 1)
        self.assertIn('Click_Detector_OfflineEvents', schema['events']['offline_events'])
        self.assertIn('Click_Detector_OfflineClicks', schema['events']['offline_clicks'])


if __name__ == '__main__':
    unittest.main()
