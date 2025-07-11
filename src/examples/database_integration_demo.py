#!/usr/bin/env python3
"""
PAMpal Database Integration Demo

This example demonstrates the enhanced database integration capabilities
of PAMpal Python, showing how to load, process, and analyze PAMGuard
database files with comprehensive event grouping and data validation.
"""

import os
import sys
import tempfile
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

# Add the parent directory to the path so we can import pampal
sys.path.insert(0, str(Path(__file__).parent.parent))

from pampal.database import (
    PAMGuardDatabase, load_database, validate_database_compatibility
)
from pampal.processing import load_database as legacy_load_database
import pandas as pd


def create_sample_database():
    """Create a comprehensive sample PAMGuard database for demonstration."""
    print("Creating sample PAMGuard database...")
    
    # Create temporary database file
    with tempfile.NamedTemporaryFile(suffix='.sqlite3', delete=False) as f:
        db_path = f.name
    
    conn = sqlite3.connect(db_path)
    
    # Create detector tables following PAMGuard naming conventions
    tables_sql = {
        'Click_Detector_Clicks': '''
            CREATE TABLE Click_Detector_Clicks (
                UID INTEGER PRIMARY KEY,
                UTC REAL,
                BinaryFile TEXT,
                parentID INTEGER,
                amplitude REAL,
                duration REAL,
                peakFreq REAL,
                bandwidth REAL,
                centroidFreq REAL,
                rms REAL
            )
        ''',
        'Click_Detector_Clicks_Beaked_Whale': '''
            CREATE TABLE Click_Detector_Clicks_Beaked_Whale (
                UID INTEGER PRIMARY KEY,
                UTC REAL,
                BinaryFile TEXT,
                parentID INTEGER,
                amplitude REAL,
                ici REAL,
                sweepRate REAL
            )
        ''',
        'WhistlesMoans_Whistles': '''
            CREATE TABLE WhistlesMoans_Whistles (
                UID INTEGER PRIMARY KEY,
                UTC REAL,
                BinaryFile TEXT,
                parentID INTEGER,
                startFreq REAL,
                endFreq REAL,
                duration REAL,
                meanFreq REAL,
                maxFreq REAL,
                minFreq REAL
            )
        ''',
        'WhistlesMoans_Cepstrum': '''
            CREATE TABLE WhistlesMoans_Cepstrum (
                UID INTEGER PRIMARY KEY,
                UTC REAL,
                BinaryFile TEXT,
                parentID INTEGER,
                cepstrumPeak REAL,
                cepstrumMean REAL,
                quefrency REAL
            )
        ''',
        'GPL_Detector_Detections': '''
            CREATE TABLE GPL_Detector_Detections (
                UID INTEGER PRIMARY KEY,
                UTC REAL,
                BinaryFile TEXT,
                parentID INTEGER,
                confidence REAL,
                templateMatch REAL,
                snr REAL
            )
        ''',
        'Click_Detector_OfflineEvents': '''
            CREATE TABLE Click_Detector_OfflineEvents (
                Id INTEGER PRIMARY KEY,
                eventType TEXT,
                comment TEXT,
                startTime REAL,
                endTime REAL,
                species TEXT,
                confidence INTEGER
            )
        ''',
        'Click_Detector_OfflineClicks': '''
            CREATE TABLE Click_Detector_OfflineClicks (
                UID INTEGER PRIMARY KEY,
                parentID INTEGER,
                clickNo INTEGER,
                eventComment TEXT
            )
        ''',
        'Detection_Group_Localiser_Groups': '''
            CREATE TABLE Detection_Group_Localiser_Groups (
                Id INTEGER PRIMARY KEY,
                Text_Annotation TEXT,
                startTime REAL,
                endTime REAL,
                localisationContents INTEGER,
                referenceHydrophones INTEGER
            )
        ''',
        'Detection_Group_Localiser_Groups_Children': '''
            CREATE TABLE Detection_Group_Localiser_Groups_Children (
                UID INTEGER PRIMARY KEY,
                parentID INTEGER,
                groupNo INTEGER,
                timeDelay REAL
            )
        '''
    }
    
    # Create all tables
    for table_name, sql in tables_sql.items():
        conn.execute(sql)
    
    # Generate realistic test data
    base_time = datetime(2023, 6, 15, 14, 30, 0, tzinfo=timezone.utc).timestamp() * 1000
    
    # Click detector data (main)
    click_data = []
    for i in range(50):
        click_data.append((
            i + 1,  # UID
            base_time + i * 1000,  # UTC (1 second intervals)
            f'Clicks_{i//10 + 1:03d}.pgdf',  # BinaryFile
            (i // 10) + 1,  # parentID (group every 10 clicks)
            0.3 + (i % 10) * 0.05,  # amplitude
            0.0008 + (i % 5) * 0.0002,  # duration
            40000 + (i % 20) * 1000,  # peakFreq
            5000 + (i % 10) * 500,  # bandwidth
            42000 + (i % 15) * 800,  # centroidFreq
            0.15 + (i % 8) * 0.02  # rms
        ))
    
    conn.executemany('''
        INSERT INTO Click_Detector_Clicks 
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', click_data)
    
    # Beaked whale click data
    beaked_whale_data = []
    for i in range(20):
        beaked_whale_data.append((
            i + 51,  # UID (continue from main clicks)
            base_time + 60000 + i * 2000,  # UTC (2 second intervals)
            f'BeakedWhale_{i//5 + 1:03d}.pgdf',  # BinaryFile
            6 + (i // 5),  # parentID
            0.6 + (i % 5) * 0.08,  # amplitude
            0.15 + (i % 3) * 0.05,  # ici
            -2.5 + (i % 4) * 0.5  # sweepRate
        ))
    
    conn.executemany('''
        INSERT INTO Click_Detector_Clicks_Beaked_Whale 
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', beaked_whale_data)
    
    # Whistle data
    whistle_data = []
    for i in range(15):
        whistle_data.append((
            i + 71,  # UID
            base_time + 30000 + i * 5000,  # UTC (5 second intervals)
            f'Whistles_{i//3 + 1:03d}.pgdf',  # BinaryFile
            10 + (i // 3),  # parentID
            8000 + (i % 5) * 1000,  # startFreq
            12000 + (i % 7) * 1500,  # endFreq
            2.0 + (i % 4) * 0.5,  # duration
            10000 + (i % 6) * 800,  # meanFreq
            15000 + (i % 8) * 1000,  # maxFreq
            7000 + (i % 4) * 500  # minFreq
        ))
    
    conn.executemany('''
        INSERT INTO WhistlesMoans_Whistles 
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', whistle_data)
    
    # Cepstrum data
    cepstrum_data = []
    for i in range(10):
        cepstrum_data.append((
            i + 86,  # UID
            base_time + 45000 + i * 3000,  # UTC
            f'Cepstrum_{i//2 + 1:03d}.pgdf',  # BinaryFile
            15 + (i // 2),  # parentID
            0.7 + (i % 3) * 0.1,  # cepstrumPeak
            0.4 + (i % 4) * 0.05,  # cepstrumMean
            0.002 + (i % 5) * 0.0005  # quefrency
        ))
    
    conn.executemany('''
        INSERT INTO WhistlesMoans_Cepstrum 
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', cepstrum_data)
    
    # GPL detector data
    gpl_data = []
    for i in range(8):
        gpl_data.append((
            i + 96,  # UID
            base_time + 75000 + i * 8000,  # UTC
            f'GPL_{i//2 + 1:03d}.pgdf',  # BinaryFile
            20 + (i // 2),  # parentID
            0.8 + (i % 4) * 0.05,  # confidence
            0.75 + (i % 3) * 0.08,  # templateMatch
            15.0 + (i % 5) * 2.0  # snr
        ))
    
    conn.executemany('''
        INSERT INTO GPL_Detector_Detections 
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', gpl_data)
    
    # Event data
    event_data = [
        (1, 'Dolphin', 'Bottlenose dolphin encounter', base_time - 5000, base_time + 15000, 'Tursiops truncatus', 8),
        (2, 'Dolphin', 'Possible Rissos dolphin', base_time + 10000, base_time + 25000, 'Grampus griseus', 6),
        (3, 'Whale', 'Sperm whale clicks', base_time + 20000, base_time + 40000, 'Physeter macrocephalus', 9),
        (4, 'Beaked Whale', 'Cuvier beaked whale', base_time + 55000, base_time + 85000, 'Ziphius cavirostris', 7),
        (5, 'Unknown', 'Unidentified clicks', base_time + 80000, base_time + 100000, 'Unknown', 3)
    ]
    
    conn.executemany('''
        INSERT INTO Click_Detector_OfflineEvents 
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', event_data)
    
    # Offline clicks data (linking detections to events)
    offline_clicks_data = []
    click_id = 1
    for event_id in range(1, 6):
        clicks_per_event = [10, 10, 10, 15, 5][event_id - 1]
        for click_no in range(1, clicks_per_event + 1):
            offline_clicks_data.append((
                click_id,  # UID
                event_id,  # parentID
                click_no,  # clickNo
                f'Event {event_id} click {click_no}'  # eventComment
            ))
            click_id += 1
    
    conn.executemany('''
        INSERT INTO Click_Detector_OfflineClicks 
        VALUES (?, ?, ?, ?)
    ''', offline_clicks_data)
    
    # Detection group data
    group_data = [
        (1, 'Foraging sequence', base_time + 5000, base_time + 35000, 3, 4),
        (2, 'Social group', base_time + 40000, base_time + 70000, 2, 4),
        (3, 'Deep dive sequence', base_time + 75000, base_time + 95000, 1, 2)
    ]
    
    conn.executemany('''
        INSERT INTO Detection_Group_Localiser_Groups 
        VALUES (?, ?, ?, ?, ?, ?)
    ''', group_data)
    
    # Detection group children data
    group_children_data = []
    detection_id = 1
    for group_id in range(1, 4):
        detections_per_group = [8, 6, 4][group_id - 1]
        for group_no in range(1, detections_per_group + 1):
            group_children_data.append((
                detection_id,  # UID
                group_id,  # parentID
                group_no,  # groupNo
                (group_no - 1) * 0.001  # timeDelay
            ))
            detection_id += 1
    
    conn.executemany('''
        INSERT INTO Detection_Group_Localiser_Groups_Children 
        VALUES (?, ?, ?, ?)
    ''', group_children_data)
    
    conn.commit()
    conn.close()
    
    print(f"Sample database created: {db_path}")
    return db_path


def demonstrate_database_validation(db_path):
    """Demonstrate database validation functionality."""
    print("\n" + "="*60)
    print("DATABASE VALIDATION")
    print("="*60)
    
    validation_result = validate_database_compatibility(db_path)
    
    print(f"Database compatible: {validation_result['compatible']}")
    print(f"PAMGuard version: {validation_result['pamguard_version']}")
    print(f"Detector types found: {validation_result['detector_types_found']}")
    print(f"Event tables found: {validation_result['event_tables_found']}")
    
    if validation_result['warnings']:
        print(f"Warnings: {validation_result['warnings']}")
    
    if validation_result['errors']:
        print(f"Errors: {validation_result['errors']}")


def demonstrate_schema_discovery(db_path):
    """Demonstrate schema discovery capabilities."""
    print("\n" + "="*60)
    print("SCHEMA DISCOVERY")
    print("="*60)
    
    db = PAMGuardDatabase(db_path)
    schema = db.discover_schema()
    
    print("Detector Tables:")
    for detector_type, tables in schema['detectors'].items():
        if tables:
            print(f"  {detector_type.upper()}: {len(tables)} tables")
            for table in tables:
                print(f"    - {table}")
    
    print("\nEvent Tables:")
    for event_type, tables in schema['events'].items():
        if tables:
            print(f"  {event_type.replace('_', ' ').title()}: {len(tables)} tables")
            for table in tables:
                print(f"    - {table}")
    
    if schema['other']:
        print(f"\nOther Tables: {len(schema['other'])}")
        for table in schema['other']:
            print(f"  - {table}")


def demonstrate_detector_data_loading(db_path):
    """Demonstrate loading detector data."""
    print("\n" + "="*60)
    print("DETECTOR DATA LOADING")
    print("="*60)
    
    # Load all detector data
    result = load_database(db_path, grouping_mode='none')
    detector_data = result['detector_data']
    
    print("Loaded Detector Data:")
    for detector_type, df in detector_data.items():
        print(f"\n{detector_type.upper()} Detections:")
        print(f"  Total detections: {len(df)}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Time range: {df['UTC'].min()} to {df['UTC'].max()}")
        print(f"  Binary files: {df['BinaryFile'].nunique()} unique files")
        
        # Show sample data
        print("  Sample data:")
        print(df.head(3).to_string(index=False))


def demonstrate_event_grouping(db_path):
    """Demonstrate event-based grouping."""
    print("\n" + "="*60)
    print("EVENT-BASED GROUPING")
    print("="*60)
    
    # Load data with event grouping
    result = load_database(db_path, grouping_mode='event')
    
    if 'event_data' in result:
        event_data = result['event_data']
        
        if 'events' in event_data:
            events_df = event_data['events']
            print(f"Events found: {len(events_df)}")
            print("\nEvent Details:")
            for _, event in events_df.iterrows():
                print(f"  Event {event['Id']}: {event['eventType']} - {event['species']}")
                print(f"    Comment: {event['comment']}")
                print(f"    Confidence: {event['confidence']}/10")
        
        if 'clicks' in event_data:
            clicks_df = event_data['clicks']
            print(f"\nOffline clicks: {len(clicks_df)}")
            
            # Group by event
            clicks_by_event = clicks_df.groupby('parentID').size()
            print("Clicks per event:")
            for event_id, count in clicks_by_event.items():
                print(f"  Event {event_id}: {count} clicks")
    
    # Show grouped data if available
    if 'grouped_data' in result:
        grouped_data = result['grouped_data']
        print(f"\nGrouped detections available for {len(grouped_data)} events")
        
        for event_id, event_detections in grouped_data.items():
            print(f"\nEvent {event_id} detections:")
            for detector_type, detections_df in event_detections.items():
                print(f"  {detector_type}: {len(detections_df)} detections")


def demonstrate_detection_group_analysis(db_path):
    """Demonstrate detection group analysis."""
    print("\n" + "="*60)
    print("DETECTION GROUP ANALYSIS")
    print("="*60)
    
    # Load data with detection group grouping
    result = load_database(db_path, grouping_mode='detGroup')
    
    if 'event_data' in result:
        event_data = result['event_data']
        
        if 'groups' in event_data:
            groups_df = event_data['groups']
            print(f"Detection groups found: {len(groups_df)}")
            print("\nGroup Details:")
            for _, group in groups_df.iterrows():
                print(f"  Group {group['Id']}: {group['Text_Annotation']}")
                print(f"    Localisation contents: {group['localisationContents']}")
                print(f"    Reference hydrophones: {group['referenceHydrophones']}")
        
        if 'children' in event_data:
            children_df = event_data['children']
            print(f"\nGroup children: {len(children_df)}")
            
            # Group by parent
            children_by_group = children_df.groupby('parentID').size()
            print("Detections per group:")
            for group_id, count in children_by_group.items():
                print(f"  Group {group_id}: {count} detections")


def demonstrate_database_summary(db_path):
    """Demonstrate database summary functionality."""
    print("\n" + "="*60)
    print("DATABASE SUMMARY")
    print("="*60)
    
    db = PAMGuardDatabase(db_path)
    summary = db.get_database_summary()
    
    print(f"Database: {Path(summary['database_path']).name}")
    print(f"Total tables: {summary['total_tables']}")
    
    print(f"\nDetector tables:")
    for detector_type, count in summary['detector_tables'].items():
        print(f"  {detector_type}: {count} tables")
    
    print(f"\nEvent tables:")
    for event_type, count in summary['event_tables'].items():
        print(f"  {event_type.replace('_', ' ')}: {count} tables")
    
    print(f"\nDetection counts:")
    total_detections = 0
    for detector_type, count in summary['detection_counts'].items():
        print(f"  {detector_type}: {count} detections")
        total_detections += count
    
    print(f"\nTotal detections: {total_detections}")


def demonstrate_legacy_compatibility(db_path):
    """Demonstrate backward compatibility with legacy interface."""
    print("\n" + "="*60)
    print("LEGACY COMPATIBILITY")
    print("="*60)
    
    # Test legacy format
    legacy_data = legacy_load_database(db_path, legacy_format=True)
    
    print("Legacy format data structure:")
    for table_name, df in legacy_data.items():
        print(f"  {table_name}: {len(df)} rows, {len(df.columns)} columns")
    
    # Compare with enhanced format
    enhanced_data = legacy_load_database(db_path, legacy_format=False)
    
    print(f"\nEnhanced format structure:")
    print(f"  detector_data: {len(enhanced_data['detector_data'])} detector types")
    print(f"  schema: {len(enhanced_data['schema']['detectors'])} detector categories")
    print(f"  summary: {enhanced_data['summary']['total_tables']} total tables")


def main():
    """Main demonstration function."""
    print("PAMpal Database Integration Demo")
    print("="*60)
    
    # Create sample database
    db_path = create_sample_database()
    
    try:
        # Run all demonstrations
        demonstrate_database_validation(db_path)
        demonstrate_schema_discovery(db_path)
        demonstrate_detector_data_loading(db_path)
        demonstrate_event_grouping(db_path)
        demonstrate_detection_group_analysis(db_path)
        demonstrate_database_summary(db_path)
        demonstrate_legacy_compatibility(db_path)
        
        print("\n" + "="*60)
        print("DEMO COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"Sample database available at: {db_path}")
        print("You can use this database for further testing and development.")
        
    except Exception as e:
        print(f"\nError during demonstration: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up (comment out if you want to keep the database)
        # if os.path.exists(db_path):
        #     os.unlink(db_path)
        #     print(f"Cleaned up temporary database: {db_path}")
        pass


if __name__ == '__main__':
    main()
