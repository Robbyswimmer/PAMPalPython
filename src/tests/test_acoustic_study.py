"""
Unit tests for the AcousticStudy class.
"""
import unittest
import pandas as pd
import numpy as np
from datetime import datetime

from pampal.settings import PAMpalSettings
from pampal.acoustic_event import AcousticEvent
from pampal.acoustic_study import AcousticStudy


class TestAcousticStudy(unittest.TestCase):
    """Test cases for AcousticStudy class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create PAMpalSettings object
        self.settings = PAMpalSettings()
        
        # Create sample GPS data
        self.gps_data = pd.DataFrame({
            'datetime': [datetime.now(), datetime.now()],
            'latitude': [30.5, 30.6],
            'longitude': [-86.2, -86.1]
        })
        
        # Create sample events
        self.event1 = AcousticEvent(
            id="event1",
            detectors={
                "ClickDetector": pd.DataFrame({
                    'UTC': [datetime.now()],
                    'amplitude': [0.5]
                })
            },
            settings={"sr": 192000}
        )
        
        self.event2 = AcousticEvent(
            id="event2",
            detectors={
                "WhistlesMoans": pd.DataFrame({
                    'UTC': [datetime.now()],
                    'duration': [0.5]
                })
            }
        )
        
        # Create sample study
        self.study = AcousticStudy(
            id="test_study",
            events={
                "event1": self.event1,
                "event2": self.event2
            },
            files={
                "db": "test.db",
                "binaries": ["test_folder"]
            },
            gps=self.gps_data,
            pps=self.settings,
            ancillary={"notes": {"note1": "Study note 1"}}
        )
    
    def test_initialization(self):
        """Test initialization of AcousticStudy."""
        # Test default initialization
        empty_study = AcousticStudy()
        self.assertTrue(empty_study.id)  # Should have a default ID (today's date)
        self.assertEqual(empty_study.events, {})
        self.assertEqual(empty_study.files["db"], None)
        self.assertEqual(empty_study.files["binaries"], None)
        self.assertTrue(isinstance(empty_study.gps, pd.DataFrame))
        self.assertTrue(isinstance(empty_study.pps, PAMpalSettings))
        
        # Test initialization with parameters
        self.assertEqual(self.study.id, "test_study")
        self.assertEqual(len(self.study.events), 2)
        self.assertEqual(self.study.files["db"], "test.db")
        self.assertEqual(self.study.files["binaries"], ["test_folder"])
        pd.testing.assert_frame_equal(self.study.gps, self.gps_data)
        self.assertEqual(self.study.ancillary["notes"]["note1"], "Study note 1")
    
    def test_explicit_id(self):
        """Test providing an explicit ID."""
        study = AcousticStudy(id="explicit_id")
        self.assertEqual(study.id, "explicit_id")
    
    def test_add_event(self):
        """Test adding an event to the study."""
        # Create new study
        study = AcousticStudy(id="new_study")
        
        # Add event
        event = AcousticEvent(id="event3")
        study.add_event(event)
        
        # Check if event was added
        self.assertIn("event3", study.events)
        self.assertEqual(study.events["event3"], event)
        
        # Test adding event with duplicate ID
        with self.assertRaises(ValueError):
            study.add_event(event)
    
    def test_add_events(self):
        """Test adding multiple events to the study."""
        # Create new study
        study = AcousticStudy(id="new_study")
        
        # Create events
        event3 = AcousticEvent(id="event3")
        event4 = AcousticEvent(id="event4")
        
        # Add events
        study.add_events([event3, event4])
        
        # Check if events were added
        self.assertEqual(len(study.events), 2)
        self.assertIn("event3", study.events)
        self.assertIn("event4", study.events)
        
        # Test adding event with duplicate ID
        with self.assertRaises(ValueError):
            study.add_events([AcousticEvent(id="event3")])
    
    def test_get_event(self):
        """Test getting an event by ID."""
        # Get existing event
        event = self.study.get_event("event1")
        self.assertEqual(event, self.event1)
        
        # Get non-existent event
        self.assertIsNone(self.study.get_event("non_existent_event"))
    
    def test_get_all_events(self):
        """Test getting all events in the study."""
        events = self.study.get_all_events()
        self.assertEqual(len(events), 2)
        self.assertIn(self.event1, events)
        self.assertIn(self.event2, events)
    
    def test_add_gps(self):
        """Test adding GPS data to the study."""
        # Create new study
        study = AcousticStudy(id="gps_study")
        
        # Add GPS data
        study.add_gps(self.gps_data)
        
        # Check if GPS data was added
        pd.testing.assert_frame_equal(study.gps, self.gps_data)
    
    def test_add_notes(self):
        """Test adding notes to the study."""
        # Create new study
        study = AcousticStudy(id="notes_study")
        
        # Add notes
        study.add_notes({"note1": "Study note 1"})
        self.assertEqual(study.get_notes()["note1"], "Study note 1")
        
        # Add more notes
        study.add_notes({"note2": "Study note 2"})
        self.assertEqual(len(study.get_notes()), 2)
        
        # Overwrite existing note
        study.add_notes({"note1": "Updated study note 1"})
        self.assertEqual(study.get_notes()["note1"], "Updated study note 1")
    
    def test_get_notes(self):
        """Test getting notes from the study."""
        # Get existing notes
        notes = self.study.get_notes()
        self.assertEqual(len(notes), 1)
        self.assertEqual(notes["note1"], "Study note 1")
        
        # Get notes from study with no notes
        empty_study = AcousticStudy()
        self.assertEqual(empty_study.get_notes(), {})
    
    def test_string_representation(self):
        """Test the string representation of AcousticStudy."""
        # Get string representation
        study_str = str(self.study)
        
        # Check that key info is included
        self.assertIn("test_study", study_str)
        self.assertIn("2 AcousticEvents", study_str)
        self.assertIn("note", study_str)


if __name__ == "__main__":
    unittest.main()
