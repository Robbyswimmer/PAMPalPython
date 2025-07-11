"""
Unit tests for the AcousticEvent class.
"""
import unittest
import pandas as pd
import numpy as np
from datetime import datetime

from pampal.acoustic_event import AcousticEvent


class TestAcousticEvent(unittest.TestCase):
    """Test cases for AcousticEvent class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create sample detection data
        self.click_data = pd.DataFrame({
            'UTC': [datetime.now(), datetime.now()],
            'amplitude': [0.5, 0.7],
            'duration': [0.001, 0.002],
            'peak_freq': [120000, 115000]
        })
        
        self.whistle_data = pd.DataFrame({
            'UTC': [datetime.now(), datetime.now()],
            'min_freq': [5000, 6000],
            'max_freq': [12000, 15000],
            'duration': [0.5, 0.7]
        })
        
        # Create sample event
        self.event = AcousticEvent(
            id="test_event",
            detectors={
                "ClickDetector": self.click_data,
                "WhistlesMoans": self.whistle_data
            },
            settings={"sr": 192000},  # Sample rate in Hz
            species={"visual": "Stenella attenuata"},
            files={"binaries": ["file1.pgdf", "file2.pgdf"]},
            ancillary={"notes": {"note1": "Test note 1"}}
        )
    
    def test_initialization(self):
        """Test initialization of AcousticEvent."""
        # Test default initialization
        empty_event = AcousticEvent()
        self.assertEqual(empty_event.id, "")
        self.assertEqual(empty_event.detectors, {})
        self.assertEqual(empty_event.settings, {"sr": None})
        self.assertEqual(empty_event.species, {"id": None})
        self.assertEqual(empty_event.files, {})
        self.assertEqual(empty_event.ancillary, {})
        
        # Test initialization with parameters
        self.assertEqual(self.event.id, "test_event")
        self.assertEqual(len(self.event.detectors), 2)
        self.assertIn("ClickDetector", self.event.detectors)
        self.assertIn("WhistlesMoans", self.event.detectors)
        self.assertEqual(self.event.settings["sr"], 192000)
        self.assertEqual(self.event.species["visual"], "Stenella attenuata")
        self.assertEqual(len(self.event.files["binaries"]), 2)
    
    def test_add_detector_data(self):
        """Test adding detector data."""
        # Create new event
        event = AcousticEvent(id="new_event")
        
        # Add detector data
        cepstrum_data = pd.DataFrame({
            'UTC': [datetime.now()],
            'cepstral_peak': [0.003]
        })
        event.add_detector_data("Cepstrum", cepstrum_data)
        
        # Check if data was added
        self.assertIn("Cepstrum", event.detectors)
        pd.testing.assert_frame_equal(event.detectors["Cepstrum"], cepstrum_data)
        
        # Test overwriting existing data
        new_cepstrum_data = pd.DataFrame({
            'UTC': [datetime.now()],
            'cepstral_peak': [0.005]
        })
        event.add_detector_data("Cepstrum", new_cepstrum_data)
        pd.testing.assert_frame_equal(event.detectors["Cepstrum"], new_cepstrum_data)
    
    def test_get_detector_data(self):
        """Test getting detector data."""
        # Get existing detector data
        click_data = self.event.get_detector_data("ClickDetector")
        pd.testing.assert_frame_equal(click_data, self.click_data)
        
        # Get non-existent detector data
        self.assertIsNone(self.event.get_detector_data("NonExistentDetector"))
    
    def test_add_notes(self):
        """Test adding notes."""
        # Create new event
        event = AcousticEvent(id="notes_event")
        
        # Add notes
        event.add_notes({"note1": "Test note 1"})
        self.assertEqual(event.get_notes()["note1"], "Test note 1")
        
        # Add more notes
        event.add_notes({"note2": "Test note 2", "note3": "Test note 3"})
        self.assertEqual(len(event.get_notes()), 3)
        self.assertEqual(event.get_notes()["note2"], "Test note 2")
        self.assertEqual(event.get_notes()["note3"], "Test note 3")
        
        # Overwrite existing note
        event.add_notes({"note1": "Updated note 1"})
        self.assertEqual(event.get_notes()["note1"], "Updated note 1")
    
    def test_get_notes(self):
        """Test getting notes."""
        # Get existing notes
        notes = self.event.get_notes()
        self.assertEqual(len(notes), 1)
        self.assertEqual(notes["note1"], "Test note 1")
        
        # Get notes from event with no notes
        empty_event = AcousticEvent()
        self.assertEqual(empty_event.get_notes(), {})
    
    def test_string_representation(self):
        """Test the string representation of AcousticEvent."""
        # Get string representation
        event_str = str(self.event)
        
        # Check that key info is included
        self.assertIn("test_event", event_str)
        self.assertIn("2 detector(s)", event_str)
        self.assertIn("ClickDetector", event_str)
        self.assertIn("WhistlesMoans", event_str)
        self.assertIn("note", event_str)


if __name__ == "__main__":
    unittest.main()
