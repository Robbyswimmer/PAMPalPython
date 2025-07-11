"""
AcousticEvent class module.

This module contains the AcousticEvent class, which represents a single acoustic detection event
along with all its associated data and metadata.
"""
from typing import Dict, List, Any, Optional, Union
import pandas as pd


class AcousticEvent:
    """
    A class storing acoustic detections from an Acoustic Event and related metadata.
    
    This class is equivalent to the AcousticEvent S4 class in the R version.
    
    Attributes:
        id (str): Unique identifier for this event
        detectors (Dict): Dictionary of pandas DataFrames with acoustic detections,
            keyed by detector name
        localizations (Dict): Dictionary storing localizations, keyed by method
        settings (Dict): Dictionary of recorder settings
        species (Dict): Dictionary of species classifications for this event,
            keyed by classification method
        files (Dict): Dictionary of files used to create this object,
            keyed by file type
        ancillary (Dict): Dictionary for miscellaneous extra data
    """

    def __init__(self, 
                 id: str = "",
                 detectors: Dict[str, pd.DataFrame] = None,
                 localizations: Dict[str, Any] = None,
                 settings: Dict[str, Any] = None,
                 species: Dict[str, Any] = None,
                 files: Dict[str, Any] = None,
                 ancillary: Dict[str, Any] = None):
        """
        Initialize an AcousticEvent object.
        
        Args:
            id: Unique identifier for the event
            detectors: Dictionary of detections by detector type
            localizations: Dictionary of localizations by method
            settings: Dictionary of recorder settings
            species: Dictionary of species classifications
            files: Dictionary of files used
            ancillary: Dictionary for miscellaneous data
        """
        self.id = id
        self.detectors = detectors or {}
        self.localizations = localizations or {}
        self.settings = settings or {"sr": None}  # Sample rate
        self.species = species or {"id": None}
        self.files = files or {}
        self.ancillary = ancillary or {}
        
    def add_detector_data(self, detector_name: str, data: pd.DataFrame) -> None:
        """
        Add detection data for a specific detector.
        
        Args:
            detector_name: Name of the detector
            data: DataFrame containing the detection data
        """
        self.detectors[detector_name] = data
        
    def get_detector_data(self, detector_name: str) -> Optional[pd.DataFrame]:
        """
        Get detection data for a specific detector.
        
        Args:
            detector_name: Name of the detector
            
        Returns:
            DataFrame of detection data if it exists, None otherwise
        """
        return self.detectors.get(detector_name)
        
    def add_notes(self, notes: Dict[str, Any]) -> None:
        """
        Add notes to the event.
        
        Args:
            notes: Dictionary of notes to add
        """
        if "notes" not in self.ancillary:
            self.ancillary["notes"] = {}
            
        for key, value in notes.items():
            self.ancillary["notes"][key] = value
            
    def get_notes(self) -> Dict[str, Any]:
        """
        Get all notes for this event.
        
        Returns:
            Dictionary of notes or empty dict if none exist
        """
        return self.ancillary.get("notes", {})
    
    def __str__(self) -> str:
        """String representation of the AcousticEvent."""
        detector_names = ", ".join(self.detectors.keys())
        notes = self.get_notes()
        
        result = [f"AcousticEvent object \"{self.id}\" with {len(self.detectors)} detector(s):"]
        result.append(detector_names)
        
        if notes:
            note_count = sum(1 for _ in notes.items())
            result.append(f"And {note_count} notes:")
            # Show up to 6 notes
            for i, (key, value) in enumerate(notes.items()):
                if i >= 6:
                    break
                result.append(f"{key}: {value}")
                
        return "\n".join(result)
        
    def __repr__(self) -> str:
        """Developer string representation."""
        return self.__str__()
