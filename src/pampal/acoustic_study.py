"""
AcousticStudy class module.

This module contains the AcousticStudy class, which is the top-level container for
acoustic data and associated metadata for an entire study.
"""
from typing import Dict, List, Any, Optional, Union
import pandas as pd
import datetime
import platform
import sys

from .settings import PAMpalSettings
from .acoustic_event import AcousticEvent


class AcousticStudy:
    """
    A class storing acoustic data from an entire study along with events and metadata.
    
    This class is equivalent to the AcousticStudy S4 class in the R version.
    
    Attributes:
        id (str): Unique identifier for this study
        events (Dict[str, AcousticEvent]): Dictionary of AcousticEvent objects keyed by ID
        files (Dict): Dictionary of folders and files containing the study data
        gps (pd.DataFrame): DataFrame of GPS coordinates for the study
        pps (PAMpalSettings): PAMpalSettings object used to create this object
        settings (Dict): Dictionary of settings for detectors, localizers, etc.
        effort (pd.DataFrame): DataFrame of effort data
        models (Dict): Dictionary to store models run on the data
        ancillary (Dict): Dictionary for miscellaneous extra data
    """

    def __init__(self,
                 id: Optional[str] = None,
                 events: Dict[str, AcousticEvent] = None,
                 files: Dict[str, Any] = None,
                 gps: pd.DataFrame = None,
                 pps: PAMpalSettings = None,
                 settings: Dict[str, Any] = None,
                 effort: pd.DataFrame = None,
                 models: Dict[str, Any] = None,
                 ancillary: Dict[str, Any] = None):
        """
        Initialize an AcousticStudy object.
        
        Args:
            id: Unique identifier for the study (defaults to current date if None)
            events: Dictionary of AcousticEvent objects
            files: Dictionary of files and folders
            gps: DataFrame of GPS coordinates
            pps: PAMpalSettings object
            settings: Dictionary of settings
            effort: DataFrame of effort data
            models: Dictionary of models
            ancillary: Dictionary for miscellaneous data
        """
        # Use current date as default ID if none provided
        if id is None:
            id = datetime.date.today().isoformat()
            print(f"No ID supplied for this AcousticStudy object, will use today's "
                  f"date: {id}. Please assign a better name with study.id = 'NAME'\n"
                  f"In the future it is recommended to set the 'id' parameter.")
            
        self.id = id
        self.events = events or {}
        
        # Set up default files dictionary
        default_files = {
            "db": None,
            "binaries": None,
            "visual": None,
            "enviro": None
        }
        if files:
            # Update with provided values
            default_files.update(files)
        self.files = default_files
        
        self.gps = gps if gps is not None else pd.DataFrame()
        self.pps = pps if pps is not None else PAMpalSettings()
        
        default_settings = {
            "detectors": {},
            "localizations": {}
        }
        if settings:
            default_settings.update(settings)
        self.settings = default_settings
        
        self.effort = effort if effort is not None else pd.DataFrame()
        self.models = models or {}
        
        # Initialize ancillary data with version info
        default_ancillary = {
            "version": {
                "python": sys.version,
                "platform": platform.platform(),
                "pampal": "0.1.0"  # This should be retrieved from package metadata in production
            },
            "process_date": datetime.datetime.now()
        }
        if ancillary:
            default_ancillary.update(ancillary)
        self.ancillary = default_ancillary
        
    def add_event(self, event: AcousticEvent) -> None:
        """
        Add an AcousticEvent to this study.
        
        Args:
            event: AcousticEvent object to add
            
        Raises:
            ValueError: If an event with the same ID already exists
        """
        if event.id in self.events:
            raise ValueError(f"An event with ID '{event.id}' already exists in this study")
            
        self.events[event.id] = event
        
    def add_events(self, events: List[AcousticEvent]) -> None:
        """
        Add multiple AcousticEvent objects to this study.
        
        Args:
            events: List of AcousticEvent objects to add
            
        Raises:
            ValueError: If any event has an ID that already exists in the study
        """
        for event in events:
            self.add_event(event)
    
    def get_event(self, event_id: str) -> Optional[AcousticEvent]:
        """
        Get an AcousticEvent by its ID.
        
        Args:
            event_id: ID of the event to retrieve
            
        Returns:
            The AcousticEvent if found, None otherwise
        """
        return self.events.get(event_id)
        
    def get_all_events(self) -> List[AcousticEvent]:
        """
        Get all AcousticEvent objects in this study.
        
        Returns:
            List of all AcousticEvent objects
        """
        return list(self.events.values())
    
    def add_gps(self, gps_data: pd.DataFrame) -> None:
        """
        Add GPS data to this study.
        
        Args:
            gps_data: DataFrame of GPS data with at minimum columns for 
                      datetime, latitude, and longitude
        """
        # In a real implementation, we would validate the GPS data format
        self.gps = gps_data
        
    def add_notes(self, notes: Dict[str, Any]) -> None:
        """
        Add notes to the study.
        
        Args:
            notes: Dictionary of notes to add
        """
        if "notes" not in self.ancillary:
            self.ancillary["notes"] = {}
            
        for key, value in notes.items():
            self.ancillary["notes"][key] = value
            
    def get_notes(self) -> Dict[str, Any]:
        """
        Get all notes for this study.
        
        Returns:
            Dictionary of notes or empty dict if none exist
        """
        return self.ancillary.get("notes", {})
    
    def __str__(self) -> str:
        """String representation of the AcousticStudy."""
        notes = self.get_notes()
        
        result = [f"AcousticStudy object named '{self.id}' with {len(self.events)} AcousticEvents."]
        
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
