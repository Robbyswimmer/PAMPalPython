# PAMpal Core Classes Design Document

## Overview

This document describes the design and implementation of the core classes that form the foundation of the PAMpal Python package. These classes provide the data structures and interfaces for working with acoustic data and maintain compatibility with the R implementation of PAMpal.

## Core Classes Architecture

The PAMpal package is built around three primary classes that work together to store, organize, and process acoustic data:

1. **PAMpalSettings**: Configuration and settings management
2. **AcousticEvent**: Container for related acoustic detections
3. **AcousticStudy**: Collection of AcousticEvent objects with shared context

### Class Relationships

```
PAMpalSettings
      ^
      |
      v
AcousticStudy -----> [AcousticEvent, AcousticEvent, ...]
                          |
                          v
                     {detector1: DataFrame, detector2: DataFrame, ...}
```

- **PAMpalSettings** is used by **AcousticStudy** to configure processing
- **AcousticStudy** contains multiple **AcousticEvent** objects
- Each **AcousticEvent** contains detector data stored as DataFrames

## PAMpalSettings Class

### Purpose

The PAMpalSettings class manages configuration options for loading and processing acoustic data. It stores paths to binary files and databases, processing functions, and other settings needed for the analysis workflow.

### Key Features

- Binary file and database path management
- Processing function registration
- Configuration validation
- Serialization and deserialization for saving/loading settings

### Design Decisions

1. **Immutable Core Settings**: Once created, core paths and configuration options cannot be changed directly, ensuring consistency throughout the processing pipeline.

2. **Function Registration**: Processing functions are registered with specific detector types, allowing for customized processing pipelines.

3. **Validation on Creation**: Settings are validated when the object is created, catching configuration errors early.

4. **Dictionary-based Storage**: Settings are stored in nested dictionaries for flexibility and easy serialization.

### Implementation Details

```python
class PAMpalSettings:
    def __init__(self, db=None, binaries=None):
        self.db = db
        self.binaries = {"folder": None, "list": []} if binaries is None else binaries
        self.functions = {}
        self.calibration = {}
        self.validate()
```

## AcousticEvent Class

### Purpose

The AcousticEvent class represents a group of related acoustic detections, typically occurring close together in time. It stores detector outputs and provides methods to access and manipulate detection data.

### Key Features

- Storage for detector outputs as pandas DataFrames
- Metadata about the event (ID, time, location)
- Access methods for retrieving specific detector data
- Integration with binary data retrieval

### Design Decisions

1. **DataFrame-based Storage**: Using pandas DataFrames for detector data provides powerful data manipulation capabilities and compatibility with the scientific Python ecosystem.

2. **Flexible Detector Structure**: The class can store data from any detector type, allowing it to adapt to different Pamguard module configurations.

3. **Lazy Loading**: Binary data is only loaded when explicitly requested, reducing memory usage.

4. **Immutable Core Properties**: Event ID and core properties are immutable to maintain data integrity.

### Implementation Details

```python
class AcousticEvent:
    def __init__(self, id, detectors=None, settings=None):
        self.id = id
        self.detectors = {} if detectors is None else detectors
        self.settings = {} if settings is None else settings
        self.study = None
```

## AcousticStudy Class

### Purpose

The AcousticStudy class is the top-level container for acoustic analysis, representing a complete study or dataset. It manages a collection of AcousticEvent objects, provides study-wide settings, and offers methods for working with events in aggregate.

### Key Features

- Storage and management of multiple AcousticEvent objects
- Study-wide settings and metadata
- Access methods for retrieving specific events or data across events
- Integration with processing functions

### Design Decisions

1. **Study-Event Relationship**: Events belong to a study, and the study maintains references to all its events, enabling bidirectional navigation.

2. **Shared File References**: Binary and database file references are stored at the study level, ensuring consistency across events.

3. **Study-Level Processing**: Processing functions can be applied to all events in a study or to specific subsets.

4. **Metadata Aggregation**: The study can aggregate metadata across events for summary statistics and analysis.

### Implementation Details

```python
class AcousticStudy:
    def __init__(self, id, pps=None, files=None, events=None):
        self.id = id
        self.pps = PAMpalSettings() if pps is None else pps
        self.files = {} if files is None else files
        self.events = [] if events is None else events
        
        # Link events to this study
        for event in self.events:
            event.study = self
```

## Use Cases

### Creating a New Study

```python
# Create settings
settings = pampal.PAMpalSettings()
settings.add_binaries("/path/to/binaries")
settings.add_db("/path/to/database.sqlite")

# Process data and create study
study = pampal.process_detections(settings)
```

### Working with Events

```python
# Access events in a study
for event in study.events:
    # Get click detector data
    clicks = event.detectors.get("ClickDetector", None)
    if clicks is not None:
        # Analyze clicks
        print(f"Event {event.id} has {len(clicks)} clicks")
```

### Adding Processing Functions

```python
# Create settings
settings = pampal.PAMpalSettings()

# Add a processing function for clicks
def process_clicks(df):
    df["processed"] = True
    return df

settings.add_function("Click_Detector", process_clicks)

# Process with custom function
study = pampal.process_detections(settings)
```

## Future Enhancements

1. **Event Classification**: Add support for classifying events by species or call type.

2. **Advanced Filtering**: Implement filtering and selection methods for working with specific subsets of events.

3. **Serialization**: Add methods for saving and loading studies to/from disk.

4. **Integration with Visualization Tools**: Add support for visualizing event data using matplotlib or other plotting libraries.

5. **Spatial Analysis**: Add methods for spatial analysis and mapping of events.

## Conclusion

The core classes of PAMpal provide a flexible and powerful foundation for working with acoustic data. The design emphasizes data integrity, flexibility, and compatibility with the scientific Python ecosystem while maintaining functional equivalence with the R implementation.
