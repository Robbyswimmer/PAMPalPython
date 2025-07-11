# PAMpal Processing Module Design Document

## Overview

This document describes the design and implementation of the processing module in the PAMpal Python package. The processing module is responsible for loading, transforming, and analyzing acoustic data from Pamguard binary files and databases. It provides the core functionality for creating AcousticEvent and AcousticStudy objects from raw data.

## Architecture

The processing module consists of several interconnected components:

1. **Main Processing Pipeline**: Orchestrates the overall data processing workflow
2. **Binary and Database Loaders**: Read data from external sources
3. **Event Detection and Grouping**: Organize detections into meaningful events
4. **Processing Functions**: Apply transformations and analyses to detection data

These components work together to convert raw acoustic data into structured, analyzable objects within the PAMpal framework.

## Main Processing Pipeline

### Function: `process_detections()`

The `process_detections()` function is the primary entry point for processing acoustic data. It takes a PAMpalSettings object as input and returns an AcousticStudy object containing processed data.

#### Workflow:

1. Validate settings and check for required inputs
2. Load data from binary files using the binary parser
3. Load data from databases if specified
4. Apply processing functions to the data
5. Group detections into events based on temporal proximity
6. Create AcousticEvent objects for each group of detections
7. Create an AcousticStudy object containing all events
8. Return the completed study

#### Design Decisions:

1. **Single Entry Point**: Having a single function for the entire processing pipeline simplifies the API and makes it easier for users to get started.

2. **Settings-Based Configuration**: Using a PAMpalSettings object for configuration allows for flexible and reusable processing pipelines.

3. **Study-Centric Output**: Returning an AcousticStudy object provides a container for all processed data and maintains the hierarchical structure of the data model.

4. **Modular Processing Steps**: Breaking the pipeline into discrete steps allows for customization and extension of specific parts of the processing workflow.

## Binary and Database Loaders

### Function: `load_binaries()`

The `load_binaries()` function loads data from Pamguard binary files using the binary parser module. It takes a list of file paths as input and returns a dictionary mapping detector types to DataFrames of detections.

#### Design Decisions:

1. **Abstraction Layer**: The function provides an abstraction layer over the binary parser, simplifying the interface for the main processing pipeline.

2. **Error Handling**: Comprehensive error handling ensures that the processing pipeline can continue even if some files cannot be read.

3. **Result Structure**: The detector-type-to-DataFrame mapping allows for flexible handling of different detector outputs.

### Function: `load_database()`

The `load_database()` function loads data from Pamguard SQLite databases. It takes a database path as input and returns a dictionary mapping table names to DataFrames.

#### Design Decisions:

1. **SQL Abstraction**: The function abstracts SQL queries, making it easier for users who may not be familiar with SQL.

2. **Detector Table Focus**: The function focuses on detector-related tables, ignoring administrative tables and other non-detection data.

3. **Automatic Type Conversion**: Time columns are automatically converted to datetime objects for consistency with binary data.

## Event Detection and Grouping

### Function: `group_detections_into_events()`

The `group_detections_into_events()` function groups detections into events based on temporal proximity. It takes dictionaries of binary and database data as input and returns a dictionary mapping event IDs to event data.

#### Algorithm:

1. Sort all detections by time
2. Initialize the first event with the earliest detection
3. For each subsequent detection:
   - If it occurs within the time window of the current event, add it to that event
   - Otherwise, create a new event
4. Return the mapping of event IDs to event data

#### Design Decisions:

1. **Time-Based Grouping**: Using time as the primary grouping factor aligns with how acoustic events naturally occur in the real world.

2. **Configurable Window**: The time window for grouping can be adjusted to accommodate different types of acoustic activity.

3. **Event Data Structure**: Each event contains start and end times, detector data, and metadata, providing a complete picture of the acoustic activity.

## Processing Functions

### Function: `apply_functions()`

The `apply_functions()` function applies user-defined processing functions to detection data. It takes a dictionary of detector data and a dictionary of functions as input and returns the processed data.

#### Design Decisions:

1. **Function Registry**: Processing functions are registered in the PAMpalSettings object, allowing for flexible configuration.

2. **Detector Type Matching**: Functions are applied only to matching detector types, allowing for specialized processing.

3. **Sequential Application**: Functions are applied in the order they were registered, ensuring predictable processing sequences.

4. **Data Preservation**: The original data is never modified; instead, a copy is made before applying functions.

## Integration with Binary Parser

The processing module integrates with the binary parser through the `load_binaries()` function. This integration allows for seamless loading of binary data into the processing pipeline.

### Integration Points:

1. **File Loading**: The binary parser's `load_binary_files()` function is called to load data from binary files.

2. **Data Transformation**: The parser's output is transformed to match the expected format for event grouping.

3. **Error Handling**: Errors from the binary parser are caught and handled appropriately, ensuring the processing pipeline remains robust.

## Database Integration

The processing module integrates with SQLite databases through the `load_database()` function. This allows for loading data from Pamguard databases without requiring direct SQL knowledge.

### Integration Points:

1. **Table Discovery**: The function automatically discovers relevant detector tables.

2. **Data Reading**: Each table is read into a pandas DataFrame for further processing.

3. **Type Conversion**: Time columns are automatically converted to datetime objects for consistency with binary data.

## Use Cases

### Basic Processing Workflow

```python
# Create settings
settings = pampal.PAMpalSettings()
settings.add_binaries("/path/to/binaries")

# Process detections
study = pampal.process_detections(settings)
```

### Processing with Custom Functions

```python
# Create settings
settings = pampal.PAMpalSettings()
settings.add_binaries("/path/to/binaries")

# Add custom processing functions
def filter_low_amplitude(df):
    return df[df["amplitude"] > 0.1]

settings.add_function("Click_Detector", filter_low_amplitude)

# Process with custom function
study = pampal.process_detections(settings)
```

### Combining Binary and Database Data

```python
# Create settings
settings = pampal.PAMpalSettings()
settings.add_binaries("/path/to/binaries")
settings.add_db("/path/to/database.sqlite")

# Process both sources
study = pampal.process_detections(settings)
```

## Future Enhancements

1. **Parallel Processing**: Implement parallel processing for handling large datasets more efficiently.

2. **Advanced Event Detection**: Develop more sophisticated algorithms for event detection and grouping.

3. **Real-time Processing**: Add support for processing data in real-time as it is generated.

4. **Custom Processors**: Allow users to define custom processors for specific detector types or analysis needs.

5. **Progress Reporting**: Add progress reporting for long-running processing tasks.

## Conclusion

The processing module provides a robust and flexible framework for working with acoustic data in PAMpal. It maintains compatibility with the R implementation while leveraging Python's strengths in data processing and analysis. The modular design allows for customization and extension to meet the needs of different research applications.
