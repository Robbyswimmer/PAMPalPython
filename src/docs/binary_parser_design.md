# PAMpal Binary Parser Design Document

## Overview

This document describes the design and implementation of the PAMpal binary parser, which is responsible for reading and parsing Pamguard binary (.pgdf) files. The parser has been implemented in Python as part of the PAMpal Python conversion project, maintaining functional equivalence with the R implementation.

## Architecture

The binary parser implementation consists of two main components:

1. **Core Binary Parser (`binary_parser.py`)**: Handles low-level binary file parsing and data extraction
2. **Binary Data Retrieval (`binary_data.py`)**: High-level functions for retrieving binary data for specific detections

These components are integrated with the existing PAMpal framework through the Processing module, which uses the binary parser to load and process data from binary files.

## Design Decisions

### 1. Self-contained Binary Parser

**Decision**: Implement a complete, self-contained binary parser within the PAMpal package rather than relying on external libraries.

**Rationale**:
- The R implementation depends on the external `PamBinaries` package, but there is no equivalent Python package available.
- A self-contained implementation gives us full control over the parsing logic and error handling.
- This approach eliminates external dependencies, making the package more robust.

**Implementation**:
- Created custom binary data parsing functions to handle all PGDF file format requirements.
- Used Python's `struct` module for binary data unpacking, ensuring proper endianness and type handling.
- Implemented support for all data types used in PGDF files.

### 2. Module-specific Parsers

**Decision**: Implement specialized parsers for each detector module (Click, Whistle & Moan, GPL).

**Rationale**:
- Different detector modules store data in different formats within the binary files.
- Specialized parsers can handle module-specific data structures and extract relevant information.
- This approach allows for targeted optimizations and feature extraction for each module type.

**Implementation**:
- Created dedicated parser functions for each detector type: `read_click_detector_data()`, `read_whistles_moans_data()`, and `read_gpl_detector_data()`.
- Each parser handles the specific data structures and fields for its detector type.
- Added a generic `read_binary_file()` function that detects the file type and calls the appropriate specialized parser.

### 3. Structured Error Handling

**Decision**: Implement a consistent error handling strategy with a custom `PamBinaryError` exception class.

**Rationale**:
- Binary file parsing is prone to errors due to file corruption, version incompatibilities, or missing files.
- A custom exception class allows for clear, informative error messages and proper error handling.
- Detailed error information helps with debugging and troubleshooting.

**Implementation**:
- Created a custom `PamBinaryError` exception class.
- Implemented comprehensive error checking at each stage of the parsing process.
- Used Python's try/except blocks to catch and handle errors gracefully.
- Added verbose error messages with file paths and specific error details.

### 4. Pandas Integration

**Decision**: Use pandas DataFrames as the primary data structure for binary data.

**Rationale**:
- Pandas provides powerful data manipulation and analysis capabilities.
- DataFrames are well-suited for tabular data like detection records.
- Consistent use of pandas throughout the codebase improves integration and interoperability.

**Implementation**:
- Each binary parser function returns a pandas DataFrame.
- Data types and column names are standardized across all parsers.
- Added DataFrame-specific operations for filtering, merging, and manipulating detection data.

### 5. R Compatibility

**Decision**: Maintain functional equivalence with the R implementation while leveraging Python's strengths.

**Rationale**:
- Ensures that users migrating from R to Python will get the same results.
- Allows for direct comparison of outputs between the R and Python implementations.
- Preserves the proven design patterns of the R implementation.

**Implementation**:
- Carefully studied the R `getBinaryData` function and its dependencies.
- Implemented equivalent functionality in Python's `get_binary_data()` function.
- Maintained the same function signature, parameter names, and return structure.
- Added Python-specific optimizations where appropriate.

### 6. Sample Rate Handling

**Decision**: Implement robust sample rate handling as part of the binary data retrieval process.

**Rationale**:
- Sample rate information is critical for acoustic analysis but not always stored in the binary files.
- The R implementation relies on event settings for sample rate information.
- A consistent approach to sample rate handling ensures correct analysis results.

**Implementation**:
- Added a dedicated function for retrieving sample rates from events or studies.
- Implemented fallback to default sample rates when not explicitly specified.
- Attached sample rate information to binary data during retrieval.

## Implementation Details

### PGDF File Format

The PGDF (Pamguard Data Format) binary file format consists of:

1. **File Header**:
   - Magic number ("PGDF")
   - Version number
   - Creation and analysis timestamps
   - Module and stream information

2. **Data Blocks**:
   - Each block starts with a standard object header (timestamp, channels, sequence, UID)
   - Followed by data items specific to the detector module
   - Each data item has a name, type code, and value

The parser handles all aspects of this format, including proper byte ordering (big-endian) and data type conversions.

### Binary Data Types

The parser supports all data types used in PGDF files:

- SHORT (16-bit integer)
- INT (32-bit integer)
- LONG (64-bit integer)
- FLOAT (32-bit floating-point)
- DOUBLE (64-bit floating-point)
- CHAR (8-bit character)
- BOOLEAN (8-bit boolean)
- STRING (variable-length UTF-8 string)
- BINARY (variable-length binary data)
- ARRAY (array of any of the above types)

### Detector Type Handling

The implementation supports the following detector types:

1. **Click Detector**:
   - Matches binary files with patterns like `^Click_Detector_` or `^SoundTrap_Click_Detector_`
   - Extracts click parameters such as amplitude, duration, and waveform

2. **Whistle & Moan Detector**:
   - Matches binary files with pattern `^WhistlesMoans_`
   - Processes contour data and frequency metrics

3. **Cepstrum Processor**:
   - Special case of Whistle & Moan Detector files with "cepstrum", "burstpulse", or "burst_pulse" in the name
   - Applies special filtering when cepstrum is excluded from detector types

4. **GPL Detector**:
   - Matches binary files with pattern `^GPL_Detector_`
   - Extracts generic parameters for less common detector types

## Usage Examples

### Basic Binary File Parsing

```python
from pampal.binary_parser import read_binary_file

# Parse a single binary file
data = read_binary_file("/path/to/Click_Detector_file.pgdf")

# Print the first few rows
print(data.head())
```

### Loading Multiple Binary Files

```python
from pampal.binary_parser import load_binary_files

# List of binary files to load
files = ["/path/to/file1.pgdf", "/path/to/file2.pgdf", "/path/to/file3.pgdf"]

# Load and combine data from all files
data_dict = load_binary_files(files, verbose=True)

# Access data by detector type
click_data = data_dict.get("Click_Detector_Clicks", None)
whistle_data = data_dict.get("WhistlesMoans_Detector", None)
```

### Retrieving Binary Data for Specific UIDs

```python
from pampal.binary_data import get_binary_data

# Get binary data for specific UIDs from an AcousticStudy
binary_data = get_binary_data(study, uid=["UID1", "UID2", "UID3"], detector_type=["click", "whistle"])

# Access binary data for a specific UID
if "UID1" in binary_data:
    uid1_data = binary_data["UID1"]
    print(f"Found {len(uid1_data)} detections for UID1")
```

## Future Enhancements

1. **Performance Optimization**:
   - Implement parallel processing for parsing multiple files
   - Add memory-efficient parsing for very large files
   - Optimize critical data processing paths

2. **Extended Format Support**:
   - Add support for newer Pamguard binary format versions
   - Handle custom detector modules
   - Support alternative file formats

3. **Advanced Analysis**:
   - Add feature extraction algorithms for each detector type
   - Implement contour analysis for whistle detections
   - Add waveform processing for click detections

4. **Visualization Integration**:
   - Add functions to visualize binary data
   - Implement interactive exploratory tools
   - Create summary visualizations for binary data statistics

## Conclusion

The PAMpal binary parser implementation provides a robust, efficient, and Python-native solution for parsing Pamguard binary files. It maintains functional equivalence with the R implementation while leveraging Python's strengths and modern best practices. The modular design and comprehensive error handling ensure reliability and extensibility for future enhancements.
