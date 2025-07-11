# PAMpal Testing Framework Design Document

## Overview

This document describes the design and implementation of the testing framework for the PAMpal Python package. The testing framework ensures code reliability, facilitates continuous development, and verifies that the Python implementation maintains functional equivalence with the original R package.

## Architecture

The PAMpal testing framework consists of several components:

1. **Test Modules**: Separate test modules for each major component of the package
2. **Test Runner**: A script to execute all tests together
3. **Test Fixtures**: Common setup and teardown code for tests
4. **Virtual Environment**: Isolated environment for running tests

## Test Module Structure

The testing framework uses Python's built-in `unittest` module, with each test module focusing on a specific component of the package:

### 1. `test_settings.py`

Tests for the PAMpalSettings class, including:
- Initialization with different parameters
- Adding and retrieving binary files and databases
- Adding and applying processing functions
- Validation of settings

### 2. `test_acoustic_event.py`

Tests for the AcousticEvent class, including:
- Initialization with different parameters
- Adding and retrieving detector data
- Metadata management
- Event relationships with AcousticStudy objects

### 3. `test_acoustic_study.py`

Tests for the AcousticStudy class, including:
- Initialization with different parameters
- Adding and retrieving events
- Study-wide settings and metadata
- Aggregation methods across events

### 4. `test_processing.py`

Tests for the processing module, including:
- Basic processing workflow
- Binary data loading
- Database loading
- Function application
- Event grouping

### 5. `test_binary_parser.py` (Planned)

Tests for the binary parser module, including:
- Reading binary file headers
- Parsing different detector types
- Handling corrupt or invalid files
- Integration with the processing pipeline

## Test Runner

The `run_tests.py` script provides a simple way to run all tests together. It:

1. Discovers all test modules in the `tests` directory
2. Creates a test suite containing all discovered tests
3. Executes the suite and reports results
4. Provides a summary of passed/failed tests

## Design Decisions

### 1. Modular Test Structure

**Decision**: Organize tests into separate modules for each major component.

**Rationale**:
- Improves maintainability by keeping related tests together
- Allows running tests for specific components during focused development
- Provides a clear mapping between package structure and test structure

**Implementation**:
- Created separate test files for each major class and module
- Used consistent naming patterns for test files
- Organized tests within each file by functionality

### 2. Comprehensive Test Coverage

**Decision**: Aim for high test coverage across all components.

**Rationale**:
- Ensures reliability of the package
- Catches regressions early in the development process
- Provides confidence in package behavior

**Implementation**:
- Tests for all public methods and functions
- Tests for both normal operation and error cases
- Tests for edge cases and boundary conditions

### 3. Isolated Testing Environment

**Decision**: Run tests in an isolated virtual environment.

**Rationale**:
- Prevents conflicts with system Python packages
- Ensures consistent testing environment
- Mimics typical user installation scenarios

**Implementation**:
- Created a virtual environment using `venv`
- Installed the package in development mode within the environment
- Documented environment setup in the README

### 4. Consistent Test Patterns

**Decision**: Use consistent patterns for test case implementation.

**Rationale**:
- Makes tests easier to understand and maintain
- Reduces cognitive load when switching between test modules
- Facilitates reuse of common testing patterns

**Implementation**:
- Used the Arrange-Act-Assert (AAA) pattern for test methods
- Defined test fixtures for common setup and teardown operations
- Used clear, descriptive test method names

## Test Case Examples

### Example from `test_settings.py`

```python
def test_add_binaries(self):
    """Test adding binary files to settings."""
    # Arrange
    settings = PAMpalSettings()
    binary_folder = "/path/to/binaries"
    
    # Act
    settings.add_binaries(binary_folder)
    
    # Assert
    self.assertEqual(settings.binaries["folder"], binary_folder)
    self.assertTrue(isinstance(settings.binaries["list"], list))
```

### Example from `test_acoustic_event.py`

```python
def test_add_detector(self):
    """Test adding detector data to an event."""
    # Arrange
    event = AcousticEvent("test_event")
    detector_data = pd.DataFrame({"UTC": [datetime.now()], "amplitude": [0.5]})
    
    # Act
    event.add_detector("ClickDetector", detector_data)
    
    # Assert
    self.assertIn("ClickDetector", event.detectors)
    self.assertTrue(len(event.detectors["ClickDetector"]) == 1)
```

### Example from `test_processing.py`

```python
def test_process_detections_with_empty_settings(self):
    """Test that process_detections raises error with empty settings."""
    # Arrange
    settings = PAMpalSettings()
    
    # Act & Assert
    with self.assertRaises(ValueError):
        process_detections(settings)
```

## Test Execution

Tests can be run individually or as a complete suite:

### Running All Tests

```bash
python tests/run_tests.py
```

### Running a Specific Test Module

```bash
python -m unittest tests/test_acoustic_event.py
```

### Running a Specific Test Case

```bash
python -m unittest tests.test_acoustic_event.TestAcousticEvent.test_add_detector
```

## Integration with Development Workflow

The testing framework is designed to integrate seamlessly with the development workflow:

1. **Test-Driven Development**: Tests can be written before implementation to guide development.

2. **Continuous Testing**: Tests can be run frequently during development to catch issues early.

3. **Regression Testing**: The complete test suite ensures that new changes don't break existing functionality.

4. **Documentation by Example**: Test cases serve as examples of how to use the package.

## Future Enhancements

1. **Test Coverage Reporting**: Add tools to measure and report test coverage.

2. **Integration Tests**: Add tests that verify the integration between components.

3. **Performance Tests**: Add tests to measure and verify performance characteristics.

4. **Parameterized Tests**: Use parameterized testing to test multiple input combinations.

5. **Test Data Generation**: Add utilities for generating test data for different scenarios.

## Conclusion

The PAMpal testing framework provides a robust foundation for ensuring the reliability and correctness of the package. The modular design and comprehensive coverage ensure that the package behaves as expected and maintains compatibility with the original R implementation.
