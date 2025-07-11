"""
Binary file parser for Pamguard PGDF binary files.

This module provides functionality to read and parse Pamguard binary files (.pgdf).
It handles different detector modules including Click Detector, Whistle and Moan Detector, 
and GPL Detector.
"""
import os
import struct
import datetime
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple, BinaryIO

# Constants for binary file parsing
PGDF_MAGIC = b'PGDF'

# Data type codes for PGDF files
DATA_TYPES = {
    0: 'UNKNOWN',
    1: 'SHORT',
    2: 'INT',
    3: 'LONG',
    4: 'FLOAT',
    5: 'DOUBLE',
    6: 'CHAR',
    7: 'BOOLEAN',
    8: 'STRING',
    9: 'BINARY',
    10: 'ARRAY'
}

class PamBinaryError(Exception):
    """Exception raised for errors in parsing Pamguard binary files."""
    pass


def read_string(file: BinaryIO) -> str:
    """
    Read a string from a binary file.
    
    Args:
        file: Binary file object
        
    Returns:
        String read from the file
    """
    str_len = struct.unpack('>h', file.read(2))[0]
    if str_len <= 0:
        return ""
    return file.read(str_len).decode('utf-8')


def read_pgdf_header(file_path: str) -> Dict[str, Any]:
    """
    Read the header of a Pamguard binary file (.pgdf).
    
    Args:
        file_path: Path to the binary file
        
    Returns:
        Dictionary containing header information
        
    Raises:
        FileNotFoundError: If the file does not exist
        PamBinaryError: If the file is not a valid Pamguard binary file
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    header = {}
    
    try:
        with open(file_path, 'rb') as f:
            # Read magic number
            magic = f.read(4)
            if magic != PGDF_MAGIC:
                raise PamBinaryError(f"Not a valid Pamguard binary file: {file_path}")
                
            # Read file format version
            header['version'] = struct.unpack('>i', f.read(4))[0]
            
            # Read file creation date (milliseconds since epoch)
            ms_since_epoch = struct.unpack('>q', f.read(8))[0]
            header['create_time'] = datetime.datetime.fromtimestamp(ms_since_epoch / 1000)
            
            # Read analysis time
            ms_since_epoch = struct.unpack('>q', f.read(8))[0]
            header['analysis_time'] = datetime.datetime.fromtimestamp(ms_since_epoch / 1000)
            
            # Read file type (module identifier)
            header['file_type'] = read_string(f)
            
            # Read module type
            header['module_type'] = read_string(f)
            
            # Read module name
            header['module_name'] = read_string(f)
            
            # Read stream name
            header['stream_name'] = read_string(f)
            
            # Record data blocks start position
            header['data_start'] = f.tell()
            
    except Exception as e:
        raise PamBinaryError(f"Error reading binary file header: {str(e)}")
        
    return header


def read_binary_data(file: BinaryIO, data_type: int) -> Any:
    """
    Read a data item from the binary file.
    
    Args:
        file: Binary file object
        data_type: Type code for the data to read
        
    Returns:
        The data item with appropriate type
        
    Raises:
        PamBinaryError: If the data type is unsupported
    """
    if data_type == 1:  # SHORT
        return struct.unpack('>h', file.read(2))[0]
    elif data_type == 2:  # INT
        return struct.unpack('>i', file.read(4))[0]
    elif data_type == 3:  # LONG
        return struct.unpack('>q', file.read(8))[0]
    elif data_type == 4:  # FLOAT
        return struct.unpack('>f', file.read(4))[0]
    elif data_type == 5:  # DOUBLE
        return struct.unpack('>d', file.read(8))[0]
    elif data_type == 6:  # CHAR
        return struct.unpack('c', file.read(1))[0].decode('utf-8')
    elif data_type == 7:  # BOOLEAN
        return struct.unpack('?', file.read(1))[0]
    elif data_type == 8:  # STRING
        return read_string(file)
    elif data_type == 9:  # BINARY
        data_len = struct.unpack('>i', file.read(4))[0]
        return file.read(data_len)
    elif data_type == 10:  # ARRAY
        # For arrays, need to read element type and array length
        element_type = struct.unpack('B', file.read(1))[0]
        array_length = struct.unpack('>i', file.read(4))[0]
        
        result = []
        for _ in range(array_length):
            result.append(read_binary_data(file, element_type))
        
        return result
    else:
        raise PamBinaryError(f"Unsupported data type: {data_type}")


def read_data_block(file: BinaryIO) -> Dict[str, Any]:
    """
    Read a single data block from a Pamguard binary file.
    
    Args:
        file: Binary file object
        
    Returns:
        Dictionary containing the data block content
        
    Raises:
        PamBinaryError: If there's an error reading the data block
    """
    # Read the data block header (fixed object header format)
    try:
        # Read milliseconds
        timestamp_bytes = file.read(8)
        if len(timestamp_bytes) < 8:  # Incomplete read, likely EOF
            return None
        milliseconds = struct.unpack('>q', timestamp_bytes)[0]
        
        # Read date as UTC milliseconds since 1970-01-01
        utc = datetime.datetime.fromtimestamp(milliseconds / 1000)
        
        # Read channel bitmap
        channels_bitmap = struct.unpack('>i', file.read(4))[0]
        
        # Read sequence number - identifies sequence of objects from a single source
        sequence_number = struct.unpack('>i', file.read(4))[0]
        
        # Read UID (Unique Identifier)
        uid = struct.unpack('>q', file.read(8))[0]
        
        # Read number of data items in this block
        n_items = struct.unpack('>h', file.read(2))[0]
        
        # Create data object
        data_block = {
            'UTC': utc,
            'milliseconds': milliseconds,
            'channels': channels_bitmap,
            'sequence': sequence_number,
            'UID': uid
        }
        
        # Read the specific data items for this block
        for _ in range(n_items):
            item_name = read_string(file)
            data_type = struct.unpack('B', file.read(1))[0]
            data_block[item_name] = read_binary_data(file, data_type)
            
        return data_block
        
    except Exception as e:
        raise PamBinaryError(f"Error reading data block: {str(e)}")


def read_click_detector_data(file_path: str) -> pd.DataFrame:
    """
    Read click detector data from a Pamguard binary file.
    
    Args:
        file_path: Path to the binary file
        
    Returns:
        DataFrame containing click detections
        
    Raises:
        FileNotFoundError: If the file does not exist
        PamBinaryError: If the file is not a valid Click Detector binary file
    """
    header = read_pgdf_header(file_path)
    
    # Check if this is a Click Detector file
    if 'Click' not in header['file_type']:
        raise PamBinaryError(f"Not a Click Detector binary file: {file_path}")
    
    data_blocks = []
    
    try:
        with open(file_path, 'rb') as f:
            # Go to the start of data blocks
            f.seek(header['data_start'])
            
            # Read until end of file
            while True:
                try:
                    data_block = read_data_block(f)
                    if data_block is None:  # EOF reached
                        break
                    data_blocks.append(data_block)
                except struct.error:
                    # End of file reached
                    break
    
    except Exception as e:
        raise PamBinaryError(f"Error reading Click Detector binary file: {str(e)}")
    
    # Convert to DataFrame if we have data
    if data_blocks:
        df = pd.DataFrame(data_blocks)
        return df
    else:
        # Return empty DataFrame with expected columns
        return pd.DataFrame(columns=['UTC', 'milliseconds', 'channels', 'sequence', 'UID'])


def read_whistles_moans_data(file_path: str) -> pd.DataFrame:
    """
    Read Whistle and Moan detector data from a Pamguard binary file.
    
    Args:
        file_path: Path to the binary file
        
    Returns:
        DataFrame containing whistle detections
        
    Raises:
        FileNotFoundError: If the file does not exist
        PamBinaryError: If the file is not a valid Whistle and Moan Detector binary file
    """
    header = read_pgdf_header(file_path)
    
    # Check if this is a Whistles & Moans file
    if 'WhistlesMoans' not in header['file_type']:
        raise PamBinaryError(f"Not a Whistles & Moans Detector binary file: {file_path}")
    
    data_blocks = []
    
    try:
        with open(file_path, 'rb') as f:
            # Go to the start of data blocks
            f.seek(header['data_start'])
            
            # Read until end of file
            while True:
                try:
                    data_block = read_data_block(f)
                    if data_block is None:  # EOF reached
                        break
                    # Process contour data if present - stored in 'contour' field
                    if 'contour' in data_block and isinstance(data_block['contour'], list):
                        # Convert contour to NumPy array for easier processing
                        contour_array = np.array(data_block['contour'])
                        if len(contour_array) > 0:
                            # Store additional metrics about the contour
                            data_block['min_freq'] = np.min(contour_array)
                            data_block['max_freq'] = np.max(contour_array)
                            data_block['mean_freq'] = np.mean(contour_array)
                            data_block['contour_points'] = len(contour_array)
                    
                    data_blocks.append(data_block)
                except struct.error:
                    # End of file reached
                    break
    
    except Exception as e:
        raise PamBinaryError(f"Error reading Whistles & Moans binary file: {str(e)}")
    
    # Convert to DataFrame if we have data
    if data_blocks:
        df = pd.DataFrame(data_blocks)
        return df
    else:
        # Return empty DataFrame with expected columns
        return pd.DataFrame(columns=['UTC', 'milliseconds', 'channels', 'sequence', 'UID'])


def read_gpl_detector_data(file_path: str) -> pd.DataFrame:
    """
    Read GPL detector data from a Pamguard binary file.
    
    Args:
        file_path: Path to the binary file
        
    Returns:
        DataFrame containing GPL detections
        
    Raises:
        FileNotFoundError: If the file does not exist
        PamBinaryError: If the file is not a valid GPL Detector binary file
    """
    header = read_pgdf_header(file_path)
    
    # Check if this is a GPL Detector file
    if 'GPL' not in header['file_type']:
        raise PamBinaryError(f"Not a GPL Detector binary file: {file_path}")
    
    data_blocks = []
    
    try:
        with open(file_path, 'rb') as f:
            # Go to the start of data blocks
            f.seek(header['data_start'])
            
            # Read until end of file
            while True:
                try:
                    data_block = read_data_block(f)
                    if data_block is None:  # EOF reached
                        break
                    data_blocks.append(data_block)
                except struct.error:
                    # End of file reached
                    break
    
    except Exception as e:
        raise PamBinaryError(f"Error reading GPL Detector binary file: {str(e)}")
    
    # Convert to DataFrame if we have data
    if data_blocks:
        df = pd.DataFrame(data_blocks)
        return df
    else:
        # Return empty DataFrame with expected columns
        return pd.DataFrame(columns=['UTC', 'milliseconds', 'channels', 'sequence', 'UID'])


def read_binary_file(file_path: str) -> pd.DataFrame:
    """
    Read data from any supported Pamguard binary file.
    
    This function determines the file type and calls the appropriate specialized parser.
    
    Args:
        file_path: Path to the binary file
        
    Returns:
        DataFrame containing detections
        
    Raises:
        FileNotFoundError: If the file does not exist
        PamBinaryError: If the file format is not supported
    """
    try:
        header = read_pgdf_header(file_path)
        
        if 'Click' in header['file_type']:
            return read_click_detector_data(file_path)
        elif 'WhistlesMoans' in header['file_type']:
            return read_whistles_moans_data(file_path)
        elif 'GPL' in header['file_type']:
            return read_gpl_detector_data(file_path)
        else:
            raise PamBinaryError(f"Unsupported binary file type: {header['file_type']}")
            
    except Exception as e:
        raise PamBinaryError(f"Error reading binary file {file_path}: {str(e)}")


def load_binary_files(file_paths: List[str], verbose: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Load data from multiple Pamguard binary files.
    
    Args:
        file_paths: List of paths to binary files
        verbose: Whether to print progress information
        
    Returns:
        Dictionary mapping detector types to DataFrames of detections
    """
    if verbose:
        print(f"Loading {len(file_paths)} binary files...")
    
    result = {}
    
    for file_path in file_paths:
        try:
            header = read_pgdf_header(file_path)
            detector_type = header['file_type']
            
            if verbose:
                print(f"Reading {os.path.basename(file_path)} - {detector_type}")
                
            df = read_binary_file(file_path)
            
            # Add file information
            df['BinaryFile'] = os.path.basename(file_path)
            
            # Append to existing data or create new entry
            if detector_type in result:
                result[detector_type] = pd.concat([result[detector_type], df], ignore_index=True)
            else:
                result[detector_type] = df
                
        except Exception as e:
            if verbose:
                print(f"Error processing {file_path}: {str(e)}")
    
    return result
