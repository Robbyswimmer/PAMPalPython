"""
PAMpalSettings class module.

This module contains the PAMpalSettings class, which is the core configuration class 
for the PAMpal package. It stores settings related to all processing and analysis steps.
"""
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

from .calibration import CalibrationManager, CalibrationFunction, load_calibration_file, CalibrationError


class PAMpalSettings:
    """
    A class that stores settings related to all processing and analysis steps in PAMpal.
    
    This class is the main configuration object for all major functions in the PAMpal package.
    
    Attributes:
        db (str): Full path to a PamGuard database file
        binaries (Dict): Dictionary with keys 'folder' containing directory paths and 'list' 
            containing paths to individual binary files
        functions (Dict): Dictionary of functions to apply to data read in by PAMpal, 
            organized by PamGuard module
        calibration (Dict): Dictionary of calibration functions to apply while processing,
            organized by PamGuard module
        settings (Dict): Dictionary of settings, usually imported from Pamguard's
            "Export XML Configuration"
    """

    def __init__(self):
        """
        Initialize a PAMpalSettings object with default values.
        """
        # Initialize with empty values
        self.db = ""
        self.binaries = {"folder": [], "list": []}
        self.functions = {
            "ClickDetector": {},
            "WhistlesMoans": {},
            "Cepstrum": {}
        }
        # Initialize calibration manager
        self.calibration_manager = CalibrationManager()
        # Keep the old calibration dict for backwards compatibility
        self.calibration = {"ClickDetector": {}}
        self.settings = {
            "file": None,
            "sources": {},
            "detectors": {},
            "raw": None
        }

    def add_database(self, db_path: str) -> None:
        """
        Add a PamGuard database file to the settings.
        
        Args:
            db_path: Path to the PamGuard database file
            
        Raises:
            FileNotFoundError: If the database file does not exist
        """
        db_path = os.path.abspath(db_path)
        if not os.path.isfile(db_path):
            raise FileNotFoundError(f"Database file not found: {db_path}")
        self.db = db_path
        print(f"Added database: {db_path}")

    def add_binaries(self, folder_path: str) -> None:
        """
        Add PamGuard binary files from a directory.
        
        Args:
            folder_path: Path to directory containing binary files
            
        Raises:
            FileNotFoundError: If the directory does not exist
        """
        folder_path = os.path.abspath(folder_path)
        if not os.path.isdir(folder_path):
            raise FileNotFoundError(f"Binary folder not found: {folder_path}")
            
        # Add the folder to the binaries folder list
        self.binaries["folder"].append(folder_path)
        
        # Find all binary files in the folder
        binary_files = []
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith('.pgdf') or file.endswith('.binf'):
                    binary_files.append(os.path.join(root, file))
        
        if not binary_files:
            print(f"Warning: No binary files found in {folder_path}")
        else:
            self.binaries["list"].extend(binary_files)
            print(f"Added {len(binary_files)} binary files from {folder_path}")
    
    def add_function(self, module: str, name: str, function) -> None:
        """
        Add a processing function to apply to detections from a specific module.
        
        Args:
            module: The PamGuard module name (e.g., 'ClickDetector')
            name: The name to assign to the function
            function: The function to apply to detections
            
        Raises:
            ValueError: If the module is not supported
        """
        if module not in self.functions:
            raise ValueError(f"Unsupported module: {module}. Must be one of {list(self.functions.keys())}")
            
        if name in self.functions[module]:
            print(f"Warning: Overwriting existing function '{name}' for module '{module}'")
            
        self.functions[module][name] = function
        print(f"Added function '{name}' for module '{module}'")
    
    def add_calibration(self, module: str, name: str, function) -> None:
        """
        Add a calibration function to apply to detections from a specific module.
        
        Args:
            module: The PamGuard module name (e.g., 'ClickDetector')
            name: The name to assign to the function
            function: The calibration function to apply to detections
            
        Raises:
            ValueError: If the module is not supported
        """
        # Support both CalibrationFunction objects and legacy function objects
        if isinstance(function, CalibrationFunction):
            # Add to new calibration manager
            self.calibration_manager.add_calibration(module, name, function)
        
        # Also maintain backwards compatibility with old calibration dict
        if module not in self.calibration:
            self.calibration[module] = {}
            
        if name in self.calibration[module]:
            print(f"Warning: Overwriting existing calibration function '{name}' for module '{module}'")
            
        self.calibration[module][name] = function
        print(f"Added calibration function '{name}' for module '{module}'")
    
    def add_calibration_file(self, file_path: str, module: str = "ClickDetector", 
                            name: str = None, unit_type: int = 3, 
                            voltage_range: Optional[float] = None, 
                            bit_rate: Optional[int] = None, 
                            apply_to_all: bool = False) -> None:
        """
        Add calibration from a CSV file.
        
        Args:
            file_path: Path to the CSV calibration file
            module: PamGuard module name (e.g., 'ClickDetector')
            name: Name for the calibration function (defaults to filename)
            unit_type: Unit type (1=dB re V/uPa, 2=uPa/Counts, 3=uPa/FullScale)
            voltage_range: Voltage range for unit_type=1
            bit_rate: Bit rate for unit_type=2
            apply_to_all: If True, apply calibration to all detector types
            
        Raises:
            CalibrationError: If calibration file cannot be loaded
        """
        try:
            # Load calibration function from file
            calibration_function = load_calibration_file(
                file_path=file_path,
                unit_type=unit_type,
                voltage_range=voltage_range,
                bit_rate=bit_rate,
                name=name
            )
            
            # Add to specified module
            self.add_calibration(module, calibration_function.name, calibration_function)
            
            # Add to all modules if requested
            if apply_to_all:
                for module_name in self.functions.keys():
                    if module_name != module:
                        self.add_calibration(module_name, calibration_function.name, calibration_function)
                        
        except CalibrationError as e:
            print(f"Error loading calibration file: {str(e)}")
            raise
    
    def remove_calibration(self, module: str, name: str = None) -> bool:
        """
        Remove calibration function(s) from a module.
        
        Args:
            module: PamGuard module name
            name: Name of calibration function (removes all if None)
            
        Returns:
            True if calibration was removed, False if not found
        """
        removed = False
        
        # Remove from calibration manager
        if self.calibration_manager.remove_calibration(module, name):
            removed = True
        
        # Remove from legacy calibration dict
        if module in self.calibration:
            if name is None:
                if self.calibration[module]:
                    self.calibration[module] = {}
                    removed = True
            elif name in self.calibration[module]:
                del self.calibration[module][name]
                removed = True
        
        if removed:
            if name is None:
                print(f"Removed all calibration functions for module '{module}'")
            else:
                print(f"Removed calibration function '{name}' for module '{module}'")
        
        return removed
    
    def get_calibration(self, module: str, name: str = None) -> Optional[CalibrationFunction]:
        """
        Get a calibration function for a module.
        
        Args:
            module: PamGuard module name
            name: Name of calibration function (returns first if None)
            
        Returns:
            CalibrationFunction object or None if not found
        """
        return self.calibration_manager.get_calibration(module, name)
    
    def list_calibrations(self) -> Dict[str, List[str]]:
        """
        List all available calibration functions.
        
        Returns:
            Dictionary mapping module names to lists of calibration function names
        """
        return self.calibration_manager.list_calibrations()
    
    def has_calibration(self, module: str, name: str = None) -> bool:
        """
        Check if a calibration function exists.
        
        Args:
            module: PamGuard module name
            name: Name of calibration function (checks any if None)
            
        Returns:
            True if calibration exists, False otherwise
        """
        return self.calibration_manager.has_calibration(module, name)
        
    def add_settings(self, settings_file: str) -> None:
        """
        Add Pamguard settings from an XML configuration file.
        
        Args:
            settings_file: Path to the XML settings file
            
        Raises:
            FileNotFoundError: If the settings file does not exist
        """
        settings_file = os.path.abspath(settings_file)
        if not os.path.isfile(settings_file):
            raise FileNotFoundError(f"Settings file not found: {settings_file}")
            
        # In a real implementation, we would parse the XML file here
        # For now, just store the file path
        self.settings["file"] = settings_file
        print(f"Added settings file: {settings_file}")
        
    def __str__(self) -> str:
        """String representation of the PAMpalSettings object."""
        num_db = 1 if self.db else 0
        num_bin_dir = len(self.binaries["folder"])
        num_bin_files = len(self.binaries["list"])
        
        result = [f"PAMpalSettings object with:"]
        
        # Database info
        result.append(f"{num_db} database(s)")
        if self.db:
            result.append(f"  {os.path.basename(self.db)}")
            
        # Binary info
        result.append(f"{num_bin_dir} binary folder(s) containing {num_bin_files} binary files")
            
        # Functions info
        for module, funcs in self.functions.items():
            num_funcs = len(funcs)
            result.append(f"{num_funcs} function(s) for module type \"{module}\"")
            for func_name, func in funcs.items():
                result.append(f"  \"{func_name}\"")
                result.append(f"    {str(func)}")
                
        # Calibration info
        calibration_list = self.calibration_manager.list_calibrations()
        total_calibrations = sum(len(names) for names in calibration_list.values())
        result.append(f"{total_calibrations} calibration function(s)")
        
        for module, cal_names in calibration_list.items():
            if cal_names:
                result.append(f"  {module}: {', '.join(cal_names)}")
        
        # Settings info
        if self.settings["file"]:
            result.append(f"1 settings file:")
            result.append(f"  {os.path.basename(self.settings['file'])}")
        else:
            result.append("0 settings files")
            
        return "\n".join(result)
        
    def __repr__(self) -> str:
        """Developer string representation."""
        return self.__str__()
