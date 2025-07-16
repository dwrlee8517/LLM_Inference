"""
Concrete implementations of data loaders for different data sources.
"""

import os
import pickle
import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
from abc import abstractmethod

from core.interfaces import DataLoader, ComponentFactory
from core.config import DataConfig

# Keep the import for PairedReportDataset as it's used by multiple loaders
from implementations.datasets import PairedReportDataset, SingleReportDataset

logger = logging.getLogger(__name__)


def get_ids_to_process(config: DataConfig, all_available_ids: List[str]) -> List[str]:
    """Helper function to determine which IDs to process based on configuration."""
    # Priority: id_file > custom_ids > process_all
    if config.id_file:
        id_file_path = Path(config.id_file)
        if id_file_path.exists():
            with open(id_file_path, 'r') as f:
                ids = [line.strip() for line in f if line.strip()]
            logger.info(f"Loaded {len(ids)} IDs from file: {id_file_path}")
            return ids
        else:
            logger.warning(f"ID file not found: {id_file_path}, falling back to other options")

    if config.custom_ids:
        logger.info(f"Using {len(config.custom_ids)} custom IDs")
        return config.custom_ids

    if config.process_all:
        logger.info(f"Processing all {len(all_available_ids)} available IDs")
        return all_available_ids

    # Default: process all
    logger.info(f"No ID selection specified, processing all {len(all_available_ids)} available IDs")
    return all_available_ids

class BaseFileDataLoader(DataLoader):
    """Abstract base class for data loaders that load data from files."""

    def __init__(self, config: DataConfig):
        self.config = config
        self.loaded_data = {}
        logger.info(f"Initialized {self.__class__.__name__} with config: {config}")

    def load_data(self) -> Dict[str, Any]:
        """Load data from files based on input_files configuration."""
        data = {}

        for file_config in self.config.input_files:
            file_path = os.path.join(self.config.data_folder, file_config['path'])

            if not os.path.exists(file_path):
                if file_config.get('required', False):
                    raise FileNotFoundError(f"Required file not found: {file_path}")
                else:
                    logger.warning(f"Optional file not found: {file_path}")
                    continue

            try:
                file_data = self._read_file(file_path, file_config) # Abstract method for specific file parsing

                data[file_config['name']] = file_data
                logger.info(f"Loaded {len(file_data)} entries from {file_path}")

            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
                if file_config.get('required', False):
                    raise

        self.loaded_data = data
        return data

    @abstractmethod
    def _read_file(self, file_path: str, file_config: Dict[str, Any]) -> Dict[str, Any]:
        """Abstract method to read and parse a specific file type."""
        pass

    def get_mrn_list(self) -> List[str]:
        """Get list of MRNs based on configuration from loaded data."""
        if not self.loaded_data:
            logger.warning("No data loaded, cannot get MRN list")
            return []

        # Get all available MRNs from the first available data source
        all_available_mrns = []
        for data_source in self.loaded_data.values():
            if isinstance(data_source, dict):
                all_available_mrns = list(data_source.keys())
                break

        if not all_available_mrns:
            logger.warning("No MRNs found in loaded data")
            return []

        # Use the helper function to determine which IDs to process
        return get_ids_to_process(self.config, all_available_mrns)

class BasePairedReportDataLoader(BaseFileDataLoader):
    """Base class for data loaders that handle paired radiology and biopsy reports."""

    def create_dataset(self, data: Dict[str, Any], mrns: List[str]) -> PairedReportDataset:
        """Create a dataset object from loaded data for paired reports."""
        radreports = data.get('radiology_reports', {})
        bxreports = data.get('biopsy_reports', {})

        # Filter MRNs to only include those present in both datasets
        valid_mrns = []
        for mrn in mrns:
            if mrn in radreports and mrn in bxreports:
                valid_mrns.append(mrn)
            else:
                logger.warning(f"MRN {mrn} not found in both datasets")

        logger.info(f"Created dataset with {len(valid_mrns)} valid MRNs")
        return PairedReportDataset(radreports, bxreports, valid_mrns)

    def validate_data(self, data: Dict[str, Any]) -> bool:
        """Validate the loaded data structure for paired reports."""
        required_keys = ['radiology_reports', 'biopsy_reports']

        for key in required_keys:
            if key not in data:
                logger.error(f"Missing required data key: {key}")
                return False

            if not isinstance(data[key], dict):
                logger.error(f"Data key {key} must be a dictionary")
                return False

        # Check for overlapping MRNs
        rad_mrns = set(data['radiology_reports'].keys())
        bx_mrns = set(data['biopsy_reports'].keys())

        overlap = rad_mrns & bx_mrns
        if not overlap:
            logger.error("No overlapping MRNs found between radiology and biopsy reports")
            return False

        logger.info(f"Data validation passed. {len(overlap)} overlapping MRNs found")
        return True

class PickleDataLoader(BasePairedReportDataLoader):
    """Data loader for pickle files."""

    def _read_file(self, file_path: str, file_config: Dict[str, Any]) -> Dict[str, Any]:
        """Read and parse a pickle file."""
        with open(file_path, 'rb') as f:
            return pickle.load(f)

class JSONDataLoader(BasePairedReportDataLoader):
    """Data loader for JSON files."""

    def _read_file(self, file_path: str, file_config: Dict[str, Any]) -> Dict[str, Any]:
        """Read and parse a JSON file."""
        with open(file_path, 'r') as f:
            return json.load(f)

class CSVDataLoader(BasePairedReportDataLoader):
    """Data loader for CSV files."""

    def _read_file(self, file_path: str, file_config: Dict[str, Any]) -> Dict[str, Any]:
        """Read and parse a CSV file."""
        df = pd.read_csv(file_path)
        mrn_col = file_config.get('mrn_column', 'mrn')
        text_col = file_config.get('text_column', 'text')

        if mrn_col not in df.columns or text_col not in df.columns:
            raise ValueError(f"Required columns not found in {file_path}")

        return dict(zip(df[mrn_col], df[text_col]))

class SingleReportDataLoader(BaseFileDataLoader):
    """Data loader for single report data."""

    def _read_file(self, file_path: str, file_config: Dict[str, Any]) -> Dict[str, Any]:
        """Read and parse a single report file based on type."""
        if file_path.endswith('.pkl'):
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        elif file_path.endswith('.json'):
            with open(file_path, 'r') as f:
                return json.load(f)
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
            mrn_col = file_config.get('mrn_column', 'mrn')
            text_col = file_config.get('text_column', 'report_text')
            return dict(zip(df[mrn_col], df[text_col]))
        else:
            raise ValueError(f"Unsupported file type for single report: {file_path}")

    def create_dataset(self, data: Dict[str, Any], mrns: List[str]) -> SingleReportDataset:
        """Create a dataset object from loaded data for single reports."""
        # Use the first available data source
        reports = None
        for data_source in data.values():
            if isinstance(data_source, dict):
                reports = data_source
                break

        if reports is None:
            raise ValueError("No valid data source found for single report dataset")

        valid_mrns = []
        for mrn in mrns:
            if mrn in reports:
                valid_mrns.append(mrn)
            else:
                logger.warning(f"MRN {mrn} not found in dataset")

        logger.info(f"Created dataset with {len(valid_mrns)} valid MRNs")
        return SingleReportDataset(reports, valid_mrns)

    def validate_data(self, data: Dict[str, Any]) -> bool:
        """Validate the loaded data structure for single reports."""
        if not data:
            logger.error("No data loaded")
            return False

        for key, value in data.items():
            if not isinstance(value, dict):
                logger.error(f"Data key {key} must be a dictionary")
                return False

        logger.info("Data validation passed")
        return True

# Register data loaders with the factory
ComponentFactory.register_data_loader("pickle", PickleDataLoader)
ComponentFactory.register_data_loader("json", JSONDataLoader)
ComponentFactory.register_data_loader("csv", CSVDataLoader)
ComponentFactory.register_data_loader("single_report", SingleReportDataLoader) 