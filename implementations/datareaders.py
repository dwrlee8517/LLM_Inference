"""
Concrete implementations of data loaders for different data sources.
"""

import os
import pickle
import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import logging
from abc import abstractmethod
from torch.utils.data import Dataset

from core.interfaces import DataReader, ComponentFactory
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

class BaseFileReader(DataReader):
    """Abstract base class for data readers that load data from files."""

    def __init__(self, config: DataConfig):
        self.config = config
        self.loaded_data = {}
        logger.info(f"Initialized {self.__class__.__name__} with config: {config}")

    def load_data(self) -> Tuple[Dataset, List[str]]:
        """Load data from files based on input_files configuration and return Dataset and IDs."""
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
        # Call get_all_ids here, which internally calls get_ids_to_process once.
        all_ids = self.get_all_ids()
        dataset = self._create_dataset()
        return dataset, all_ids

    @abstractmethod
    def _read_file(self, file_path: str, file_config: Dict[str, Any]) -> Dict[str, Any]:
        """Abstract method to read and parse a specific file type."""
        pass

    def get_all_ids(self) -> List[str]:
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

    @abstractmethod
    def _create_dataset(self) -> Dataset:
        """Abstract method to create a PyTorch Dataset from loaded data."""
        pass

class BasePairedReportReader(BaseFileReader):
    """Base class for data readers that handle paired radiology and biopsy reports."""
    pass

class PairedReportDataReader(BasePairedReportReader):
    """Unified data reader for paired reports across pickle/json/csv.

    Expects two input files configured (radiology_reports, biopsy_reports) of matching ids.
    """

    def _read_file(self, file_path: str, file_config: Dict[str, Any]) -> Dict[str, Any]:
        if file_path.endswith('.pkl'):
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        elif file_path.endswith('.json'):
            with open(file_path, 'r') as f:
                return json.load(f)
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
            mrn_col = file_config.get('mrn_column', 'mrn')
            text_col = file_config.get('text_column', 'text')
            if mrn_col not in df.columns or text_col not in df.columns:
                raise ValueError(f"Required columns not found in {file_path}")
            return dict(zip(df[mrn_col], df[text_col]))
        else:
            raise ValueError(f"Unsupported file type: {file_path}")

    def _create_dataset(self) -> PairedReportDataset:
        """Create a dataset object from loaded data."""
        radreports = self.loaded_data.get('radiology_reports', {})
        bxreports = self.loaded_data.get('biopsy_reports', {})
        
        # Get the list of MRNs to process from the configuration
        all_ids = self.get_all_ids()
        
        return PairedReportDataset(radreports, bxreports, all_ids)

class SingleReportDataReader(BaseFileReader):
    """Data reader for single report data."""

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

    def _create_dataset(self) -> SingleReportDataset:
        """Create a dataset object from loaded data for single reports."""
        # Use the first available data source
        reports = None
        for data_source in self.loaded_data.values():
            if isinstance(data_source, dict):
                reports = data_source
                break

        if reports is None:
            raise ValueError("No valid data source found for single report dataset")

        all_ids = self.get_all_ids()
        
        return SingleReportDataset(reports, all_ids)

# Register data readers with the factory
ComponentFactory.register_data_reader("paired_report", PairedReportDataReader)
ComponentFactory.register_data_reader("single_report", SingleReportDataReader)