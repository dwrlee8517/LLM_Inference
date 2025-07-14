"""
Concrete implementations of the OutputManager for various formats.
"""
import os
import json
import time
from pathlib import Path
from typing import Dict, Any, Set
import logging
from ruamel.yaml import YAML

from core.interfaces import OutputManager, ComponentFactory
from core.config import OutputConfig

logger = logging.getLogger(__name__)

class JSONOutputManager(OutputManager):
    """Manages output in JSON format."""

    def __init__(self, config: OutputConfig):
        self.config = config
        logger.info("Initialized JSONOutputManager")

    def get_output_path(self, filename: str = None) -> Path:
        """Construct the full output path for a file."""
        results_dir = Path(self.config.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        
        base_filename = filename or self.config.default_filename
        if self.config.add_timestamp:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            stem = Path(base_filename).stem
            suffix = Path(base_filename).suffix
            return results_dir / f"{stem}_{timestamp}{suffix}"
        
        return results_dir / base_filename

    def save_results(self, results: Dict[str, Any], filename: str = None) -> str:
        """Save results to a JSON file."""
        output_path = self.get_output_path(filename)
        
        if output_path.exists() and self.config.backup_existing:
            self.backup_existing_file(output_path)

        try:
            with open(output_path, "w") as f:
                json.dump(results, f, indent=4)
            logger.info(f"Results saved successfully to {output_path}")
            return str(output_path)
        except Exception as e:
            logger.error(f"Failed to save results to {output_path}: {e}")
            raise

    def load_existing_results(self, filename: str = None) -> Dict[str, Any]:
        """Load existing results from a JSON file."""
        output_path = self.get_output_path(filename)
        if not output_path.exists():
            return {}
        
        try:
            with open(output_path, "r") as f:
                results = json.load(f)
            logger.info(f"Loaded {len(results)} existing results from {output_path}")
            return results
        except (json.JSONDecodeError, IOError):
            logger.warning(f"Could not parse existing results file {output_path}. Starting fresh.")
            return {}

    def backup_existing_file(self, file_path: Path) -> str:
        """Create a timestamped backup of an existing file."""
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        backup_path = file_path.with_name(f"{file_path.stem}_backup_{timestamp}{file_path.suffix}")
        try:
            file_path.rename(backup_path)
            logger.info(f"Backed up existing file to {backup_path}")
            return str(backup_path)
        except OSError as e:
            logger.error(f"Failed to back up existing file: {e}")
            return ""

    def get_processed_ids(self, results: Dict[str, Any]) -> Set[str]:
        """Get the set of already processed IDs from the results dictionary."""
        return set(results.keys())

# Register the output manager with the factory
ComponentFactory.register_output_manager("json", JSONOutputManager) 