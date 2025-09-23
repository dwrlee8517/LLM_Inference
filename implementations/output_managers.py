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
from core.config import OutputConfig, PROJECT_ROOT

logger = logging.getLogger(__name__)

class JSONOutputManager(OutputManager):
    """Manages output in JSON format."""

    def __init__(self, config: OutputConfig):
        self.config = config
        logger.debug("Initialized JSONOutputManager")

    def get_output_path(self, filename: str = None) -> Path:
        """Construct the full output path for a file."""
        results_dir = PROJECT_ROOT / self.config.results_dir
        results_dir.mkdir(parents=True, exist_ok=True)
        
        base_filename = filename or self.config.default_filename

        # Add timestamp to the filename if specified
        if self.config.add_timestamp:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            stem = Path(base_filename).stem
            suffix = Path(base_filename).suffix
            return results_dir / f"{stem}_{timestamp}{suffix}"
        
        return results_dir / base_filename

    def save_results(self, results: Dict[str, Any], filename: str = None) -> str:
        """Save results to a JSON file."""
        output_path = self.get_output_path(filename)
        
        # Backup the existing file if specified
        if output_path.exists() and self.config.backup_existing:
            self.backup_existing_file(output_path)

        # Save the results to the file
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

        # If the file does not exist, return an empty dictionary
        if not output_path.exists():
            return {}
        
        # Load the results from the file
        try:
            with open(output_path, "r") as f:
                results = json.load(f)
            logger.info(f"Loaded {len(results)} existing results from {output_path}")
            return results
        except (json.JSONDecodeError, IOError):
            logger.warning(f"Could not parse existing results file {output_path}. Starting fresh.")
            return {}

    def backup_existing_file(self, file_path: Path) -> str:
        """Create a timestamped backup of an existing file.
        This is used to save the existing file with a timestamped backup.
        This is useful to avoid losing the existing file if the inference fails.
        It is also useful to avoid overwriting the existing file if the inference is successful.
        """
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        backup_path = file_path.with_name(f"{file_path.stem}_backup_{timestamp}{file_path.suffix}")

        # Create a timestamped backup of the existing file
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


class JSONLOutputManager(OutputManager):
    """Append-friendly JSONL output manager.

    Writes one JSON object per line: {"id": <id>, "result": <text>}.
    Also supports reading existing IDs to enable resume.
    """

    def __init__(self, config: OutputConfig):
        self.config = config
        logger.debug("Initialized JSONLOutputManager")

    def _get_jsonl_path(self, filename: str = None) -> Path:
        results_dir = PROJECT_ROOT / self.config.results_dir
        results_dir.mkdir(parents=True, exist_ok=True)
        base_filename = filename or self.config.default_filename
        # Force .jsonl extension
        stem = Path(base_filename).stem
        return results_dir / f"{stem}.jsonl"

    def save_results(self, results: Dict[str, Any], filename: str = None) -> str:
        """Write all results to JSONL (overwrites by default to consolidate)."""
        output_path = self._get_jsonl_path(filename)
        # Backup existing file if configured (will delete backup after successful write)
        backup_path_str = None
        if output_path.exists() and self.config.backup_existing:
            backup_path_str = self.backup_existing_file(output_path)

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                for _id, text in results.items():
                    f.write(json.dumps({"id": _id, "result": text}, ensure_ascii=False) + "\n")
            logger.info(f"Results saved successfully to {output_path}")
            # Remove backup after successful write, if requested earlier
            if backup_path_str:
                try:
                    Path(backup_path_str).unlink(missing_ok=True)
                except Exception as e:
                    logger.debug(f"Failed to delete backup {backup_path_str}: {e}")
            return str(output_path)
        except Exception as e:
            logger.error(f"Failed to save results to {output_path}: {e}")
            raise

    def append_result(self, _id: str, text: Any, filename: str = None) -> None:
        """Append a single result line safely."""
        output_path = self._get_jsonl_path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"id": _id, "result": text}, ensure_ascii=False) + "\n")

    def load_existing_results(self, filename: str = None) -> Dict[str, Any]:
        """Return dict of id->result from JSONL (may be large; primarily used for resume IDs)."""
        output_path = self._get_jsonl_path(filename)
        if not output_path.exists():
            return {}
        results: Dict[str, Any] = {}
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        _id = obj.get("id")
                        if _id is not None and _id not in results:
                            results[_id] = obj.get("result")
                    except json.JSONDecodeError:
                        continue
            logger.info(f"Loaded {len(results)} existing results from {output_path}")
            return results
        except Exception as e:
            logger.warning(f"Could not parse existing results file {output_path}: {e}")
            return {}

    def backup_existing_file(self, file_path: Path) -> str:
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
        # When resuming, caller should pass loaded results; otherwise, we can parse JSONL quickly
        return set(results.keys())


# Register the JSONL manager
ComponentFactory.register_output_manager("jsonl", JSONLOutputManager)