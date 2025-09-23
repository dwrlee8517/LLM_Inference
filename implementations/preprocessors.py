"""
Preprocessor implementations that turn dataset batches into raw inputs
consumable by ModelManagers. Supports file-based prompt templates.
"""

import os
from typing import Any, List, Tuple
from pathlib import Path
import logging

from ruamel.yaml import YAML

from core.interfaces import Preprocessor, ComponentFactory
from core.config import InferenceConfig, PROJECT_ROOT

logger = logging.getLogger(__name__)

class RadPathPreprocessor(Preprocessor):
    """
    File-driven preprocessor for paired (biopsy, radiology) reports.

    Produces per-sample raw inputs:
      - If config.multimodal is True: (system_text, [user_parts])
      - Else: (system_text, user_text)

    The YAML prompt file should contain keys:
      - system: str
      - user_template: str with placeholders {bxreport} and {radreport}
    """

    required_keys = {"radreport", "bxreport"}

    def __init__(self, config: InferenceConfig):
        self.config = config
        self.system_text = ""
        self.user_template = "{bxreport}\n{radreport}"

        if not getattr(config, "prompt_file", None):
            logger.warning("No inference.prompt_file set; using default minimal template")
            return

        prompt_path = Path(config.prompt_file)
        if not prompt_path.is_absolute():
            prompt_path = (PROJECT_ROOT / prompt_path).resolve()

        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

        yaml = YAML(typ="safe")
        data = yaml.load(prompt_path.read_text(encoding="utf-8")) or {}
        self.system_text = data.get("system", self.system_text)
        self.user_template = data.get("user_template", self.user_template)

        logger.info(f"Loaded prompt template from {prompt_path}")

    def prepare_inputs(self, batch: Any) -> Tuple[List[str], List[Any]]:
        # Accept dict samples or legacy tuples; normalize to dict
        if isinstance(batch, (list, tuple)) and batch and isinstance(batch[0], (list, tuple)):
            # DataLoader returns list of items per field when batch_size=1; flatten
            sample = [b[0] for b in batch]
        else:
            sample = batch

        if len(sample) == 3:
            mrn, rad, bx = sample
            # If rad/bx are lists of strings, join into a single string
            if isinstance(rad, list) and all(isinstance(s, str) for s in rad):
                rad = "\n".join(rad)
            if isinstance(bx, list) and all(isinstance(s, str) for s in bx):
                bx = "\n".join(bx)
            mrns = [mrn]
            radreports = [rad]
            bxreports = [bx]
        else:
            mrn, rad = sample
            if isinstance(rad, list) and all(isinstance(s, str) for s in rad):
                rad = "\n".join(rad)
            mrns = [mrn]
            radreports = [rad]
            bxreports = [None]
        raw_items: List[Any] = []

        # Paired dataset: (mrns, radreports, bxreports)
        if len(batch) == 3:
            _, radreports, bxreports = batch
            for bx, rad in zip(bxreports, radreports):
                user_text = self.user_template.format(bxreport=bx, radreport=rad)
                if self.config.multimodal:
                    raw_items.append((self.system_text, [user_text]))
                else:
                    raw_items.append((self.system_text, user_text))
        else:
            # Single report dataset: (mrns, reports)
            _, reports = batch
            for rep in reports:
                user_text = self.user_template.format(bxreport=rep, radreport="")
                if self.config.multimodal:
                    raw_items.append((self.system_text, [user_text]))
                else:
                    raw_items.append((self.system_text, user_text))

        return list(mrns), raw_items


# Single-report preprocessor (radiology-only)
class RadReportPreprocessor(Preprocessor):
    required_keys = {"radreport"}
    """
    File-driven preprocessor for radiology-only reports.

    Produces per-sample raw inputs:
      - If config.multimodal is True: (system_text, [user_text])
      - Else: (system_text, user_text)

    YAML prompt file keys:
      - system: str
      - user_template: str with placeholder {radreport}
    """

    def __init__(self, config: InferenceConfig):
        self.config = config
        self.system_text = ""
        self.user_template = "{radreport}"

        if not getattr(config, "prompt_file", None):
            logger.warning("No inference.prompt_file set; using default minimal template for RadReport")
            return

        prompt_path = Path(config.prompt_file)
        if not prompt_path.is_absolute():
            prompt_path = (PROJECT_ROOT / prompt_path).resolve()

        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

        yaml = YAML(typ="safe")
        data = yaml.load(prompt_path.read_text(encoding="utf-8")) or {}
        self.system_text = data.get("system", self.system_text)
        self.user_template = data.get("user_template", self.user_template)

        logger.info(f"Loaded prompt template from {prompt_path}")

    def prepare_inputs(self, batch: Any) -> Tuple[List[str], List[Any]]:
        # Normalize batch to a dict sample
        if isinstance(batch, (list, tuple)) and batch and isinstance(batch[0], (list, tuple)):
            sample = [b[0] for b in batch]
        else:
            sample = batch

        if len(sample) == 3:
            mrn, rad, _ = sample
            if isinstance(rad, list) and all(isinstance(s, str) for s in rad):
                rad = "\n".join(rad)
            mrns = [mrn]
            radreports = [rad]
        else:
            mrn, rad = sample
            if isinstance(rad, list) and all(isinstance(s, str) for s in rad):
                rad = "\n".join(rad)
            mrns = [mrn]
            radreports = [rad]
        raw_items: List[Any] = []

        # Paired dataset: (mrns, radreports, bxreports) — use only radreports
        if len(batch) == 3:
            _, radreports, _ = batch
            for rad in radreports:
                user_text = self.user_template.format(radreport=rad)
                if self.config.multimodal:
                    raw_items.append((self.system_text, [user_text]))
                else:
                    raw_items.append((self.system_text, user_text))
        else:
            # Single report dataset: (mrns, reports)
            _, reports = batch
            for rad in reports:
                user_text = self.user_template.format(radreport=rad)
                if self.config.multimodal:
                    raw_items.append((self.system_text, [user_text]))
                else:
                    raw_items.append((self.system_text, user_text))
        if batch == 1:
            return list(mrns), raw_items[0]
        return list(mrns), raw_items

# Register
ComponentFactory.register_preprocessor("RadPath", RadPathPreprocessor)
ComponentFactory.register_preprocessor("RadReport", RadReportPreprocessor)


