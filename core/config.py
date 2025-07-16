"""
Configuration management for LLM inference system.
Handles loading, validation, and merging of configuration from multiple sources.
"""

import os
import yaml
import argparse
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from pathlib import Path
import logging
import sys
import logging.handlers

logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Configuration for data loading and processing."""
    source_type: str
    data_folder: str
    input_files: List[Dict[str, Any]]
    process_all: bool
    custom_ids: List[str]
    id_file: Optional[str]
    resume_from_existing: bool
    check_processed: bool

@dataclass
class QuantizationConfig:
    """Configuration for model quantization."""
    enabled: bool
    bits: Optional[int]

@dataclass
class ModelConfig:
    """Configuration for model settings."""
    cache_dir: str
    model_name: str
    quantization: QuantizationConfig
    cuda_devices: str
    max_memory: Optional[Dict[str, str]]

@dataclass
class GenerationConfig:
    """Configuration for text generation parameters."""
    do_sample: bool
    temperature: Optional[float]
    top_p: Optional[float]
    top_k: Optional[int]
    max_tokens: int


@dataclass
class InferenceConfig:
    """Configuration for inference parameters."""
    prompt_template: str
    generation: GenerationConfig
    batch_size: int
    shuffle: bool
    multimodal: bool

@dataclass
class OutputConfig:
    """Configuration for output settings."""
    results_dir: str
    default_filename: str
    format: str
    save_elapsed_time: bool
    save_intermediate_results: bool
    backup_existing: bool
    add_timestamp: bool


@dataclass
class LoggingConfig:
    """Configuration for logging settings."""
    level: str
    file: str
    console: bool
    progress_bar: bool
    gpu_monitoring: bool


@dataclass
class SystemConfig:
    """Configuration for system settings."""
    num_workers: int
    pin_memory: bool
    continue_on_error: bool
    max_retries: int
    retry_delay: int
    memory_threshold: float
    gpu_memory_threshold: float

class ConfigManager:
    """Manages configuration loading and validation."""
    
    def __init__(self, config_path: Optional[str] = None):

        if not config_path.startswith("configs/"):
            # If the configuration yaml file is not in the configs/ directory, ask the user if they want to use the default configuration file directory
            user_input = input("Is the configuration yaml file in the configs/ directory? (y/n): ")
            if user_input.lower() in ["y", "yes"]:
                self.config_path = f"configs/{config_path}"
            else:
                self.config_path = config_path
        else:
            self.config_path = config_path

        self.config = self._load_config()
        self._validate_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {self.config_path}")
            return config
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {self.config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML configuration: {e}")
            raise
    
    def _validate_config(self) -> None:
        """Validate configuration structure and values."""
        required_sections = ['data', 'model', 'inference', 'output', 'logging', 'system']
        
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        # Validate data configuration
        data_config = self.config['data']
        if not os.path.exists(data_config['data_folder']):
            logger.warning(f"Data folder does not exist: {data_config['data_folder']}")
        
        # Validate model configuration
        model_config = self.config['model']
        if not os.path.exists(model_config['cache_dir']):
            os.makedirs(model_config['cache_dir'], exist_ok=True)
            logger.info(f"Created cache directory: {model_config['cache_dir']}")
        
        # Validate output configuration
        output_config = self.config['output']
        if not os.path.exists(output_config['results_dir']):
            os.makedirs(output_config['results_dir'], exist_ok=True)
            logger.info(f"Created results directory: {output_config['results_dir']}")
    
    def get_data_config(self) -> DataConfig:
        """Get data configuration as a dataclass."""
        return DataConfig(**self.config['data'])
    
    def get_model_config(self) -> ModelConfig:
        """Get model configuration as a dataclass."""
        model_conf = self.config['model']
        quantization_conf = model_conf.get('quantization', {})
        return ModelConfig(
            cache_dir=model_conf['cache_dir'],
            model_name=model_conf['model_name'],
            quantization=QuantizationConfig(
                enabled=quantization_conf.get('enabled', False),
                bits=quantization_conf.get('bits', None)
            ),
            cuda_devices=model_conf['cuda_devices'],
            max_memory=model_conf.get('max_memory')
        )
    
    def get_inference_config(self) -> InferenceConfig:
        """Get inference configuration as a dataclass."""
        inference_conf = self.config['inference']
        generation_conf = inference_conf.get('generation', {})
        return InferenceConfig(
            prompt_template=inference_conf['prompt_template'],
            generation=GenerationConfig(
                do_sample=generation_conf.get('do_sample', False),
                temperature=generation_conf.get('temperature', None),
                top_p=generation_conf.get('top_p', None),
                top_k=generation_conf.get('top_k', None),
                max_tokens=generation_conf['max_tokens']
            ),
            batch_size=inference_conf['batch_size'],
            shuffle=inference_conf['shuffle'],
            multimodal=inference_conf['multimodal']
        )
    
    def get_output_config(self) -> OutputConfig:
        """Get output configuration as a dataclass."""
        return OutputConfig(**self.config['output'])
    
    def get_logging_config(self) -> LoggingConfig:
        """Get logging configuration as a dataclass."""
        return LoggingConfig(**self.config['logging'])
    
    def get_system_config(self) -> SystemConfig:
        """Get system configuration as a dataclass."""
        return SystemConfig(**self.config['system'])
    
    def override_config(self, overrides: Dict[str, Any]) -> None:
        """Override configuration values with provided dictionary."""
        def deep_update(base_dict, update_dict):
            for key, value in update_dict.items():
                if isinstance(value, dict) and key in base_dict:
                    deep_update(base_dict[key], value)
                else:
                    base_dict[key] = value
        
        deep_update(self.config, overrides)
        self._validate_config()
    
    def update_from_args(self, args: argparse.Namespace) -> None:
        """Update configuration from command-line arguments."""
        overrides = {}
        
        # Map command-line arguments to configuration keys
        arg_mappings = {
            'data_folder': ['data', 'data_folder'],
            'model_name': ['model', 'model_name'],
            'cache_dir': ['model', 'cache_dir'],
            'cuda_devices': ['model', 'cuda_devices'],
            'quantization': ['model', 'quantization', 'bits'], # Map to bits
            'batch_size': ['inference', 'batch_size'],
            'prompt_template': ['inference', 'prompt_template'],
            'max_tokens': ['inference', 'generation', 'max_tokens'],
            'output_file': ['output', 'default_filename'],
            'results_dir': ['output', 'results_dir'],
            'log_level': ['logging', 'level'],
            'process_all': ['data', 'process_all'],
            'custom_ids': ['data', 'custom_ids'],
            'id_file': ['data', 'id_file'],
            'format': ['output', 'format'],
        }
        
        for arg_name, config_path in arg_mappings.items():
            if hasattr(args, arg_name) and getattr(args, arg_name) is not None:
                # Special handling for quantization enabling
                if arg_name == 'quantization':
                    current = overrides
                    for key in config_path[:-2]: # Navigate to 'model' key
                        if key not in current:
                            current[key] = {}
                        current = current[key]
                    if 'quantization' not in current:
                        current['quantization'] = {}
                    current['quantization']['enabled'] = True # Enable quantization if bits are provided
                    current['quantization']['bits'] = getattr(args, arg_name)
                else:
                    # Navigate to the correct nested dictionary
                    current = overrides
                    for key in config_path[:-1]:
                        if key not in current:
                            current[key] = {}
                        current = current[key]
                    current[config_path[-1]] = getattr(args, arg_name)

        # Handle --no-resume separately
        if hasattr(args, 'no_resume') and getattr(args, 'no_resume'):
            if 'data' not in overrides:
                overrides['data'] = {}
            overrides['data']['resume_from_existing'] = False

        if overrides:
            self.override_config(overrides)
    
    def get_full_config(self) -> Dict[str, Any]:
        """Get the complete configuration dictionary."""
        return self.config.copy()
    
    def save_config(self, path: str) -> None:
        """Save current configuration to a file."""
        with open(path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)
        logger.info(f"Configuration saved to {path}")


def create_argument_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="LLM Batch Inference System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Configuration file
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )
    
    # Data arguments
    parser.add_argument(
        '--data-folder',
        type=str,
        help='Path to data folder'
    )
    
    parser.add_argument(
        '--process-all',
        action='store_true',
        help='Process all available IDs'
    )
    
    parser.add_argument(
        '--custom-ids',
        type=str,
        nargs='+',
        help='List of specific IDs to process'
    )
    
    parser.add_argument(
        '--id-file',
        type=str,
        help='Path to file containing IDs to process (one per line)'
    )
    
    # Model arguments
    parser.add_argument(
        '--model-name',
        type=str,
        help='Name of the model to use'
    )
    
    parser.add_argument(
        '--cache-dir',
        type=str,
        help='Model cache directory'
    )
    
    parser.add_argument(
        '--cuda-devices',
        type=str,
        help='CUDA devices to use (e.g., "0,1")'
    )
    
    parser.add_argument(
        '--quantization',
        type=int,
        choices=[4, 8],
        help='Quantization bits (4 or 8)'
    )
    
    # Inference arguments
    parser.add_argument(
        '--batch-size',
        type=int,
        help='Batch size for inference'
    )
    
    parser.add_argument(
        '--prompt-template',
        type=str,
        help='Prompt template to use'
    )
    
    parser.add_argument(
        '--max-tokens',
        type=int,
        help='Maximum tokens to generate'
    )
    
    # Output arguments
    parser.add_argument(
        '--output-file',
        type=str,
        help='Output file name'
    )
    
    parser.add_argument(
        '--results-dir',
        type=str,
        help='Results directory'
    )
    
    parser.add_argument(
        '--format',
        type=str,
        choices=['json', 'yaml', 'csv'],
        help='Output format'
    )
    
    # System arguments
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    parser.add_argument(
        '--no-resume',
        action='store_true',
        help='Do not resume from existing results'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show configuration and exit without running inference'
    )
    
    parser.add_argument(
        '--list-models',
        action='store_true',
        help='List all supported models and exit'
    )
    
    return parser


def setup_logging(config: LoggingConfig) -> None:
    """Setup logging configuration."""
    level = getattr(logging, config.level.upper())
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    if config.console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # File handler
    if config.file:
        file_handler = logging.FileHandler(config.file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler) 