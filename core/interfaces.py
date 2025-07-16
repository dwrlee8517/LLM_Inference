"""
Abstract base classes and interfaces for the LLM inference system.
These interfaces allow for easy extension and customization of components.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Iterator, Tuple
from dataclasses import dataclass
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import GenerationConfig, DataConfig, ModelConfig, OutputConfig, InferenceConfig # Import the new dataclasses
import logging

logger = logging.getLogger(__name__)

@dataclass
class InferenceRequest:
    """Represents a single inference request."""
    id: str
    data: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class InferenceResult:
    """Represents the result of an inference request."""
    id: str
    result: Any
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time: Optional[float] = None


class DataLoader(ABC):
    """Abstract base class for data loaders."""
    
    @abstractmethod
    def load_data(self, config: DataConfig) -> Dict[str, Any]:
        """Load data from the configured source."""
        pass
    
    @abstractmethod
    def get_mrn_list(self) -> List[str]:
        """Get list of MRNs based on configuration."""
        pass
    
    @abstractmethod
    def create_dataset(self, data: Dict[str, Any], mrns: List[str]) -> Any:
        """Create a dataset object from loaded data."""
        pass
    
    @abstractmethod
    def validate_data(self, data: Dict[str, Any]) -> bool:
        """Validate the loaded data structure."""
        pass


class ModelManager(ABC):
    """Abstract base class for model management."""
    
    @abstractmethod
    def load_model(self, config: ModelConfig) -> Tuple[Any, Any]:
        """Load model and tokenizer."""
        pass
    
    @abstractmethod
    def generate(self, 
                 tokenizer: Any, 
                 model: Any, 
                 messages: List[Dict[str, str]], 
                 generation_config: GenerationConfig) -> str:
        """Generate text using the model."""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Clean up model resources."""
        pass


class PromptManager(ABC):
    """Abstract base class for prompt management."""
    
    @abstractmethod
    def get_prompt_template(self, template_name: str) -> callable:
        """Get a prompt template function by name."""
        pass
    
    @abstractmethod
    def create_messages(self, 
                       template_name: str, 
                       data: Any) -> List[Dict[str, str]]:
        """Create messages for the model using the specified template."""
        pass
    
    @abstractmethod
    def list_available_templates(self) -> List[str]:
        """List all available prompt templates."""
        pass
    
    @abstractmethod
    def validate_template(self, template_name: str) -> bool:
        """Validate that a template exists and is callable."""
        pass


class OutputManager(ABC):
    """Abstract base class for output management."""
    
    @abstractmethod
    def save_results(self, 
                     results: Dict[str, Any], 
                     output_config: OutputConfig) -> str:
        """Save results to the configured output format."""
        pass
    
    @abstractmethod
    def load_existing_results(self, file_path: str) -> Dict[str, Any]:
        """Load existing results from a file."""
        pass
    
    @abstractmethod
    def backup_existing_file(self, file_path: str) -> str:
        """Create a backup of an existing file."""
        pass
    
    @abstractmethod
    def get_processed_ids(self, results: Dict[str, Any]) -> set:
        """Get set of already processed IDs from results."""
        pass


class InferenceEngine(ABC):
    """Abstract base class for inference engines."""
    
    @abstractmethod
    def setup(self, config: InferenceConfig) -> None:
        """Setup the inference engine with configuration."""
        pass
    
    @abstractmethod
    def process_batch(self, 
                      batch: List[InferenceRequest]) -> List[InferenceResult]:
        """Process a batch of inference requests."""
        pass
    
    @abstractmethod
    def process_single(self, 
                       request: InferenceRequest) -> InferenceResult:
        """Process a single inference request."""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        pass


class MonitoringSystem(ABC):
    """Abstract base class for monitoring systems."""
    
    @abstractmethod
    def start_monitoring(self) -> None:
        """Start system monitoring."""
        pass
    
    @abstractmethod
    def stop_monitoring(self) -> None:
        """Stop system monitoring."""
        pass
    
    @abstractmethod
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        pass
    
    @abstractmethod
    def check_resource_limits(self) -> bool:
        """Check if system resources are within limits."""
        pass
    
    @abstractmethod
    def log_metrics(self, metrics: Dict[str, Any]) -> None:
        """Log performance metrics."""
        pass


class ProgressTracker(ABC):
    """Abstract base class for progress tracking."""
    
    @abstractmethod
    def start_tracking(self, total_items: int) -> None:
        """Start tracking progress for a given number of items."""
        pass
    
    @abstractmethod
    def update_progress(self, 
                       current_item: int, 
                       status: str, 
                       metadata: Optional[Dict[str, Any]] = None) -> None:
        """Update progress with current item and status."""
        pass
    
    @abstractmethod
    def finish_tracking(self) -> None:
        """Finish progress tracking."""
        pass


class ErrorHandler(ABC):
    """Abstract base class for error handling."""
    
    @abstractmethod
    def handle_error(self, 
                     error: Exception, 
                     context: Dict[str, Any]) -> bool:
        """Handle an error and return whether to continue processing."""
        pass
    
    @abstractmethod
    def should_retry(self, 
                     error: Exception, 
                     attempt: int) -> bool:
        """Determine if an operation should be retried."""
        pass
    
    @abstractmethod
    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics."""
        pass


class ValidationSystem(ABC):
    """Abstract base class for validation systems."""
    
    @abstractmethod
    def validate_input(self, data: Dict[str, Any]) -> bool:
        """Validate input data."""
        pass
    
    @abstractmethod
    def validate_output(self, result: Any) -> bool:
        """Validate output result."""
        pass
    
    @abstractmethod
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate configuration."""
        pass
    
    @abstractmethod
    def get_validation_errors(self) -> List[str]:
        """Get list of validation errors."""
        pass


class CacheManager(ABC):
    """Abstract base class for cache management."""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache with optional TTL."""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all cache entries."""
        pass
    
    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        pass


class PluginManager(ABC):
    """Abstract base class for plugin management."""
    
    @abstractmethod
    def load_plugin(self, plugin_name: str, plugin_config: Dict[str, Any]) -> Any:
        """Load a plugin with configuration."""
        pass
    
    @abstractmethod
    def unload_plugin(self, plugin_name: str) -> None:
        """Unload a plugin."""
        pass
    
    @abstractmethod
    def get_plugin(self, plugin_name: str) -> Optional[Any]:
        """Get a loaded plugin."""
        pass
    
    @abstractmethod
    def list_plugins(self) -> List[str]:
        """List all available plugins."""
        pass
    
    @abstractmethod
    def list_loaded_plugins(self) -> List[str]:
        """List all loaded plugins."""
        pass


class InferenceOrchestrator(ABC):
    """Abstract base class for inference orchestration."""
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the orchestrator with configuration."""
        pass
    
    @abstractmethod
    def run_inference(self, 
                     data_ids: List[str], 
                     resume: bool = True) -> Dict[str, Any]:
        """Run inference on the specified data IDs."""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Clean up resources."""
        pass
    
    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """Get current orchestrator status."""
        pass


# Factory pattern for creating components
class ComponentFactory:
    """Factory for creating system components."""
    
    _data_loaders: Dict[str, type] = {}
    _model_managers: Dict[str, type] = {}
    _prompt_managers: Dict[str, type] = {}
    _output_managers: Dict[str, type] = {}
    _inference_engines: Dict[str, type] = {}
    
    # Model registry mapping model names to model managers
    _model_registry: Dict[str, str] = {}  # model_name -> manager_name
    _model_patterns: Dict[str, List[str]] = {}  # manager_name -> list of model name patterns
    
    @classmethod
    def register_data_loader(cls, name: str, loader_class: type) -> None:
        """Register a data loader class."""
        cls._data_loaders[name] = loader_class
    
    @classmethod
    def register_model_manager(cls, name: str, manager_class: type) -> None:
        """Register a model manager class."""
        cls._model_managers[name] = manager_class
    
    @classmethod
    def register_model_manager_for_patterns(cls, name: str, manager_class: type, patterns: List[str]) -> None:
        """Register a model manager class with supported model name patterns."""
        cls._model_managers[name] = manager_class
        cls._model_patterns[name] = patterns
        
        # Also register exact matches in the registry
        for pattern in patterns:
            cls._model_registry[pattern] = name
    
    @classmethod
    def register_model_for_manager(cls, model_name: str, manager_name: str) -> None:
        """Register a specific model name to use a specific manager."""
        if manager_name not in cls._model_managers:
            raise ValueError(f"Unknown model manager: {manager_name}")
        cls._model_registry[model_name] = manager_name
    
    @classmethod
    def get_model_manager_for_model(cls, model_name: str) -> str:
        """Get the appropriate model manager name for a given model name."""
        # First check exact matches
        if model_name in cls._model_registry:
            return cls._model_registry[model_name]
        
        # Then check patterns
        for manager_name, patterns in cls._model_patterns.items():
            for pattern in patterns:
                if model_name.startswith(pattern):
                    return manager_name
        
        # If no specific manager found, raise an error
        raise ValueError(f"No model manager found for model: {model_name}. "
                        f"Supported models: {list(cls._model_registry.keys())} "
                        f"and patterns: {cls._model_patterns}")
    
    @classmethod
    def validate_model_support(cls, model_name: str) -> bool:
        """Validate that a model name is supported by the registered managers."""
        try:
            cls.get_model_manager_for_model(model_name)
            return True
        except ValueError:
            return False
    
    @classmethod
    def list_supported_models(cls) -> Dict[str, List[str]]:
        """List all supported models grouped by manager."""
        supported = {}
        for manager_name, patterns in cls._model_patterns.items():
            supported[manager_name] = patterns
        
        # Add exact matches
        for model_name, manager_name in cls._model_registry.items():
            if manager_name not in supported:
                supported[manager_name] = []
            if model_name not in supported[manager_name]:
                supported[manager_name].append(model_name)
        
        return supported
    
    @classmethod
    def register_prompt_manager(cls, name: str, manager_class: type) -> None:
        """Register a prompt manager class."""
        cls._prompt_managers[name] = manager_class
    
    @classmethod
    def register_output_manager(cls, name: str, manager_class: type) -> None:
        """Register an output manager class."""
        cls._output_managers[name] = manager_class
    
    @classmethod
    def register_inference_engine(cls, name: str, engine_class: type) -> None:
        """Register an inference engine class."""
        cls._inference_engines[name] = engine_class
    
    @classmethod
    def create_data_loader(cls, name: str, config: DataConfig) -> DataLoader:
        """Create a data loader instance."""
        if name not in cls._data_loaders:
            raise ValueError(f"Unknown data loader: {name}")
        return cls._data_loaders[name](config)
    
    @classmethod
    def create_model_manager(cls, name: str, config: ModelConfig) -> ModelManager:
        """Create a model manager instance."""
        if name not in cls._model_managers:
            raise ValueError(f"Unknown model manager: {name}")
        return cls._model_managers[name](config)
    
    @classmethod
    def create_model_manager_for_model(cls, config: ModelConfig) -> ModelManager:
        """Create a model manager instance based on the model name in config."""
        model_name = config.model_name
        
        # Validate that the model is supported
        if not cls.validate_model_support(model_name):
            supported_models = cls.list_supported_models()
            raise ValueError(f"Model '{model_name}' is not supported. "
                           f"Supported models: {supported_models}")
        
        # Get the appropriate manager name
        manager_name = cls.get_model_manager_for_model(model_name)
        
        # Create and return the manager
        logger.info(f"Creating model manager '{manager_name}' for model '{model_name}'")
        return cls._model_managers[manager_name](config)
    
    @classmethod
    def create_prompt_manager(cls, name: str, config: InferenceConfig) -> PromptManager:
        """Create a prompt manager instance."""
        if name not in cls._prompt_managers:
            raise ValueError(f"Unknown prompt manager: {name}")
        return cls._prompt_managers[name](config)
    
    @classmethod
    def create_output_manager(cls, name: str, config: OutputConfig) -> OutputManager:
        """Create an output manager instance."""
        if name not in cls._output_managers:
            raise ValueError(f"Unknown output manager: {name}")
        return cls._output_managers[name](config)
    
    @classmethod
    def create_inference_engine(cls, name: str, config: InferenceConfig, model_manager: ModelManager, prompt_manager: PromptManager) -> InferenceEngine:
        """Create an inference engine instance."""
        if name not in cls._inference_engines:
            raise ValueError(f"Unknown inference engine: {name}")
        return cls._inference_engines[name](config, model_manager, prompt_manager)