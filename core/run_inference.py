import sys
import os
import argparse
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
import time

# Add the current directory to the path so we can import our modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from core.config import ConfigManager, create_argument_parser, setup_logging
from core.interfaces import ComponentFactory
from implementations.datareaders import *  # This registers the data loaders
from implementations.model_managers import *
from implementations.preprocessors import *
from implementations.output_managers import *
from implementations.inference_engines import *

logger = logging.getLogger(__name__)

class LLMInferenceOrchestrator:
    """
    Main orchestrator for LLM inference operations.
    Coordinates all components and manages the inference workflow.
    """
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.data_reader = None
        self.model_manager = None
        self.preprocessor = None
        self.output_manager = None
        self.inference_engine = None
        
        # Configuration objects
        self.data_config = config_manager.get_data_config()
        self.model_config = config_manager.get_model_config()
        self.inference_config = config_manager.get_inference_config()
        self.output_config = config_manager.get_output_config()
        self.system_config = config_manager.get_system_config()
        
        logger.debug("LLM Inference Orchestrator initialized")
    
    def validate_model_configuration(self) -> None:
        """Validate that the model configuration is supported."""
        model_name = self.model_config.model_name
        
        # Check if the model is supported
        if not ComponentFactory.validate_model_support(model_name):
            supported_models = ComponentFactory.list_supported_models()
            
            logger.error(f"Model '{model_name}' is not supported.")
            logger.error(f"Supported models by manager:")
            for manager_name, patterns in supported_models.items():
                logger.error(f"  {manager_name}: {patterns}")
            
            raise ValueError(f"Model '{model_name}' is not supported. "
                           f"Supported models: {supported_models}")
        
        # Log which manager will be used
        manager_name = ComponentFactory.get_model_manager_for_model(model_name)
        logger.debug(f"Model '{model_name}' will use manager: {manager_name}")
    
    def setup_components(self):
        """Initialize and set up all components."""
        try:
            # Validate model configuration first
            self.validate_model_configuration()
            # Initialize data reader
            self.data_reader = ComponentFactory.create_data_reader(self.data_config)
            logger.debug(f"Data reader initialized: {self.data_config.source_type}")
            
            # Initialize model manager based on model name
            self.model_manager = ComponentFactory.create_model_manager_for_model(
                self.model_config
            )
            logger.debug(f"Model manager initialized for model: {self.model_config.model_name}")
            
            # Initialize preprocessor
            self.preprocessor = ComponentFactory.create_preprocessor(
                self.inference_config
            )
            logger.debug(f"Preprocessor initialized: {self.preprocessor.__class__.__name__}")
            
            # Initialize output manager
            self.output_manager = ComponentFactory.create_output_manager(
                self.output_config.format,
                self.output_config
            )
            logger.debug(f"Output manager initialized: {self.output_manager.__class__.__name__}")
            
            # Initialize inference engine
            self.inference_engine = ComponentFactory.create_inference_engine(
                "default",
                self.inference_config,
                self.model_manager,
                self.preprocessor
            )
            logger.debug(f"Inference engine initialized: {self.inference_engine.__class__.__name__}")
            
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise
    
    def load_and_validate_data(self) -> Dict[str, Any]:
        """Load and validate data from configured sources."""
        try:
            # Load data
            data = self.data_reader.load_data(self.data_config)
            
            # Validate data
            if not self.data_reader.validate_data(data):
                raise ValueError("Data validation failed")
            
            logger.info("Data loaded and validated successfully")
            return data
            
        except Exception as e:
            logger.error(f"Failed to load/validate data: {e}")
            raise
    
    def get_mrn_list(self) -> List[str]:
        """Get the list of MRNs to process."""
        try:
            mrns = self.data_reader.get_mrn_list()
            
            if not mrns:
                raise ValueError("No MRNs found based on configuration")
            
            logger.info(f"Found {len(mrns)} MRNs to process")
            return mrns
            
        except Exception as e:
            logger.error(f"Failed to get MRN list: {e}")
            raise
    
    def create_dataset(self, data: Dict[str, Any], mrns: List[str]) -> Any:
        """Create a dataset from the loaded data."""
        try:
            dataset = self.data_reader.create_dataset(data, mrns)
            logger.info(f"Dataset created with {len(dataset)} entries")
            return dataset
            
        except Exception as e:
            logger.error(f"Failed to create dataset: {e}")
            raise
    
    def run_inference(self):
        """Run the full inference pipeline."""
        try:
            # Step 1: Loading and creating dataset
            logger.info("1. Loading and creating dataset...")
            dataset, all_ids = self.data_reader.load_data()
            
            # Step 2: Handle resume logic
            unprocessed_ids = self._get_unprocessed_ids(all_ids)
            
            if not unprocessed_ids:
                logger.info("No new IDs to process. Exiting.")
                return

            # Step 3: Create a new dataset with only the unprocessed IDs
            if hasattr(dataset, 'filter_ids'):
                unprocessed_dataset = dataset.filter_ids(unprocessed_ids)
            else:
                logger.warning("Dataset does not have a 'filter_ids' method. Processing all loaded data as is.")
                unprocessed_dataset = dataset
            
            # Step 4: Run inference
            logger.info("2. Running inference...")
            # If output manager supports incremental append (JSONL), provide a hook
            if hasattr(self.output_manager, 'append_result'):
                def _append_result(_id, text):
                    try:
                        self.output_manager.append_result(_id, text, self.output_config.default_filename)
                    except Exception as e:
                        logger.warning(f"Failed to append result for {_id}: {e}")
                setattr(self.inference_engine, '_append_result', _append_result)

            new_results = self.inference_engine.process_dataset(unprocessed_dataset)
            
            # Step 5: Save results
            logger.info("3. Saving results...")

            # If resuming, load existing results and merge them
            if self.data_config.resume_from_existing:
                existing_results = self.output_manager.load_existing_results(
                    self.output_config.default_filename
                )
                existing_results.update(new_results)
                results_to_save = existing_results
            else:
                results_to_save = new_results
            
            output_path = self.output_manager.save_results(
                results_to_save, 
                self.output_config.default_filename
            )
            logger.info(f"Results saved to: {output_path}")

            logger.info("Inference pipeline completed successfully.")
            print("\nInference completed successfully!")
            print(f"   Processed {len(new_results)} IDs")
            print(f"   Results saved to output directory\n")

        except Exception as e:
            logger.error(f"Inference pipeline failed: {e}", exc_info=True)
            print(f"\nFatal error: {e}")

            
    def _get_unprocessed_ids(self, all_ids: List[str]) -> List[str]:
        """Get the list of IDs that have not yet been processed."""
        if not self.data_config.resume_from_existing:
            logger.info("Resume is disabled by config. Processing all specified IDs.")
            return all_ids

        logger.info("Resume is enabled. Checking for existing results...")
        existing_results = self.output_manager.load_existing_results(
            self.output_config.default_filename
        )

        if not existing_results:
            logger.info(f"No existing results found. Processing selected IDs: {len(all_ids)}.")
            return all_ids
        
        processed_ids = set(existing_results.keys())
        unprocessed_ids = [id for id in all_ids if id not in processed_ids]
        
        logger.info(f"Found {len(unprocessed_ids)} new IDs to process.")
        return unprocessed_ids
    
    def cleanup(self) -> None:
        """Clean up resources."""
        try:
            if self.model_manager:
                self.model_manager.cleanup()
            logger.info("Cleanup completed")
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")


def show_configuration(config_manager: ConfigManager) -> None:
    """Display the current configuration."""
    print("\n" + "="*60)
    print("REFACTORED LLM INFERENCE SYSTEM - CONFIGURATION")
    print("="*60)
    
    config = config_manager.get_full_config()
    
    print("\nDATA CONFIGURATION:")
    data_config = config['data']
    print(f"  Source Type: {data_config['source_type']}")
    print(f"  Data Folder: {data_config['data_folder']}")
    print(f"  Input Files: {len(data_config['input_files'])} files")
    print(f"  Process All: {data_config['process_all']}")
    if data_config['custom_ids']:
        print(f"  Custom IDs: {len(data_config['custom_ids'])} IDs")
    if data_config['id_file']:
        print(f"  ID File: {data_config['id_file']}")
    print(f"  Resume: {data_config['resume_from_existing']}")
    
    print("\nMODEL CONFIGURATION:")
    model_config = config['model']
    model_name = model_config['model_name']
    print(f"  Model Name: {model_name}")
    print(f"  Cache Dir: {model_config['cache_dir']}")
    print(f"  CUDA Devices: {model_config['cuda_devices']}")
    # Quantization removed
    
    # Show which model manager will be used
    try:
        manager_name = ComponentFactory.get_model_manager_for_model(model_name)
        print(f"  Model Manager: {manager_name}")
    except ValueError as e:
        print(f"  Model Manager: ERROR - {e}")
        print("  Available model managers:")
        supported_models = ComponentFactory.list_supported_models()
        for manager_name, patterns in supported_models.items():
            print(f"    {manager_name}: {patterns}")
    
    # Show if model is supported
    is_supported = ComponentFactory.validate_model_support(model_name)
    print(f"  Model Supported: {'Yes' if is_supported else 'No'}")
    
    print("\nINFERENCE CONFIGURATION:")
    inference_config = config['inference']
    # Preprocessor settings
    print(f"  Preprocessor: {inference_config.get('preprocessor')}")
    print(f"  Prompt File: {inference_config.get('prompt_file')}")
    print(f"  Batch Size: {inference_config['batch_size']}")
    print(f"  Max New Tokens: {inference_config['generation']['max_new_tokens']}")
    
    print("\nOUTPUT CONFIGURATION:")
    output_config = config['output']
    print(f"  Results Dir: {output_config['results_dir']}")
    print(f"  Format: {output_config['format']}")
    print(f"  Default File: {output_config['default_filename']}")
    
    print("\nSYSTEM CONFIGURATION:")
    system_config = config['system']
    print(f"  Continue on Error: {system_config['continue_on_error']}")
    print(f"  Max Retries: {system_config['max_retries']}")
    print(f"  Memory Threshold: {system_config['memory_threshold']}")
    
    print("\n" + "="*60)


def list_supported_models() -> None:
    """List all supported models grouped by manager."""
    print("\n" + "="*60)
    print("SUPPORTED MODELS BY MANAGER")
    print("="*60)
    
    supported_models = ComponentFactory.list_supported_models()
    
    if not supported_models:
        print("No model managers registered.")
        return
    
    for manager_name, patterns in supported_models.items():
        print(f"\n{manager_name.upper()} MANAGER:")
        if patterns:
            print("  Supported patterns:")
            for pattern in patterns:
                print(f"    - {pattern}")
        else:
            print("  No patterns registered")
    
    print("\nTo use a model, specify its name in the configuration file or use --model-name.")
    print("Example: --model-name unsloth/Llama-3.3-70B-Instruct-bnb-4bit")
    print("\n" + "="*60)


def main():
    """Main entry point for the refactored inference system."""
    try:
        # Parse command-line arguments
        parser = create_argument_parser()
        args = parser.parse_args()
        
        # Load configuration
        config_manager = ConfigManager(args.config)
        
        # Update configuration with command-line arguments
        config_manager.update_from_args(args)
        
        # Setup logging
        logging_config = config_manager.get_logging_config()
        setup_logging(logging_config)
        
        logger.info("Starting refactored LLM inference system")
        
        # List supported models if requested
        if args.list_models:
            list_supported_models()
            return
        
        # Show configuration if requested
        if args.dry_run:
            show_configuration(config_manager)
            print("\nDRY RUN MODE - Configuration shown above")
            print("To run actual inference, remove the --dry-run flag")
            return
        
        # Initialize orchestrator
        orchestrator = LLMInferenceOrchestrator(config_manager)
        
        # Initialize components
        orchestrator.setup_components()
        
        orchestrator.run_inference()
        
    except KeyboardInterrupt:
        logger.info("Inference interrupted by user")
        print("\nInference interrupted by user")
        try:
            if 'orchestrator' in locals() and hasattr(orchestrator, 'output_manager'):
                existing = orchestrator.output_manager.load_existing_results(
                    orchestrator.output_config.default_filename
                )
                processed_ids = list(existing.keys())
                # Derive output dir from manager path helper
                out_path = orchestrator.output_manager.get_output_path(orchestrator.output_config.default_filename)
                out_dir = Path(out_path).parent
                ids_path = out_dir / "processed_ids.txt"
                with open(ids_path, "w", encoding="utf-8") as f:
                    for pid in processed_ids:
                        f.write(f"\n".join(processed_ids))
                        break
                logger.info(f"Wrote processed IDs to {ids_path}")
        except Exception:
            pass
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"\nFatal error: {e}")
        sys.exit(1)
        
    finally:
        # Cleanup
        try:
            if 'orchestrator' in locals():
                orchestrator.cleanup()
        except:
            pass

if __name__ == "__main__":
    main()