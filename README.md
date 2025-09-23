# LLM Batch Inference System

This repository provides a flexible system for running inference with Large Language Models (LLMs) from Hugging Face. It is designed to be easily configurable and customizable for various inference tasks using medical reports and various prompt styles.

## Table of Contents
- [Setup](#setup)
- [Configuration](#configuration)
- [Running Inference](#running-inference)
- [Command-Line Arguments](#command-line-arguments)
- [Downloading Models](#downloading-models)
- [Customization](#customization)
  - [Adding a New Model Manager](#adding-a-new-model-manager)
  - [Adding a New Data Reader](#adding-a-new-data-reader)
  - [Adding a New Prompt Manager](#adding-a-new-prompt-manager)
  - [Adding a New Output Manager](#adding-a-new-output-manager)
  - [Adding a New Inference Engine](#adding-a-new-inference-engine)
- [Logging](#logging)
- [Troubleshooting](#troubleshooting)

## Setup

1.  **Clone the repository**:
    ```bash
    git clone <repository_url>
    cd LLM_Inference
    ```

2.  **Create a virtual environment** (recommended):
    ```bash
    conda create --name llm_env python
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Hugging Face Token**: Set your Hugging Face authentication token as an environment variable. This is required for downloading models and accessing gated models.
    ```bash
    export HF_TOKEN="your_huggingface_token"
    ```
    You can obtain your token from [Hugging Face Settings](https://huggingface.co/settings/tokens).

## Configuration

The system's behavior is controlled by the `configs/your_config_file.yaml` file. Take a look at `configs/BaseConfig.yaml` to learn how to customize a config. Please do not change the BaseConfig as it will be a guide template. 

This file defines settings for data loading, model, inference, output, logging, and system parameters.

**Example `config/config.yaml` structure (key sections)**:

```yaml
# config/config.yaml
data:
  source_type: "pickle"  # Options: "pickle", "json", "csv", "single_report"
  data_folder: "data" # Path relative to project root
  input_files:
    - name: "radiology_reports"
      path: "mrnacc_ultrasound_generate_radreport.pkl"
      required: true
  process_all: false
  custom_ids: ['mrn0-acc0'] # List of specific IDs to process (overrides process_all)
  resume_from_existing: true # If true, continues from previous run's results
model:
  cache_dir: "./models" # Path relative to project root
  model_name: "unsloth/Llama-3.3-70B-Instruct-bnb-4bit"
  quantization:
    enabled: false
    bits: null
  cuda_devices: "0,1"
  max_memory: # Optional: omit this section or set to null if not needed
    "0": "32GB"
    "1": "32GB"
inference:
  prompt_template: "prompt_chat_template_Qwen1" # Name of the prompt template function
  prompt_manager: "RadPath" # Or "RadPath", "SingleReport"
  generation:
    do_sample: false
    max_tokens: 32768
  batch_size: 1
output:
  results_dir: "llm_results" # Path relative to project root
  default_filename: "inference_results.json"
  format: "json"
  add_timestamp: false # If true, adds timestamp to filename (e.g., results_20230716-103000.json)
  backup_existing: true # If true, backs up existing file before overwriting
logging:
  level: "INFO"  # General console log level (e.g., INFO, DEBUG, WARNING, ERROR)
  console: true # Enable/disable console output
  progress_bar: true # Enable/disable TQDM progress bar
  gpu_monitoring: true # Enable/disable GPU memory monitoring (requires gpustat)
  file:
    enabled: true # Set to true to write logs to a file
    path: "logs/application.log" # Path to the log file (relative to project root)
    level: "DEBUG" # Specific log level for the file (e.g., DEBUG to capture everything)
system:
  num_workers: 1
  continue_on_error: true
  max_retries: 3
```

You can specify a different configuration file using the `--config` argument:

```bash
python -m core.run_inference --config config/my_custom_config.yaml
```

## Running Inference

To run the inference pipeline, execute the `run_inference.py` module:

```bash
python -m core.run_inference
```

### Dry Run
You can perform a dry run to see the effective configuration without executing inference:

```bash
python -m core.run_inference --dry-run
```

### Command-Line Overrides
Many configuration parameters can be overridden directly via command-line arguments. For example:

```bash
python -m core.run_inference \
    --model-name "another/model-name" \
    --output-file "my_results.json" \
    --log-level DEBUG \
```

For a full list of available arguments, use the `--help` flag:

```bash
python -m core.run_inference --help
```

## Command-Line Arguments

The `core/run_inference.py` script supports several command-line arguments to override configuration values or perform specific actions.

```
usage: run_inference.py [-h] [-c CONFIG] [--data-folder DATA_FOLDER]
                        [--process-all] [--custom-ids CUSTOM_IDS [CUSTOM_IDS ...]]
                        [--id-file ID_FILE] [--model-name MODEL_NAME]
                        [--cache-dir CACHE_DIR] [--cuda-devices CUDA_DEVICES]
                        [--quantization {4,8}] [--batch-size BATCH_SIZE]
                        [--prompt-template PROMPT_TEMPLATE]
                        [--max-tokens MAX_TOKENS] [--output-file OUTPUT_FILE]
                        [--results-dir RESULTS_DIR] [--format {json,yaml,csv}]
                        [--log-level {DEBUG,INFO,WARNING,ERROR}] [--no-resume]
                        [--dry-run] [--list-models]

LLM Batch Inference System

options:
  -h, --help            show this help message and exit
  -c CONFIG, --config CONFIG
                        Path to configuration file (default: config/config.yaml)

Data arguments:
  --data-folder DATA_FOLDER
                        Path to data folder (default: data)
  --process-all         Process all available IDs (default: false)
  --custom-ids CUSTOM_IDS [CUSTOM_IDS ...]
                        List of specific IDs to process (default: None)
  --id-file ID_FILE     Path to file containing IDs to process (one per line)
                        (default: None)

Model arguments:
  --model-name MODEL_NAME
                        Name of the model to use (default: None)
  --cache-dir CACHE_DIR
                        Model cache directory (default: None)
  --cuda-devices CUDA_DEVICES
                        CUDA devices to use (e.g., "0,1") (default: None)
  --quantization {4,8}
                        Quantization bits (4 or 8) (default: None)

Inference arguments:
  --batch-size BATCH_SIZE
                        Batch size for inference (default: None)
  --prompt-template PROMPT_TEMPLATE
                        Prompt template to use (default: None)
  --max-tokens MAX_TOKENS
                        Maximum tokens to generate (default: None)

Output arguments:
  --output-file OUTPUT_FILE
                        Output file name (default: None)
  --results-dir RESULTS_DIR
                        Results directory (default: None)
  --format {json,yaml,csv}
                        Output format (default: None)

System arguments:
  --log-level {DEBUG,INFO,WARNING,ERROR}
                        Logging level (default: None)
  --no-resume           Do not resume from existing results (default: false)
  --dry-run             Show configuration and exit without running inference
                        (default: false)
  --list-models         List all supported models and exit (default: false)
```

## Downloading Models

Use the `download_models_from_HF.py` script in the root directory to download models from Hugging Face and save them locally. This is useful for offline inference or managing local model caches. You will need your HuggingFace Token in your environment with variable named 'HF_TOKEN' and can check by command below.

```bash
echo $HF_TOKEN
```

Below is

```bash
python download_models_from_HF.py \
    --model_name "meta-llama/Llama-3-8B-Instruct" \
    --save_directory "./models/Llama-3-8B-Instruct" \
    --cuda_devices "0"
```

**Arguments**:
- `--model_name` (required): The exact name of the model on Hugging Face (e.g., `meta-llama/Llama-3.3-70B-Instruct`).
- `--save_directory` (required): The local path where the model and tokenizer will be saved.
- `--cuda_devices` (optional): Comma-separated list of CUDA devices to use (e.g., `"0,1"`). If not provided, the download will not be pinned to specific devices.

## Customization

The system is designed with a modular architecture, allowing you to easily extend its functionality by adding new implementations for various components. Each component type has a base interface defined in `core/interfaces.py` and concrete implementations in the `implementations/` directory.

To add a new component, you generally need to:
1.  Create a new Python file in the appropriate `implementations/` subdirectory (e.g., `implementations/model_managers.py` for a new model manager).
2.  Implement a class that adheres to the corresponding interface defined in `core/interfaces.py`.
3.  Register your new implementation with the `ComponentFactory`.

### Adding a New Model Manager

If you want to integrate a new type of LLM (e.g., a custom model, or one requiring specific loading logic not covered by existing managers), you can create a new model manager.

1.  **Create a new file**: E.g., `implementations/my_model_manager.py`.
2.  **Implement the interface**: Your class should inherit from `ModelManager` (from `core.interfaces`).
    ```python
    # implementations/my_model_manager.py
    from core.interfaces import ModelManager
    from core.config import ModelConfig, GenerationConfig
    # ... other necessary imports

    class MyCustomModelManager(ModelManager):
        def __init__(self, config: ModelConfig):
            super().__init__(config)
            # Your initialization logic

        def load_model(self):
            # Load your custom model and tokenizer
            pass

        def generate(self, messages, generation_config: GenerationConfig):
            # Implement generation logic for your model
            pass

        def cleanup(self):
            # Clean up resources if necessary
            pass

        def get_model_info(self):
            # Return information about the model
            pass
    ```
3.  **Register with Factory**: In `implementations/my_model_manager.py` (or `implementations/__init__.py` if you have one):
    ```python
    from core.interfaces import ComponentFactory
    # ...

    ComponentFactory.register_model_manager("my_custom_model", MyCustomModelManager)

    # If your model manager handles a family of models based on name patterns:
    # ComponentFactory.register_model_manager_for_patterns(
    #     "my_custom_model", MyCustomModelManager, ["my-org/my-model-", "another-pattern/"]
    # )
    ```
4.  **Update `config.yaml`**: Set `model.model_manager_type` (or `model.model_name` if using patterns) to match your registration key.

### Adding a New Data Reader

If you have data in a format not currently supported (e.g., a custom database, a new file format), you can implement a new data reader.

1.  **Create a new file**: E.g., `implementations/my_data_reader.py`.
2.  **Implement the interface**: Your class should inherit from `DataReader` (from `core.interfaces`).
    ```python
    class MyCustomDataReader(DataReader):
        def __init__(self, config: DataConfig):
            self.config = config
            # Your initialization logic, e.g., establish connection
            
        def load_data(self) -> Dataset:
            # Load data from your custom source and return a Dataset object
            # This method should return a PyTorch Dataset, not raw data
            pass

        def get_all_ids(self) -> List[str]:
            # Return a list of all unique IDs available in the data source
            pass
    ```
3.  **Register with Factory**: In `implementations/my_data_reader.py` (or `implementations/__init__.py` if you have one):
    ```python
    from core.interfaces import ComponentFactory
    # ...
    ComponentFactory.register_data_reader("my_custom_data_source", MyCustomDataReader)
    ```
4.  **Update `config.yaml`**: Set `data.source_type` to `"my_custom_data_source"`.

### Adding a New Prompt Manager

If you need a new way to construct prompts for your LLMs, you can create a custom prompt manager.

1.  **Create a new file**: E.g., `implementations/my_prompt_manager.py`.
2.  **Implement the interface**: Your class should inherit from `PromptManager` (from `core.interfaces`).
    ```python
    # implementations/my_prompt_manager.py

    class MyCustomPromptManager(PromptManager):
        def __init__(self, config: InferenceConfig):
            self.config = config
            # Initialize prompt templates

        def get_prompt_template(self, template_name: str) -> Callable:
            # Return a callable prompt template function
            pass
        
        def create_messages(self, template_name: str, data: Dict[str, Any]) -> List[Dict[str, str]]:
            # Create messages for the model using the specified template and data
            pass

        def list_available_templates(self) -> List[str]:
            # List all available prompt templates
            pass

        def validate_template(self, template_name: str) -> bool:
            # Validate that a template exists
            pass
    ```
3.  **Register with Factory**:
    ```python
    from core.interfaces import ComponentFactory
    # ...
    ComponentFactory.register_prompt_manager("my_custom_prompt", MyCustomPromptManager)
    ```
4.  **Update `config.yaml`**: Set `inference.prompt_manager` to `"my_custom_prompt"`.

### Adding a New Output Manager

To handle output in a different format or save it to a custom destination (e.g., a database), you can create a new output manager.

1.  **Create a new file**: E.g., `implementations/my_output_manager.py`.
2.  **Implement the interface**: Your class should inherit from `OutputManager` (from `core.interfaces`).
    ```python
    # implementations/my_output_manager.py


    class MyCustomOutputManager(OutputManager):
        def __init__(self, config: OutputConfig):
            self.config = config
            # Your initialization logic

        def get_output_path(self, filename: Optional[str] = None) -> Path:
            # Construct the full output path
            pass

        def save_results(self, results: Dict[str, Any], filename: Optional[str] = None) -> str:
            # Save results to your custom format/destination
            pass

        def load_existing_results(self, filename: Optional[str] = None) -> Dict[str, Any]:
            # Load existing results for resume functionality
            pass
            
        def backup_existing_file(self, file_path: Path) -> str:
            # Backup an existing file
            pass

        def get_processed_ids(self, results: Dict[str, Any]) -> Set[str]:
            # Get set of already processed IDs
            pass
    ```
3.  **Register with Factory**:
    ```python
    from core.interfaces import ComponentFactory
    # ...
    ComponentFactory.register_output_manager("my_custom_format", MyCustomOutputManager)
    ```
4.  **Update `config.yaml`**: Set `output.format` to `"my_custom_format"`.

### Adding a New Inference Engine

If you need to customize the core inference loop (e.g., for specialized batching, distributed inference, or unique error handling within the loop), you can create a custom inference engine.

1.  **Create a new file**: E.g., `implementations/my_inference_engine.py`.
2.  **Implement the interface**: Your class should inherit from `InferenceEngine` (from `core.interfaces`).
    ```python
    # implementations/my_inference_engine.py


    class MyCustomInferenceEngine(InferenceEngine):
        def __init__(self, config: InferenceConfig, model_manager: ModelManager, prompt_manager: PromptManager):
            self.config = config
            self.model_manager = model_manager
            self.prompt_manager = prompt_manager
            # Initialize engine-specific state, e.g., performance stats

        def setup(self):
            # Perform any setup before inference starts
            pass

        def process_dataset(self, dataset: Any) -> Dict[str, Any]:
            # Implement the core logic for processing the entire dataset
            # This is where you'd handle batching, calling model, etc.
            pass

        def get_stats(self) -> Dict[str, Any]:
            # Return performance statistics
            pass
    ```
3.  **Register with Factory**:
    ```python
    from core.interfaces import ComponentFactory
    # ...
    ComponentFactory.register_inference_engine("my_custom_engine", MyCustomInferenceEngine)
    ```
4.  **Update `config.yaml`**: (Currently, `inference_engine` is hardcoded to `"default"` in `core/run_inference.py` in `ComponentFactory.create_inference_engine`, but you would modify that line to use your custom engine name if needed.)

## Logging

The system provides flexible logging capabilities, configurable via the `logging` section in your `config.yaml`. You can control log levels for the console and direct logs to a file.

**Key features:**
-   **Console Logging**: Messages are displayed in the terminal. Configured by `logging.level` and `logging.console`.
-   **File Logging**: All log messages can be written to a specified file, with its own configurable log level. This is particularly useful for debugging (`DEBUG` level) without cluttering the console.
-   **GPU Monitoring**: Toggle GPU resource logging.

## Troubleshooting

-   **`FileNotFoundError: config.yaml`**: Ensure your `config.yaml` is located in the `config/` directory, or specify its correct path using the `--config` argument.
-   **CUDA Errors / Device Not Recognized**: Verify your `cuda_devices` and `max_memory` settings in `config/config.yaml`. Ensure your `HF_TOKEN` environment variable is set correctly.
-   **Model Loading Issues**: Check if `model_name` is correct and if you have sufficient memory (adjust `max_memory` in `config.yaml` if needed). Ensure your Hugging Face token is correctly set for gated models.
-   **`TypeError: transformers.generation.utils.GenerationMixin.generate()`**: This usually indicates an issue with how inputs are passed to the model's generate method. Ensure the inputs are correctly prepared (e.g., as a `torch.Tensor` or `dict` as expected by the model). (This was addressed in recent updates.) 