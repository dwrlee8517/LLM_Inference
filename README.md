# LLM Batch Inference System

This repository provides a flexible and extensible system for running batch inference with Large Language Models (LLMs) from Hugging Face. It is designed to be easily configurable and customizable for various inference tasks.

## Table of Contents
- [Features](#features)
- [Setup](#setup)
- [Configuration](#configuration)
- [Running Inference](#running-inference)
- [Downloading Models](#downloading-models)
- [Customization](#customization)
  - [Adding a New Model Manager](#adding-a-new-model-manager)
  - [Adding a New Data Loader](#adding-a-new-data-loader)
  - [Adding a New Prompt Manager](#adding-a-new-prompt-manager)
  - [Adding a New Output Manager](#adding-a-new-output-manager)
  - [Adding a New Inference Engine](#adding-a-new-inference-engine)
- [Command-Line Arguments](#command-line-arguments)
- [Logging](#logging)
- [Troubleshooting](#troubleshooting)

## Features
- **Configurable**: All system behaviors are controlled via a YAML configuration file.
- **Extensible**: Easily add new model types, data sources, prompt templates, and output formats.
- **Modular Design**: Clear separation of concerns with well-defined interfaces.
- **Resume Functionality**: Continue inference from previous runs.
- **Command-Line Overrides**: Override configuration settings directly from the command line.
- **GPU Support**: Configurable CUDA device usage and memory management.

## Setup

1.  **Clone the repository**:
    ```bash
    git clone <repository_url>
    cd LLM_Inference
    ```

2.  **Create a virtual environment** (recommended):
    ```bash
    python -m venv llm_env
    source llm_env/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    _Note: Ensure `requirements.txt` includes `torch`, `transformers`, `accelerate`, `pyyaml`, and other necessary libraries._

4.  **Hugging Face Token**: Set your Hugging Face authentication token as an environment variable. This is required for downloading models and accessing gated models.
    ```bash
    export HF_TOKEN="your_huggingface_token"
    ```
    You can obtain your token from [Hugging Face Settings](https://huggingface.co/settings/tokens).

## Configuration

The system's behavior is controlled by the `config/config.yaml` file. This file defines settings for data loading, model, inference, output, logging, and system parameters.

**Example `config/config.yaml` structure (key sections)**:

```yaml
# config/config.yaml
data:
  source_type: "pickle"
  data_folder: "data"
  input_files:
    - name: "radiology_reports"
      path: "mrnacc_ultrasound_generate_radreport.pkl"
      required: true
model:
  cache_dir: "./models"
  model_name: "unsloth/Llama-3.3-70B-Instruct-bnb-4bit"
  quantization:
    enabled: false
    bits: null
  cuda_devices: "0,1"
  max_memory: # Optional: omit this section or set to null if not needed
    "0": "32GB"
    "1": "32GB"
inference:
  prompt_template: "default"
  generation:
    do_sample: false
    max_tokens: 32768
  batch_size: 1
output:
  results_dir: "llm_results"
  default_filename: "inference_results.json"
logging:
  level: "INFO"
system:
  num_workers: 1
  continue_on_error: true
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
    --batch-size 4 \
    --output-file "my_results.json" \
    --log-level DEBUG
```

For a full list of available arguments, use the `--help` flag:

```bash
python -m core.run_inference --help
```

## Downloading Models

Use the `download_models_from_HF.py` script to download models from Hugging Face and save them locally. This is useful for offline inference or managing local model caches.

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

### Adding a New Data Loader

If you have data in a format not currently supported (e.g., a custom database, a new file format), you can implement a new data loader.

1.  **Create a new file**: E.g., `implementations/my_data_loader.py`.
2.  **Implement the interface**: Your class should inherit from `DataLoader` (from `core.interfaces`).
    ```python
    # implementations/my_data_loader.py
    from core.interfaces import DataLoader
    from core.config import DataConfig
    # ...

    class MyCustomDataLoader(DataLoader):
        def __init__(self, config: DataConfig):
            super().__init__(config)
            # Your initialization logic

        def load_data(self):
            # Load data from your custom source
            pass

        def get_dataset(self, data_ids=None):
            # Prepare and return a dataset object (e.g., PyTorch Dataset)
            pass

        def get_total_items(self):
            # Return total number of items
            pass

        def validate_data(self, data):
            # Implement data validation
            pass

        def get_processed_ids(self, results_dir, filename):
            # Implement logic to get already processed IDs
            pass
    ```
3.  **Register with Factory**:
    ```python
    from core.interfaces import ComponentFactory
    # ...
    ComponentFactory.register_data_loader("my_custom_data_source", MyCustomDataLoader)
    ```
4.  **Update `config.yaml`**: Set `data.source_type` to `"my_custom_data_source"`.

### Adding a New Prompt Manager

If you need custom prompt formatting logic for specific models or use cases, create a new prompt manager.

1.  **Create a new file**: E.g., `implementations/my_prompt_manager.py`.
2.  **Implement the interface**: Your class should inherit from `PromptManager` (from `core.interfaces`).
    ```python
    # implementations/my_prompt_manager.py
    from core.interfaces import PromptManager
    # ...

    class MyCustomPromptManager(PromptManager):
        def __init__(self, config=None):
            super().__init__(config)
            # Your initialization logic

        def create_messages(self, template_name, data_batch):
            # Generate messages based on template and data
            pass

        def get_available_templates(self):
            # Return a list of available template names
            pass
    ```
3.  **Register with Factory**:
    ```python
    from core.interfaces import ComponentFactory
    # ...
    ComponentFactory.register_prompt_manager("my_custom_prompt", MyCustomPromptManager)
    ```
4.  **Update `config.yaml`**: Set `inference.prompt_manager_type` (or `inference.prompt_template` if your manager handles multiple templates).

### Adding a New Output Manager

To save inference results in a different format or integrate with external systems, implement a new output manager.

1.  **Create a new file**: E.g., `implementations/my_output_manager.py`.
2.  **Implement the interface**: Your class should inherit from `OutputManager` (from `core.interfaces`).
    ```python
    # implementations/my_output_manager.py
    from core.interfaces import OutputManager
    from core.config import OutputConfig
    # ...

    class MyCustomOutputManager(OutputManager):
        def __init__(self, config: OutputConfig):
            super().__init__(config)
            # Your initialization logic

        def save_results(self, results, filename=None):
            # Save results in your custom format
            pass

        def load_results(self, filename=None):
            # Load results for resume functionality
            pass

        def get_output_path(self, filename=None):
            # Return the full path for the output file
            pass

        def backup_existing_results(self, filename):
            # Implement backup logic
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

If you need a different inference strategy (e.g., streaming, specific hardware acceleration, complex orchestration), create a new inference engine.

1.  **Create a new file**: E.g., `implementations/my_inference_engine.py`.
2.  **Implement the interface**: Your class should inherit from `InferenceEngine` (from `core.interfaces`).
    ```python
    # implementations/my_inference_engine.py
    from core.interfaces import InferenceEngine, ModelManager, PromptManager
    from core.config import InferenceConfig
    # ...

    class MyCustomInferenceEngine(InferenceEngine):
        def __init__(self, config: InferenceConfig, model_manager: ModelManager, prompt_manager: PromptManager):
            super().__init__(config, model_manager, prompt_manager)
            # Your initialization logic

        def setup(self):
            # Setup model, tokenizer, and other resources
            pass

        def process_dataset(self, dataset):
            # Implement full dataset processing logic
            pass

        def process_batch(self, batch):
            # Implement batch processing (optional, if needed)
            pass

        def process_single(self, request):
            # Implement single request processing (optional, if needed)
            pass

        def get_stats(self):
            # Return performance statistics
            pass
    ```
3.  **Register with Factory**:
    ```python
    from core.interfaces import ComponentFactory
    # ...
    ComponentFactory.register_inference_engine("my_custom_engine", MyCustomInferenceEngine)
    ```
4.  **Update `config.yaml`**: Set `inference.engine_type` to `"my_custom_engine"`.

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

## Logging

Logging behavior is configured in `config/config.yaml` under the `logging` section. You can specify the logging level, whether to log to a file, and whether to output to the console.

## Troubleshooting

-   **`FileNotFoundError: config.yaml`**: Ensure your `config.yaml` is located in the `config/` directory, or specify its correct path using the `--config` argument.
-   **CUDA Errors / Device Not Recognized**: Verify your `cuda_devices` and `max_memory` settings in `config/config.yaml`. Ensure your `HF_TOKEN` environment variable is set correctly.
-   **Model Loading Issues**: Check if `model_name` is correct and if you have sufficient memory (adjust `max_memory` in `config.yaml` if needed). Ensure your Hugging Face token is correctly set for gated models.
-   **`TypeError: transformers.generation.utils.GenerationMixin.generate()`**: This usually indicates an issue with how inputs are passed to the model's generate method. Ensure the inputs are correctly prepared (e.g., as a `torch.Tensor` or `dict` as expected by the model). (This was addressed in recent updates.) 