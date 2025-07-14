# LLM Inference System Refactoring Guide

## 🎯 Overview

This document outlines the complete refactoring of the original `llm_batch_inference.py` script into a modular, configurable, and extensible system.

## 📋 Table of Contents

- [Problems with Original System](#problems-with-original-system)
- [Refactored Architecture](#refactored-architecture)
- [Key Benefits](#key-benefits)
- [File Structure](#file-structure)
- [How to Use](#how-to-use)
- [Extension Guide](#extension-guide)
- [Migration from Original](#migration-from-original)
- [Best Practices](#best-practices)

## 🚨 Problems with Original System

### 1. **Hardcoded Configuration**
```python
# Original - Everything hardcoded
data_folder = '/radraid2/dongwoolee/RadPath/data'
cache_dir = "/radraid2/dongwoolee/.llms"
model_name = "unsloth/Llama-3.3-70B-Instruct-bnb-4bit"
```

### 2. **Monolithic Structure**
- Single 200+ line function
- All concerns mixed together
- Hard to test individual components

### 3. **Limited Extensibility**
- Adding new data sources requires code changes
- New model types need core modifications
- Prompt management is scattered

### 4. **Poor Error Handling**
- No systematic error management
- Limited logging and monitoring
- No graceful degradation

## 🏗️ Refactored Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Configuration Layer                      │
├─────────────────────────────────────────────────────────────┤
│  config.yaml + ConfigManager + Command Line Interface       │
└─────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────┐
│                    Interface Layer                          │
├─────────────────────────────────────────────────────────────┤
│  Abstract Base Classes + Factory Pattern                    │
└─────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────┐
│                  Implementation Layer                       │
├─────────────────────────────────────────────────────────────┤
│  DataLoaders │ ModelManagers │ PromptManagers │ OutputMgrs  │
└─────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────┐
│                   Orchestration Layer                       │
├─────────────────────────────────────────────────────────────┤
│              LLMInferenceOrchestrator                       │
└─────────────────────────────────────────────────────────────┘
```

### Key Components

1. **Configuration System** (`core/config.py`)
   - YAML-based configuration
   - Command-line overrides
   - Environment-specific settings

2. **Interface Layer** (`core/interfaces.py`)
   - Abstract base classes
   - Factory pattern for component creation
   - Type safety and contracts

3. **Implementation Layer** (`helpers/`)
   - Concrete implementations
   - Plugin-style architecture
   - Easy to extend and replace

4. **Orchestration** (`refactored_main.py`)
   - Coordinates all components
   - Manages workflow
   - Error handling and cleanup

## ✨ Key Benefits

### 1. **Configuration-Driven**
```yaml
# config.yaml
data:
  source_type: "pickle"
  data_folder: "/your/data/path"
  
model:
  model_name: "your-model"
  cache_dir: "/your/cache/path"
```

### 2. **Modular Design**
```python
# Easy to swap components
data_loader = ComponentFactory.create_data_loader("csv", config)
model_manager = ComponentFactory.create_model_manager("huggingface", config)
```

### 3. **Extensible Architecture**
```python
# Add new data loader
class DatabaseDataLoader(DataLoader):
    def load_data(self, config):
        # Custom database loading logic
        pass

# Register with factory
ComponentFactory.register_data_loader("database", DatabaseDataLoader)
```

### 4. **Better Error Handling**
```python
# Systematic error management
try:
    results = orchestrator.run_inference()
except DataValidationError as e:
    logger.error(f"Data validation failed: {e}")
except ModelLoadingError as e:
    logger.error(f"Model loading failed: {e}")
```

## 📁 File Structure

```
project/
├── config.yaml                    # Main configuration file
├── refactored_main.py             # Main entry point
├── core/
│   ├── __init__.py
│   ├── config.py                  # Configuration management
│   └── interfaces.py              # Abstract base classes
├── implementations/
│   ├── __init__.py
│   ├── data_loaders.py           # Data loading implementations
│   ├── model_managers.py         # Model management (to be created)
│   ├── prompt_managers.py        # Prompt management (to be created)
│   └── output_managers.py        # Output management (to be created)
├── plugins/                       # Optional plugin directory
└── tests/                         # Unit tests
```

## 🚀 How to Use

### Basic Usage

```bash
# Use default configuration
python refactored_main.py

# Use custom configuration file
python refactored_main.py --config my_config.yaml

# Override specific settings
python refactored_main.py --data-folder /path/to/data --model-name custom-model

# Show configuration without running
python refactored_main.py --dry-run
```

### Advanced Usage

```bash
# Process specific custom IDs
python refactored_main.py --custom-ids "ID1" "ID2" "ID3" --output-file custom_results.json

# Process IDs from file
python refactored_main.py --id-file ids.txt --output-file file_results.json

# Process all available IDs
python refactored_main.py --process-all --output-file all_results.json

# Custom quantization and devices
python refactored_main.py --quantization 8 --cuda-devices "0,1,2,3"

# Different output format
python refactored_main.py --format yaml --results-dir /custom/output/path
```

## 🔧 Extension Guide

### Adding a New Data Loader

1. **Create the Implementation**
```python
# implementations/data_loaders.py
class DatabaseDataLoader(DataLoader):
    def __init__(self, config: DataConfig):
        self.config = config
        self.connection = None
    
    def load_data(self, config: Dict[str, Any]) -> Dict[str, Any]:
        # Connect to database
        self.connection = create_database_connection(config)
        
        # Load data
        radreports = self.connection.execute("SELECT * FROM radreports")
        bxreports = self.connection.execute("SELECT * FROM bxreports")
        
        return {
            'radiology_reports': dict(radreports),
            'biopsy_reports': dict(bxreports)
        }
    
    def validate_data(self, data: Dict[str, Any]) -> bool:
        # Custom validation logic
        return True
```

2. **Register with Factory**
```python
# At the end of implementations/data_loaders.py
ComponentFactory.register_data_loader("database", DatabaseDataLoader)
```

3. **Update Configuration**
```yaml
# config.yaml
data:
  source_type: "database"
  connection_string: "postgresql://user:pass@localhost/db"
```

### Adding a New Model Manager

1. **Create the Implementation**
```python
# implementations/model_managers.py
class HuggingFaceModelManager(ModelManager):
    def load_model(self, config: Dict[str, Any]) -> Tuple[Any, Any]:
        # Load HuggingFace model
        tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
        model = AutoModelForCausalLM.from_pretrained(config['model_name'])
        return tokenizer, model
    
    def generate(self, tokenizer, model, messages, generation_config):
        # Custom generation logic
        pass
```

2. **Register and Configure**
```python
ComponentFactory.register_model_manager("huggingface", HuggingFaceModelManager)
```

### Adding a New Prompt Template

1. **Create the Template**
```python
# helpers/llm_prompts.py
def prompt_chat_template_custom(bxreports, radreports):
    # Your custom prompt logic
    pass
```

2. **Update Configuration**
```yaml
inference:
  prompt_template: "prompt_chat_template_custom"
```

### Adding Custom Output Formats

1. **Create Output Manager**
```python
# implementations/output_managers.py
class CSVOutputManager(OutputManager):
    def save_results(self, results, output_config):
        # Save to CSV format
        pass
```

2. **Register and Configure**
```python
ComponentFactory.register_output_manager("csv", CSVOutputManager)
```

## 🔄 Migration from Original

### Step 1: Update Configuration

```python
# Original
data_folder = '/radraid2/dongwoolee/RadPath/data'
cache_dir = "/radraid2/dongwoolee/.llms"
model_name = "unsloth/Llama-3.3-70B-Instruct-bnb-4bit"

# New - config.yaml
data:
  data_folder: "/radraid2/dongwoolee/RadPath/data"
model:
  cache_dir: "/radraid2/dongwoolee/.llms"
  model_name: "unsloth/Llama-3.3-70B-Instruct-bnb-4bit"
```

### Step 2: Update ID Selection

```python
# Original
dev_mrns = [...]
test_mrns = [...]
mrns = dev_mrns  # Hardcoded selection

# New - config.yaml
data:
  # ID selection options:
  process_all: true              # Process all available IDs
  custom_ids: [...]              # List of specific IDs to process
  id_file: "ids.txt"             # File containing IDs to process
```

### Step 3: Update Prompt Selection

```python
# Original
batch_messages = prompt_chat_template_Qwen1(bxreports_batch, radreports_batch)

# New - config.yaml
inference:
  prompt_template: "prompt_chat_template_Qwen1"
```

### Step 4: Run Migration Script

```bash
# Create migration script
python migrate_from_original.py --original-config original_settings.py --output config.yaml
```

## 💡 Best Practices

### 1. **Configuration Management**
- Use environment-specific config files
- Never hardcode paths or credentials
- Use environment variables for sensitive data

### 2. **Component Development**
- Follow the interface contracts
- Add comprehensive logging
- Include proper error handling
- Write unit tests

### 3. **Testing**
```python
# Unit test example
def test_pickle_data_loader():
    config = DataConfig(source_type="pickle", ...)
    loader = PickleDataLoader(config)
    
    data = loader.load_data(config)
    assert loader.validate_data(data)
```

### 4. **Deployment**
- Use configuration files for different environments
- Implement health checks
- Add monitoring and alerting
- Use containerization for consistency

### 5. **Performance**
- Profile different components
- Implement caching where appropriate
- Use async processing for I/O operations
- Monitor resource usage

## 🆚 Comparison: Original vs Refactored

| Aspect | Original | Refactored |
|--------|----------|------------|
| **Configuration** | Hardcoded | YAML + CLI |
| **Extensibility** | Monolithic | Plugin-based |
| **Testing** | Difficult | Unit testable |
| **Deployment** | Manual setup | Config-driven |
| **Error Handling** | Basic | Comprehensive |
| **Logging** | Minimal | Structured |
| **Customization** | Code changes | Configuration |
| **Maintenance** | High effort | Low effort |

## 🎯 Next Steps

1. **Complete Implementation**
   - Implement remaining managers (Model, Prompt, Output)
   - Add monitoring and caching systems
   - Create comprehensive tests

2. **Advanced Features**
   - Add plugin system for custom components
   - Implement distributed processing
   - Add real-time monitoring dashboard

3. **Documentation**
   - Create API documentation
   - Add usage examples
   - Write deployment guides

4. **Community**
   - Create contribution guidelines
   - Set up CI/CD pipeline
   - Add issue templates

## 📚 Resources

- [Configuration Management Best Practices](https://12factor.net/config)
- [Factory Pattern in Python](https://refactoring.guru/design-patterns/factory-method/python/example)
- [Abstract Base Classes](https://docs.python.org/3/library/abc.html)
- [Logging Best Practices](https://docs.python.org/3/howto/logging.html)

This refactoring transforms your specific medical report processing system into a general-purpose, extensible LLM inference framework that can be easily adapted for different domains and use cases. 