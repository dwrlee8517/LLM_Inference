# LLM Batch Inference System

Configurable pipeline to run Hugging Face LLMs over your datasets (single radiology reports or paired radiology/biopsy reports). It supports file-driven prompts, preprocessors, incremental JSONL outputs with resume, and simple extension points for data readers and preprocessors.

## Table of Contents
- [Features](#features)
- [Requirements & Setup](#requirements--setup)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
  - [Single-Report example (`RadReport`)](#single-report-example-radreport)
  - [Paired-Report example (`RadPath`)](#paired-report-example-radpath)
- [Running Inference](#running-inference)
  - [Dry run](#dry-run)
  - [Common CLI overrides](#common-cli-overrides)
- [Outputs, Resume, and Interrupts](#outputs-resume-and-interrupts)
- [Datasets & DataReaders](#datasets--datareaders)
  - [Built-in readers](#built-in-readers)
  - [Create a custom DataReader](#create-a-custom-datareader)
- [Prompts & Preprocessors](#prompts--preprocessors)
  - [Prompt YAML schema](#prompt-yaml-schema)
  - [Create a custom Preprocessor](#create-a-custom-preprocessor)
- [Models, and Support Matrix](#models-and-support-matrix)
- [Logging](#logging)
- [Troubleshooting](#troubleshooting)

## Features
- File-based configuration with CLI overrides
- Preprocessors build model-ready inputs from your dataset tuples and prompt YAMLs
- Incremental JSONL writes (safe for long runs) and resume from existing results
- Simple factories to register new data readers or preprocessors
- Suppressed noisy logs from urllib3/accelerate for cleaner output

## Requirements & Setup
```bash
pip install -r requirements.txt

# Optional: set your Hugging Face token for gated models
export HF_TOKEN="<your_hf_token>"
```

## Quick Start
1) Pick and edit a config:
- `configs/RadReportConfig.yaml` for single-report datasets
- `configs/RadPathConfig.yaml` for paired-report datasets

2) Run a dry run to verify the effective config:
```bash
python -m core.run_inference --dry-run --config configs/RadReportConfig.yaml
```

3) Run inference:
```bash
python -m core.run_inference --config configs/RadReportConfig.yaml
```

Results are written to `llm_results/<filename>.jsonl` by default.

## Configuration
Configurations live in `configs/*.yaml`. Key sections: `data`, `model`, `inference`, `output`, `logging`, `system` (see examples below). Most users only need to change paths, model name, preprocessor, and prompt file.

### Single-Report example (`RadReport`)
```yaml
# configs/RadReportConfig.yaml
data:
  source_type: "single_report"   # Built-in reader
  data_folder: "/radraid2/dongwoolee/LLM_Inference/data"
  input_files:
    - name: "radiology_reports"
      path: "thygraph_exp41_radreports.pkl"
      required: true
  process_all: false
  custom_ids: ['5090489-47314031']
  id_file: null
  resume_from_existing: true
  check_processed: true

model:
  cache_dir: "/radraid2/dongwoolee/.llms"
  model_name: "meta-llama/Llama-3.1-8B-Instruct"
  cuda_devices: "0"
  # max_memory: {"0": "40GB"}

inference:
  preprocessor: "RadReport"
  prompt_file: "prompts/radreport.yaml"
  generation:
    do_sample: false
    temperature: null
    top_p: null
    top_k: null
    max_new_tokens: 4096
  # Batch is internally 1; iteration is sample-by-sample
  batch_size: 1
  shuffle: false
  multimodal: false

output:
  results_dir: "llm_results"
  default_filename: "inference_results.jsonl"
  format: "jsonl"          # jsonl recommended for long runs
  save_elapsed_time: true
  save_intermediate_results: true
  backup_existing: true     # backup is deleted after successful save
  add_timestamp: false

logging:
  level: "INFO"
  file: { enabled: true, path: "logs/inference.log", level: "DEBUG" }
  console: true
  progress_bar: true
  gpu_monitoring: true

system:
  num_workers: 1
  pin_memory: true
  continue_on_error: true
  max_retries: 3
  retry_delay: 5
  memory_threshold: 0.9
  gpu_memory_threshold: 0.95
```

### Paired-Report example (`RadPath`)
```yaml
# configs/RadPathConfig.yaml
data:
  source_type: "paired_report"
  data_folder: "/radraid2/dongwoolee/LLM_Inference/data"
  input_files:
    - name: "radiology_reports"
      path: "mrnacc_ultrasound_generate_radreport.pkl"
      required: true
    - name: "biopsy_reports"
      path: "mrnacc_ultrasound_generate_bxreport.pkl"
      required: true
  process_all: false
  custom_ids: ['mrn0-acc0']
  id_file: null
  resume_from_existing: true
  check_processed: true

model:
  cache_dir: "/radraid2/dongwoolee/.llms"
  model_name: "meta-llama/Llama-3.1-8B-Instruct"
  cuda_devices: "0,1"

inference:
  preprocessor: "RadPath"
  prompt_file: "prompts/radpath_1.yaml"
  generation: { do_sample: false, temperature: null, top_p: null, top_k: null, max_new_tokens: 4096 }
  batch_size: 1
  shuffle: false
  multimodal: false

output:
  results_dir: "llm_results"
  default_filename: "inference_results.jsonl"
  format: "jsonl"
  save_elapsed_time: true
  save_intermediate_results: true
  backup_existing: true
  add_timestamp: false

logging:
  level: "INFO"
  file: { enabled: true, path: "logs/inference.log", level: "DEBUG" }
  console: true
  progress_bar: true
  gpu_monitoring: true

system:
  num_workers: 1
  pin_memory: true
  continue_on_error: true
  max_retries: 3
  retry_delay: 5
  memory_threshold: 0.9
  gpu_memory_threshold: 0.95
```

## Running Inference
```bash
# Show effective configuration and exit
python -m core.run_inference --dry-run --config configs/RadReportConfig.yaml

# Run inference
python -m core.run_inference --config configs/RadReportConfig.yaml

# List supported model patterns
python -m core.run_inference --list-models
```

### Dry run
`--dry-run` prints the fully merged config, the resolved model manager, and key values; no inference is executed.

### Common CLI overrides
Only frequently used overrides are shown here (see `--help` for all):
```bash
python -m core.run_inference --config configs/RadReportConfig.yaml \
  --model-name meta-llama/Llama-3.1-8B-Instruct \
  --cache-dir /radraid2/dongwoolee/.llms \
  --cuda-devices 0 \
  --max-tokens 1024 \
  --data-folder /path/to/data \
  --custom-ids id1 id2 id3 \
  --results-dir llm_results \
  --output-file my_results.jsonl
```

Notes:
- `--max-tokens` maps to `inference.generation.max_new_tokens`.
- `preprocessor` and `prompt_file` are set in the config (no dedicated CLI flags).

## Outputs, Resume, and Interrupts
- `output.format: jsonl` is recommended. Each result is appended as a single line: `{ "id": "<ID>", "result": "<text>" }`.
- Resume: with `data.resume_from_existing: true`, previously saved IDs are skipped automatically.
- Interrupts: on `Ctrl+C`, a `processed_ids.txt` is written under the output directory with IDs that were saved so far.
- Successful `save_results` deletes any timestamped backup created before the write.

## Datasets & DataReaders

### Built-in readers
- `single_report` → expects one mapping of `ID -> text` (supports `.pkl`, `.json`, `.csv`).
  - CSV default columns: `mrn` and `report_text` (configurable per file via `mrn_column`, `text_column`).
- `paired_report` → expects two mappings of `ID -> text` for radiology and biopsy (supports `.pkl`, `.json`, `.csv`).
  - CSV default columns: `mrn` and `text`.

Datasets return tuples:
- `SingleReportDataset.__getitem__(i) -> (mrn, radreport)`
- `PairedReportDataset.__getitem__(i) -> (mrn, radreport, bxreport)`

Filtering: both datasets implement `filter_ids(ids)` to subset before inference.

### Create a custom DataReader
Register a new reader if your storage format isn’t covered.
```python
from core.interfaces import DataReader, ComponentFactory
from core.config import DataConfig
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, records): self.records = records
    def __len__(self): return len(self.records)
    def __getitem__(self, i): return self.records[i]  # e.g., (mrn, text)

class MyDataReader(DataReader):
    def __init__(self, config: DataConfig): self.config = config
    def load_data(self):
        # Populate `dataset` and `all_ids`
        dataset = MyDataset(records)
        return dataset, all_ids
    def get_all_ids(self):
        return list(all_ids)

ComponentFactory.register_data_reader("my_source", MyDataReader)
```
Then set `data.source_type: "my_source"` in your config.

## Prompts & Preprocessors
Preprocessors build the model input from dataset tuples plus a YAML prompt. Two built-ins are provided:
- `RadReport` (single report)
- `RadPath` (paired rad + biopsy)

### Prompt YAML schema
```yaml
system: |
  You are an expert assistant... (global instruction)
user_template: |
  Radiology_Report: {radreport}
  # For RadPath, {bxreport} is also available
```

### Create a custom Preprocessor
```python
from core.interfaces import Preprocessor, ComponentFactory
from core.config import InferenceConfig

class MyPreprocessor(Preprocessor):
    required_keys = {"radreport"}  # or {"radreport", "bxreport"}
    def __init__(self, config: InferenceConfig): self.config = config
    def prepare_inputs(self, batch):
        # batch comes from a DataLoader with batch_size=1
        # return ([mrn], [raw_item]) where raw_item is either
        # (system_text, user_text) or (system_text, [parts]) for multimodal
        return [mrn], [(system_text, user_text)]

ComponentFactory.register_preprocessor("MyPrep", MyPreprocessor)
```
Then set in config: `inference.preprocessor: "MyPrep"` and point `inference.prompt_file` to your YAML.

## Models and Support Matrix
- Set your model via `model.model_name` (e.g., `meta-llama/Llama-3.1-8B-Instruct`).
- Use `--list-models` to see supported model families/patterns mapped to internal managers.
- Device placement is automatic (`device_map: auto`). Inputs are placed to the model’s device to avoid mismatches.

## Logging
Configured by `logging` in your config. By default, `urllib3` and `accelerate` loggers are suppressed below ERROR for cleaner logs. File logging can be enabled with its own level.

## Troubleshooting
- No/gated model access: ensure `HF_TOKEN` is set (`echo $HF_TOKEN`).
- CUDA/device errors: check `model.cuda_devices` and available memory; reduce `max_new_tokens`.
- Resume didn’t skip: ensure `output.format: jsonl` and `data.resume_from_existing: true`; also ensure the same `default_filename`.
- Processed too many/too few: verify `data.process_all`, `data.custom_ids`, or `data.id_file`.
- CSV parsing: set `mrn_column` and `text_column` per file in `data.input_files`.
- Key errors on generation config: use `inference.generation.max_new_tokens` (not `max_tokens`).

---