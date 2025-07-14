# RadPath LLM Inference

A medical report parsing system that uses Large Language Models to extract structured information from radiology and biopsy reports for thyroid nodule analysis.

## 📋 Table of Contents
- [Setup](#setup)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Customization](#customization)
- [Evaluation](#evaluation)
- [Troubleshooting](#troubleshooting)
- [File Descriptions](#file-descriptions)
- [Contributing](#contributing)
- [License](#license)

## 🚀 Setup

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- Access to medical report datasets

### Installation
1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Configuration
Before running inference, you'll need to update the following paths in the scripts:
- **Data folder**: Update `data_folder` in `llm_batch_inference.py` (line 18)
- **Cache directory**: Update `cache_dir` in `llm_batch_inference.py` (line 32)
- **Model path**: Ensure model cache directory exists on your system

## ⚡ Quick Start

### Basic Inference
```bash
python llm_batch_inference.py
```

The script will prompt you for:
- Output filename (default: `inference_results.json`)
- CUDA device selection
- Model quantization options

### Alternative Inference (Radiology Only)
```bash
python llm_inference_ashwath.py
```

## 📁 Project Structure

```
.
├── llm_batch_inference.py         # Main inference script
├── llm_inference_ashwath.py       # Radiology-only inference
├── helpers/
│   ├── llm_helper.py              # Core helper functions
│   ├── llm_prompts.py             # 🔥 Prompt templates (MODIFY HERE)
│   ├── annotation_dataset.py      # Dataset classes
│   ├── annotation_metrics.py      # Evaluation metrics
│   └── temp.py                    # Utility functions
├── data/
│   ├── mrnacc_ultrasound_generate_radreport.pkl
│   └── mrnacc_ultrasound_generate_bxreport.pkl
├── manual_annotations/
│   ├── dev_corrected.yaml         # Dev set annotations
│   ├── example_YAML.yaml          # Example output format
│   └── annotations_template.yaml  # Template for new annotations
├── parse_llm_results.py           # Parse LLM JSON to YAML
├── eval.py                        # Evaluation script
└── llm_results/                   # Output directory
```

## 🎯 Usage

### 1. Running Inference

#### Main Script: `llm_batch_inference.py`
- **Purpose**: Process both radiology and biopsy reports
- **Output**: Structured YAML with nodule details and matching
- **Features**: Resume capability, multiple MRN sets, GPU monitoring

#### Key Configuration Options:
```python
# Select MRN set (lines 42-90)
mrns = test_mrns      # Change to: dev_mrns, steven_new50, chandler_new50, etc.

# Select prompt template (line 169)
batch_messages = prompt_chat_template_Qwen1(bxreports_batch, radreports_batch)
```

### 2. Processing Results
```bash
# Parse LLM JSON output to YAML
python parse_llm_results.py --filename your_output.json

# Evaluate results
python eval.py --manual dev_corrected.yaml --llm parsed_your_output.yaml
```

### 3. Available MRN Sets
- `dev_mrns`: Development set (90 samples)
- `test_mrns`: Test set (300+ samples)
- `steven_new50`: Steven's annotation set (50 samples)
- `chandler_new50`: Chandler's annotation set (50 samples)
- `bx_loc_error_mrns`: Error cases for biopsy location

## 🛠️ Customization

### Modifying Prompts
The main file to modify is `helpers/llm_prompts.py`. Available prompt templates:

- `prompt_chat_template_2`: Basic extraction
- `prompt_chat_template_Qwen1`: Current production prompt
- `prompt_chat_template_6`: Alternative approach
- `prompt_chat_template_ashwath`: Radiology-only

### Adding New Prompts
1. Create a new function in `helpers/llm_prompts.py`
2. Follow the existing format with system and user messages
3. Update the prompt call in `llm_batch_inference.py`

### Example Prompt Structure:
```python
def prompt_chat_template_new(bxreports, radreports):
    prompts = []
    for bxreport, radreport in zip(bxreports, radreports):
        messages = [
            {"role": "system", "content": "Your system prompt here..."},
            {"role": "user", "content": f"Biopsy_Report: {bxreport}, Radiology_Report: {radreport}"}
        ]
        prompts.append(messages)
    return prompts
```

### Modifying Dataset
- Update data paths in scripts
- Ensure pickle files contain `{mrn: report_text}` format
- Add new MRN lists as needed

## 📊 Evaluation

### Manual Evaluation
```bash
python eval.py --manual manual_annotations/dev_corrected.yaml --llm llm_results/parsed_your_output.yaml
```

### Output Metrics
- **Information Extraction**: Accuracy for each field (location, size, etc.)
- **Nodule Matching**: Precision, Recall, F1 for matching radiology to biopsy nodules
- **Confusion Matrix**: Detailed performance breakdown

### Annotation Format
See `manual_annotations/example_YAML.yaml` for the expected YAML structure.

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size or use higher quantization (4-bit)
   - Set `CUDA_VISIBLE_DEVICES` to use specific GPUs

2. **Model Loading Errors**
   - Verify cache directory exists and has sufficient space
   - Check internet connection for model downloads

3. **Data File Not Found**
   - Update data paths in scripts
   - Ensure pickle files are in the correct directory

4. **YAML Parsing Errors**
   - Check LLM output format
   - Verify `ruamel.yaml` is installed correctly

### Performance Tips
- Use quantization for faster inference
- Monitor GPU usage with `gpustat`
- Resume interrupted runs using existing JSON files

## 📝 File Descriptions

### Core Scripts
- **`llm_batch_inference.py`**: Main inference script with full functionality
- **`llm_inference_ashwath.py`**: Simplified radiology-only inference
- **`parse_llm_results.py`**: Convert JSON outputs to structured YAML
- **`eval.py`**: Comprehensive evaluation against manual annotations

### Helper Modules
- **`helpers/llm_prompts.py`**: All prompt templates (primary customization point)
- **`helpers/llm_helper.py`**: Model loading, GPU monitoring, output parsing
- **`helpers/annotation_dataset.py`**: Dataset classes and data loading
- **`helpers/annotation_metrics.py`**: Evaluation metrics and scoring

### Data Files
- **`thygraph_exp41_radreports.pkl`**: Radiology reports dataset
- **`data/mrnacc_ultrasound_generate_*.pkl`**: Paired radiology/biopsy reports
- **`manual_annotations/`**: Ground truth annotations for evaluation

## 🤝 Contributing

1. Test your changes with a small MRN set first
2. Update documentation if adding new features
3. Follow the existing code style and structure
4. Validate outputs against manual annotations

## 📄 License

[Add your license information here]