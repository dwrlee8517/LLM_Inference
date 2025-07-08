import json
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import os
import torch
import pickle
import re
from torch.utils.data import Dataset
import gpustat
from ruamel.yaml import YAML

class PairedReportDataset(Dataset):
    def __init__(self, all_radreport: dict, all_bxreport: dict, mrns: list):
        self.mrns = mrns
        self.radreports = {mrn: all_radreport[mrn] for mrn in mrns}
        self.bxreports = {mrn: all_bxreport[mrn] for mrn in mrns}

    def __len__(self):
        return len(self.mrns)
    
    def __getitem__(self, index):
        mrn = self.mrns[index]
        return mrn, self.radreports[mrn], self.bxreports[mrn]
    
def show_gpu_status():
    try:
        # Get and print GPU status using gpustat
        gpu_stats = gpustat.new_query()
        print("\nCurrent GPU Status:")
        for gpu in gpu_stats:
            print(f"GPU {gpu.entry['index']}: {gpu.entry['name']} | "
                  f"Memory: {gpu.entry['memory.used']} / {gpu.entry['memory.total']} MB | "
                  f"Utilization: {gpu.entry['utilization.gpu']}% | "
                  f"Processes: {len(gpu.entry['processes'])}")
        print("\n")
    except Exception as e:
        print(f"Error retrieving GPU status: {e}")

def list_directories(cache_dir):
    try:
        print("\nAvailable directories under cache_dir:")
        for item in os.listdir(cache_dir):
            item_path = os.path.join(cache_dir, item)
            if os.path.isdir(item_path):
                print(f"- {item}")
        print("\n")
    except FileNotFoundError:
        print(f"Error: The directory '{cache_dir}' does not exist.")
    except Exception as e:
        print(f"Error listing directories: {e}")

def load_model(cache_dir=None, model_name=None, cuda_devices=None, quantization=None):

    if not cache_dir:
        cache_dir = input("Enter directory of the model (base directory is /radraid/dongwoolee/.models): ")
    if not cache_dir:
        cache_dir = "/radraid/dongwoolee/.models"
    list_directories(cache_dir)
    
    if not model_name:
        model_name = input("Enter model name to load: ").strip()

    show_gpu_status()
    if not cuda_devices:
        cuda_devices = input("Enter the CUDA visible devices (e.g., 0,1,2): ").strip()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_devices)

    if not quantization:
        quantization = input("Enter Quantization of the model (e.g. 4,8 or blank for no quantization): ")
    
    # Load tokenizer and model
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        if not quantization:
            max_memory = {
                0: "32GB",  # Limit for GPU 0
                1: "32GB"   # Limit for GPU 1
            }
            model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, device_map="auto", max_memory=max_memory)
        else:
            if int(quantization) == 8:
                bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            elif int(quantization) == 4:
                bnb_config = BitsAndBytesConfig(load_in_4bit=True)
            else:
                raise ValueError("Not a valid quantization")
            model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, quantization_config=bnb_config, device_map="auto")

        print("Model and tokenizer loaded successfully.")
        show_gpu_status()
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        return None, None
    
    return tokenizer, model

def fix_yaml_format(yaml_content: str) -> str: 
    # Ensure there's a space after colons (for key-value pairs)
    yaml_content = re.sub(r'(?<=\S):(?=\S)', ': ', yaml_content)
    return yaml_content

def parse_llm_output(llm_output:str):
    pattern = r"(?:yaml|yml)(.*?)(?:```|$)"  # Non-greedy match for YAML block
    matches = re.findall(pattern, llm_output, re.DOTALL) 

    if matches:
        yaml_content = matches[-1].strip()  # Extract YAML block and remove extra whitespace
        yaml_content = fix_yaml_format(yaml_content)
        try:
            yaml = YAML(typ='safe')  # 'safe' mode ensures parsing without Python-specific tags
            parsed_yaml = yaml.load(yaml_content)  # Convert YAML to ordered Python dictionary
            return parsed_yaml  # Successfully parsed
        except Exception as e:
            print(f"Error parsing YAML: {e}")  # Handle YAML syntax errors
            print(yaml_content)
            return None
    else:
        print("No YAML block found in LLM output.")  # If no match found
        return None
