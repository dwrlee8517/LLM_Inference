import json
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import os
import torch
import pickle
from tqdm import tqdm
import re
import time
from torch.utils.data import DataLoader, Dataset
import gpustat
from tqdm import tqdm
import time
from helpers.llm_helper import *
from helpers.llm_prompts import *

class RadReportDataset(Dataset):
    def __init__(self, all_radreport: dict):
        self.mrns = list(all_radreport.keys())
        self.radreports = {mrn: all_radreport[mrn] for mrn in self.mrns}

    def __len__(self):
        return len(self.mrns)
    
    def __getitem__(self, index):
        mrn = self.mrns[index]
        return mrn, self.radreports[mrn]

def main():
    ##### Load Data ##########

    with open("thygraph_exp41_radreports.pkl", "rb") as f:
        all_radreport = pickle.load(f) # {mrn: report}

    # Set output file for JSON
    output_file = str(input("Enter file name to save the data (end with .json, default='inference_results.json'): "))
    if not output_file:
        output_file = "inference_results.json"

    ##### Load Model #########
    tokenizer, model = load_model(
        cache_dir="/radraid2/dongwoolee/.llms",
        model_name="unsloth/Llama-3.3-70B-Instruct-bnb-4bit",
        )

    output_file = os.path.join(os.getcwd(), "llm_results", output_file)
    # Load previous results (if any)
    if os.path.exists(output_file):
        try:
            with open(output_file, "r") as f:
                results = json.load(f)  # Load existing results
        except (json.JSONDecodeError, IOError):
            print("Warning: Corrupt or empty JSON file. Resetting results.")
            results = {}  # Reset if file is corrupted
    else:
        results = {}
   
    ###### Create Dataset and Define DataLoader ####################################
    dataset = RadReportDataset(all_radreport)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    ###### Process Each Batch and Append Results ############################
    pbar = tqdm(dataloader, desc="Processing Reports")
    for batch in pbar:
        # Unpack the batch (each batch contains one sample since batch_size=1)
        mrns_batch, radreports_batch = batch
        mrns_batch = list(mrns_batch)
        radreports_batch = list(radreports_batch)

        batch_start_time = time.strftime("%H:%M:%S")
        pbar.set_postfix({"batch_start": batch_start_time, "MRN": mrns_batch})

        batch_messages = prompt_chat_template_ashwath(radreports_batch)
        batch_generated_texts = generate_text_with_chat_template(tokenizer, model, batch_messages, max_tokens=2048)
        
        for mrn, gen_text in zip(mrns_batch, batch_generated_texts):
            results[mrn] = gen_text

        with open(output_file, "w") as outfile:
            json.dump(results, outfile, indent=4)
            outfile.flush()

    print("Inference completed. Results appended in 'inference_results.json'.")

if __name__=='__main__':
    main()