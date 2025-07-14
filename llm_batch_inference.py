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


def main():
    ##### Load Data ##########
    data_folder = '/radraid2/dongwoolee/RadPath/data' # change according to your system
    radreport_path = os.path.join(data_folder, 'mrnacc_ultrasound_generate_radreport.pkl')
    with open(radreport_path, 'rb') as f:
        all_radreport = pickle.load(f)

    bxreport_path = os.path.join(data_folder, 'mrnacc_ultrasound_generate_bxreport.pkl')
    with open(bxreport_path, 'rb') as f:
        all_bxreport = pickle.load(f)

    # Set output file for JSON
    output_file = str(input("Enter file name to save the data (end with .json, default='inference_results.json'): "))
    if not output_file:
        output_file = "inference_results.json"

    ##### Load Model #########
    tokenizer, model = load_model(
        cache_dir="/radraid2/dongwoolee/.llms", # change according to your system
        model_name="unsloth/Llama-3.3-70B-Instruct-bnb-4bit",
        #model_name="unsloth/QwQ-32B-unsloth-bnb-4bit"
        )

    ###########################################################################
    #                                Put MRN Here
    ###########################################################################
    dev_mrns = []
    test_mrns = []
    
    mrns = dev_mrns
    print(mrns)
    ############################################################################
    ####### Refine MRNs #######################################################
    output_file = os.path.join(os.getcwd(), "llm_results", output_file)
    elapsed_time_list = []
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

    processed_mrns = set(results.keys())    
    mrns = [mrn for mrn in mrns if mrn not in processed_mrns]
    print(f"Remaining MRNs to process: {len(mrns)}") 
   
    ###### Create Dataset and Define DataLoader ####################################
    dataset = PairedReportDataset(all_radreport, all_bxreport, mrns)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    ###### Process Each Batch and Append Results ############################
    pbar = tqdm(dataloader, desc="Processing Reports")
    for batch in pbar:
        # Unpack the batch (each batch contains one sample since batch_size=1)
        mrns_batch, radreports_batch, bxreports_batch = batch
        mrns_batch = list(mrns_batch)
        radreports_batch = list(radreports_batch)
        bxreports_batch = list(bxreports_batch)

        batch_start_time = time.strftime("%H:%M:%S")
        start_time = time.time()
        pbar.set_postfix({"batch_start": batch_start_time, "MRN": mrns_batch})

        batch_messages = prompt_chat_template_Qwen1(bxreports_batch, radreports_batch)
        batch_generated_texts = generate_text_with_chat_template(tokenizer, model, batch_messages, do_sample=False, temperature=None, top_p=None, max_tokens=32768)
        #batch_generated_texts = generate_text_with_chat_template(tokenizer, model, batch_messages, do_sample=True, temperature=0.6, top_p=0.95, top_k=20, max_tokens=32768) # Qwen
        
        elapsed_time = time.time()-start_time
        elapsed_time_list.append(elapsed_time)
        for mrn, gen_text in zip(mrns_batch, batch_generated_texts):
            results[mrn] = gen_text

        with open(output_file, "w") as outfile:
            json.dump(results, outfile, indent=4)
            outfile.flush()
        
        with open(f"{output_file.strip('.json')}_elapsed_time.json", "w") as f:
            json.dump(elapsed_time_list, f)
            f.flush()

    print("Inference completed. Results appended in 'inference_results.json'.")

if __name__=='__main__':
    main()