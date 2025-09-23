import os
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
import logging
import time

def download_model(model_name, save_directory, cuda_devices=None):
    """
    Downloads a model and tokenizer from Hugging Face and saves them to a directory.
    """
    if cuda_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
        print(f"Using CUDA devices: {cuda_devices}")

    print(f"Downloading model: {model_name}")
    print(f"Saving to directory: {save_directory}")

    try:
        # Create save directory if it doesn't exist
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
            print(f"Created directory: {save_directory}")

        # Download and save tokenizer
        print("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=save_directory)
        print("Tokenizer downloaded and saved.")

        # Download and save model
        print("Downloading model...")
        start_time = time.time()
        model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=save_directory)
        end_time = time.time()
        print(f"Model downloaded and saved successfully in {end_time - start_time} seconds.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download a model from Hugging Face.")
    
    parser.add_argument(
        "--model_name", 
        type=str, 
        required=True, 
        help="The name of the model on Hugging Face. Should be copied from the model card (e.g., 'meta-llama/Llama-3.3-70B-Instruct')."
    )
    
    parser.add_argument(
        "--save_directory", 
        type=str, 
        required=True, 
        help="The directory where the model should be saved. Should be a path to a directory."
    )
    
    parser.add_argument(
        "--cuda_devices", 
        type=str, 
        help="Optional: Comma-separated list of CUDA devices to use (e.g., '0,1')."
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    HF_TOKEN = os.getenv("HF_TOKEN")
    if not HF_TOKEN:
        logger.error("HF_TOKEN environment variable is not set")
        sys.exit(1)

    logger.info(f"Hugging Face Token has been successfully set")

    download_model(args.model_name, args.save_directory, args.cuda_devices)
