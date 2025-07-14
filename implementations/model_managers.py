"""
Concrete implementation of the ModelManager for Hugging Face Transformers.
"""
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import Tuple, Any, List, Dict
import logging

from core.interfaces import ModelManager, ComponentFactory
from core.config import ModelConfig, GenerationConfig

logger = logging.getLogger(__name__)

class TransformersModelManager(ModelManager):
    """Manages Hugging Face Transformers models."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        logger.info("Initialized TransformersModelManager")

    def load_model(self) -> Tuple[Any, Any]:
        """Load tokenizer and model from Hugging Face."""
        logger.info(f"Loading model: {self.config.model_name}")
        
        os.environ["CUDA_VISIBLE_DEVICES"] = self.config.cuda_devices
        
        bnb_config = None
        if self.config.quantization.enabled:
            if self.config.quantization.bits == 8:
                bnb_config = BitsAndBytesConfig(load_in_8bit=True)
                logger.info("Quantization enabled: 8-bit")
            elif self.config.quantization.bits == 4:
                bnb_config = BitsAndBytesConfig(load_in_4bit=True)
                logger.info("Quantization enabled: 4-bit")
            else:
                raise ValueError("Invalid quantization bits specified in config.")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name, cache_dir=self.config.cache_dir
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                cache_dir=self.config.cache_dir,
                quantization_config=bnb_config,
                device_map="auto",
                max_memory=self.config.max_memory
            )
            
            logger.info("Model and tokenizer loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading model or tokenizer: {e}")
            raise

        return self.tokenizer, self.model

    def generate(
        self,
        messages: List[Dict[str, str]],
        generation_config: GenerationConfig,
    ) -> List[str]:
        """Generate text using the loaded model."""
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model and tokenizer must be loaded before generation.")

        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128000,
        ).to("cuda")

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=generation_config.max_tokens,
            do_sample=generation_config.do_sample,
            temperature=generation_config.temperature,
            top_p=generation_config.top_p,
            top_k=generation_config.top_k,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        processed_texts = []
        for output in outputs:
            decoded = self.tokenizer.decode(output, skip_special_tokens=True)
            # Process to extract only the assistant's response
            processed = decoded.split("assistant")[-1].strip().strip("`").strip('json')
            processed_texts.append(processed)

        return processed_texts

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "model_name": self.config.model_name,
            "quantization": self.config.quantization.__dict__,
            "is_loaded": self.model is not None,
        }

    def cleanup(self) -> None:
        """Clean up model and tokenizer resources."""
        self.model = None
        self.tokenizer = None
        torch.cuda.empty_cache()
        logger.info("Model resources cleaned up.")

# Register the model manager with the factory
ComponentFactory.register_model_manager("transformers", TransformersModelManager) 