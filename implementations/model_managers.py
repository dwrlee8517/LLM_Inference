"""
Concrete implementation of the ModelManager for Hugging Face Transformers.
"""
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoModelForImageTextToText, AutoProcessor
from typing import Tuple, Any, List, Dict, Type
import logging
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.interfaces import ModelManager, ComponentFactory
from core.config import ModelConfig, GenerationConfig

logger = logging.getLogger(__name__)

class BaseHuggingFaceManager(ModelManager):
    """
    Base class for Hugging Face model managers, providing a common structure.
    Subclasses should define `_model_name_patterns` and override model-specific logic.
    """
    _model_name_patterns: List[str] = [] # List of patterns (e.g., "unsloth/", "meta-llama/")

    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        logger.info(f"Initialized {self.__class__.__name__}")

    def load_model(self) -> Tuple[Any, Any]:
        """Load tokenizer and model."""
        logger.info(f"Loading model: {self.config.model_name}")
        os.environ["CUDA_VISIBLE_DEVICES"] = self.config.cuda_devices
        
        bnb_config = self._create_bnb_config()

        try:
            self.model, self.tokenizer = self._load_specific_model_and_tokenizer(bnb_config)
            logger.info("Model and tokenizer loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading model or tokenizer: {e}", exc_info=True)
            raise

        return self.model, self.tokenizer
        
    def _create_bnb_config(self) -> BitsAndBytesConfig:
        """Create BitsAndBytesConfig if quantization is enabled."""
        if self.config.quantization.enabled:
            if self.config.quantization.bits == 8:
                logger.info("Quantization enabled: 8-bit")
                return BitsAndBytesConfig(load_in_8bit=True)
            elif self.config.quantization.bits == 4:
                logger.info("Quantization enabled: 4-bit")
                return BitsAndBytesConfig(load_in_4bit=True)
            else:
                raise ValueError("Invalid quantization bits specified in config.")
        return None

    def _load_specific_model_and_tokenizer(self, bnb_config: BitsAndBytesConfig) -> Tuple[Any, Any]:
        """Default loading logic using Hugging Face. Subclasses should replace this method."""
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name, cache_dir=self.config.cache_dir
        )

        model_kwargs = {
            "pretrained_model_name_or_path": self.config.model_name,
            "cache_dir": self.config.cache_dir,
            "device_map": "auto",
        }

        if self.config.max_memory:
            # Convert string keys to integers for GPU devices in max_memory
            max_memory = {}
            for key, value in self.config.max_memory.items():
                if key.isdigit():
                    max_memory[int(key)] = value
                else:
                    max_memory[key] = value
            model_kwargs["max_memory"] = max_memory

        if bnb_config is not None:
            model_kwargs["quantization_config"] = bnb_config

        model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
        return model, tokenizer

    def generate(self, messages: List[Dict[str, str]], generation_config: GenerationConfig,) -> List[str]:
        """This is a default implementation of the generate method.
        Subclasses should override this method to provide model-specific generation logic.
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model and tokenizer must be loaded before generation.")

        inputs = self._convert_to_specific_chat_template(messages)
        outputs = self._generate_from_specific_chat_template(inputs, generation_config)

        processed_texts = []
        for output in outputs:
            decoded = self.tokenizer.decode(output, skip_special_tokens=True)
            processed = self._post_process_output(decoded)
            processed_texts.append(processed)

        return processed_texts
    
    def _convert_to_specific_chat_template(self, messages: List[Dict[str, str]]) -> torch.Tensor:
        """Convert the messages to the specific chat template for the model."""
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128000,
        ).to("cuda")
        return inputs
    
    def _generate_from_specific_chat_template(self, inputs: torch.Tensor, generation_config: GenerationConfig) -> List[str]:
        """Generate from the specific chat template for the model."""
        outputs = self.model.generate(
            inputs,
            max_new_tokens=generation_config.max_tokens,
            do_sample=generation_config.do_sample,
            temperature=generation_config.temperature,
            top_p=generation_config.top_p,
            top_k=generation_config.top_k,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        return outputs
    
    def _post_process_output(self, decoded_text: str) -> str:
        """Default post-processing logic. Subclasses can override."""
        return decoded_text.split("assistant")[-1].strip().strip("`").strip('json')

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "model_family": self.__class__.__name__,
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

class LlamaModelManager(BaseHuggingFaceManager):
    """Model manager specifically for Meta's Llama models."""
    _model_name_patterns = [
        "meta-llama/",
        "unsloth/llama-",
    ]

    def _load_specific_model_and_tokenizer(self, bnb_config: BitsAndBytesConfig) -> Tuple[Any, Any]:
        """Load model and tokenizer using huggingface."""

        model_kwargs = {
            "pretrained_model_name_or_path": self.config.model_name,
            "cache_dir": self.config.cache_dir,
            "device_map": "auto",
        }

        if self.config.max_memory:
            # Convert string keys to integers for GPU devices in max_memory
            max_memory = {}
            for key, value in self.config.max_memory.items():
                if key.isdigit():
                    max_memory[int(key)] = value
                else:
                    max_memory[key] = value
            model_kwargs["max_memory"] = max_memory

        if bnb_config is not None:
            model_kwargs["quantization_config"] = bnb_config

        model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
        
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            cache_dir=self.config.cache_dir,
        )
        return model, tokenizer
    
    def _convert_to_specific_chat_template(self, messages: List[Dict[str, str]]) -> torch.Tensor:
        """Convert the messages to the specific chat template for the model."""
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128000,
        ).to("cuda")
        return inputs
    
    def _generate_from_specific_chat_template(self, inputs: torch.Tensor, generation_config: GenerationConfig) -> List[str]:
        """Generate from the specific chat template for the model."""
        outputs = self.model.generate(
            inputs,
            max_new_tokens=generation_config.max_tokens,
            do_sample=generation_config.do_sample,
            temperature=generation_config.temperature,
            top_p=generation_config.top_p,
            top_k=generation_config.top_k,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        return outputs

    def _post_process_output(self, decoded_text: str) -> str:
        """Llama-specific post-processing logic."""
        return decoded_text.split("assistant")[-1].strip().strip("`").strip('json')
    
class Qwen3ModelManager(BaseHuggingFaceManager):
    """Model manager specifically for Qwen3 models."""
    _model_name_patterns = [
        "Qwen3/",
    ]

    def _load_specific_model_and_tokenizer(self, bnb_config: BitsAndBytesConfig) -> Tuple[Any, Any]:
        """Load model and tokenizer using huggingface."""

        model_kwargs = {
            "pretrained_model_name_or_path": self.config.model_name,
            "cache_dir": self.config.cache_dir,
            "device_map": "auto",
        }

        if self.config.max_memory:
            # Convert string keys to integers for GPU devices in max_memory
            max_memory = {}
            for key, value in self.config.max_memory.items():
                if key.isdigit():
                    max_memory[int(key)] = value
                else:
                    max_memory[key] = value
            model_kwargs["max_memory"] = max_memory

        if bnb_config is not None:
            model_kwargs["quantization_config"] = bnb_config

        model = AutoModelForCausalLM.from_pretrained(**model_kwargs)

        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            cache_dir=self.config.cache_dir,
        )
        return model, tokenizer
    
    def _convert_to_specific_chat_template(self, messages: List[Dict[str, str]]) -> torch.Tensor:
        """Convert the messages to the specific chat template for the model."""
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking = True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128000,
        ).to("cuda")
        return inputs
    
    def _generate_from_specific_chat_template(self, inputs: torch.Tensor, generation_config: GenerationConfig) -> List[str]:
        """Generate from the specific chat template for the model."""
        outputs = self.model.generate(
            inputs,
            max_new_tokens=generation_config.max_tokens,
            do_sample=generation_config.do_sample,
            temperature=generation_config.temperature,
            top_p=generation_config.top_p,
            top_k=generation_config.top_k,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        return outputs
    
    def _post_process_output(self, decoded_text: str) -> str:
        """Qwen3-specific post-processing logic."""
        return decoded_text.split("assistant")[-1].strip().strip("`").strip('json')
    
class QwQModelManager(BaseHuggingFaceManager):
    """Model manager specifically for QwQ models."""
    _model_name_patterns = [
        "Qwen/QwQ-",
        "unsloth/QwQ-",
    ]

    def _load_specific_model_and_tokenizer(self, bnb_config: BitsAndBytesConfig) -> Tuple[Any, Any]:
        """Load model and tokenizer using huggingface."""

        model_kwargs = {
            "pretrained_model_name_or_path": self.config.model_name,
            "cache_dir": self.config.cache_dir,
            "device_map": "auto",
        }

        if self.config.max_memory:
            # Convert string keys to integers for GPU devices in max_memory
            max_memory = {}
            for key, value in self.config.max_memory.items():
                if key.isdigit():
                    max_memory[int(key)] = value
                else:
                    max_memory[key] = value
            model_kwargs["max_memory"] = max_memory

        if bnb_config is not None:
            model_kwargs["quantization_config"] = bnb_config

        model = AutoModelForCausalLM.from_pretrained(**model_kwargs)

        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            cache_dir=self.config.cache_dir,
        )
        return model, tokenizer
    
    def _convert_to_specific_chat_template(self, messages: List[Dict[str, str]]) -> torch.Tensor:
        """Convert the messages to the specific chat template for the model."""
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128000,
        ).to("cuda")
        return inputs
    
    def _generate_from_specific_chat_template(self, inputs: torch.Tensor, generation_config: GenerationConfig) -> List[str]:
        """Generate from the specific chat template for the model."""
        outputs = self.model.generate(
            inputs,
            max_new_tokens=generation_config.max_tokens,
            do_sample=generation_config.do_sample,
            temperature=generation_config.temperature,
            top_p=generation_config.top_p,
            top_k=generation_config.top_k,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        return outputs
    
    def _post_process_output(self, decoded_text: str) -> str:
        """Qwen3-specific post-processing logic."""
        return decoded_text.split("assistant")[-1].strip().strip("`").strip('json')

class MedGemmaModelManager(BaseHuggingFaceManager):
    """Model manager specifically for MedGemma models."""
    _model_name_patterns = [
        "medgemma/",
        "google/medgemma-",
    ]

    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.processor = None
        logger.info(f"Initialized {self.__class__.__name__}")

    def load_model(self) -> Tuple[Any, Any]:
        """Load tokenizer and model."""
        logger.info(f"Loading model: {self.config.model_name}")
        os.environ["CUDA_VISIBLE_DEVICES"] = self.config.cuda_devices
        
        bnb_config = self._create_bnb_config()

        try:
            self.model, self.processor = self._load_specific_model_and_tokenizer(bnb_config)
            logger.info("Model and tokenizer loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading model or tokenizer: {e}", exc_info=True)
            raise

        return self.model, self.processor
    
    def _load_specific_model_and_tokenizer(self, bnb_config: BitsAndBytesConfig) -> Tuple[Any, Any]:
        """Load model and tokenizer using huggingface."""

        model_kwargs = {
            "pretrained_model_name_or_path": self.config.model_name,
            "cache_dir": self.config.cache_dir,
            "torch_dtype": torch.bfloat16,
            "device_map": "auto",
        }

        if self.config.max_memory:
            # Convert string keys to integers for GPU devices in max_memory
            max_memory = {}
            for key, value in self.config.max_memory.items():
                if key.isdigit():
                    max_memory[int(key)] = value
                else:
                    max_memory[key] = value
            model_kwargs["max_memory"] = max_memory

        if bnb_config is not None:
            model_kwargs["quantization_config"] = bnb_config

        model = AutoModelForImageTextToText.from_pretrained(**model_kwargs)

        processor = AutoProcessor.from_pretrained(
            self.config.model_name,
            cache_dir=self.config.cache_dir,
        )

        return model, processor
    
    def generate(self, messages: List[Dict[str, str]], generation_config: GenerationConfig,) -> List[str]:
        """This is a default implementation of the generate method.
        Subclasses should override this method to provide model-specific generation logic.
        """
        if not self.model or not self.processor:
            raise RuntimeError("Model and processor must be loaded before generation.")

        inputs, input_len = self._convert_to_specific_chat_template(messages)
        outputs = self._generate_from_specific_chat_template(inputs, input_len, generation_config)

        processed_texts = []
        for output in outputs:
            decoded = self.processor.decode(output, skip_special_tokens=True)
            processed = self._post_process_output(decoded)
            processed_texts.append(processed)

        return processed_texts
    
    def _convert_to_specific_chat_template(self, messages: List[Dict[str, str]]) -> torch.Tensor:
        """Convert the messages to the specific chat template for the model."""
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            max_length=128000,
        ).to("cuda", dtype=torch.bfloat16)

        input_len = inputs["input_ids"].shape[-1]

        return inputs, input_len
    
    def _generate_from_specific_chat_template(self, inputs: torch.Tensor, input_len: int, generation_config: GenerationConfig) -> List[str]:
        """Generate from the specific chat template for the model."""

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=generation_config.max_tokens,
                do_sample=generation_config.do_sample,
                temperature=generation_config.temperature,
                top_p=generation_config.top_p,
                top_k=generation_config.top_k,
            )
            outputs = outputs[0][input_len:]
        return outputs
    
    def _post_process_output(self, decoded_text: str) -> str:
        """MedGemma-specific post-processing logic."""
        return decoded_text.split("assistant")[-1].strip().strip("`").strip('json')

ComponentFactory.register_model_manager_for_patterns(
    "llama", LlamaModelManager, LlamaModelManager._model_name_patterns
)
ComponentFactory.register_model_manager_for_patterns(
    "qwen3", Qwen3ModelManager, Qwen3ModelManager._model_name_patterns
)
ComponentFactory.register_model_manager_for_patterns(
    "qwq", QwQModelManager, QwQModelManager._model_name_patterns
)
ComponentFactory.register_model_manager_for_patterns(
    "medgemma", MedGemmaModelManager, MedGemmaModelManager._model_name_patterns
)

# Register some specific model names for better support
ComponentFactory.register_model_for_manager("meta-llama/Llama-3.1-8B-Instruct", "llama")
ComponentFactory.register_model_for_manager("unsloth/Llama-3.3-70B-Instruct-bnb-4bit", "llama")
ComponentFactory.register_model_for_manager("unsloth/Qwen3-30B-A3B-bnb-4bit", "qwen3")
ComponentFactory.register_model_for_manager("unsloth/QwQ-32B-unsloth-bnb-4bit", "qwq")
ComponentFactory.register_model_for_manager("Qwen/QwQ-32B", "qwq")
ComponentFactory.register_model_for_manager("unsloth/medgemma-27b-it", "medgemma")
ComponentFactory.register_model_for_manager("google/medgemma-27b-it", "medgemma")

