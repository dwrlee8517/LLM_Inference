"""
Concrete implementation of the ModelManager for Hugging Face Transformers.
"""
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForImageTextToText, AutoProcessor
from typing import Tuple, Any, List, Dict, Type
import logging
import sys
from PIL import Image

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
        logger.debug(f"Initialized {self.__class__.__name__}")

    def load_model(self) -> Tuple[Any, Any]:
        """Load tokenizer and model."""
        logger.info(f"Loading model: {self.config.model_name}")
        logger.debug(f"Model config: {self.config}")
        os.environ["CUDA_VISIBLE_DEVICES"] = self.config.cuda_devices
        
        try:
            self.model, self.tokenizer = self._load_specific_model_and_tokenizer()
            logger.info("Model and tokenizer loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading model or tokenizer: {e}", exc_info=True)
            raise

        return self.model, self.tokenizer
        
    # Quantization removed

    # Quantization removed

    def _load_specific_model_and_tokenizer(self) -> Tuple[Any, Any]:
        """Default loading logic using Hugging Face. Subclasses should replace this method."""
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name, cache_dir=self.config.cache_dir
        )

        model_kwargs = {
            "pretrained_model_name_or_path": self.config.model_name,
            "cache_dir": self.config.cache_dir,
            "dtype": "auto",
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

        # (quantization removed)

        model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
        return model, tokenizer

    def generate(self, data_or_list, generation_config: GenerationConfig,) -> List[str]:
        """Generate responses for one or many raw items.

        Accepts either a single raw item (e.g., str or tuple[str, str]) or a list of such items.
        """
        logger.debug(f"Generating with config: {generation_config}")
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model and tokenizer must be loaded before generation.")

        items = data_or_list if isinstance(data_or_list, list) else [data_or_list]
        processed_texts: List[str] = []

        for item in items:
            inputs = self._convert_to_specific_chat_template(item)
            logger.debug(f"Inputs shape: {getattr(inputs, 'shape', None)}")
            outputs = self._generate_from_specific_chat_template(inputs, generation_config)
            logger.debug(f"Outputs shape: {getattr(outputs, 'shape', None)}")
            for output in outputs:
                decoded = self._decode_output(output)
                processed = self._post_process_output(decoded)
                processed_texts.append(processed)

        logger.debug(f"Processed texts count: {len(processed_texts)}")
        return processed_texts
    
    def _convert_text_to_messages(self, text: str|tuple[str, str]) -> List[Dict[str, str]]:
        """Convert the text to the specific chat template for the model."""
        if isinstance(text, str):   # text is a single string
            return [{"role": "user", "content": text}]
        elif isinstance(text, tuple):   # text is a tuple of (user_message, assistant_message)
            return [{"role": "system", "content": text[0]}, {"role": "user", "content": text[1]}]
        else:
            raise ValueError(f"Invalid text type: {type(text)}")
    
    def _verify_messages(self, messages: List[Dict[str, str]]) -> None:
        """Verify the messages are in the correct format."""
        if not messages:
            raise ValueError("Messages are empty.")
        if not isinstance(messages, list):   # messages is a list of dictionaries
            raise ValueError("Messages are not a list.")
        if not all(isinstance(message, dict) for message in messages):   # messages is a list of dictionaries
            raise ValueError("Messages are not a list of dictionaries.")
        if not all(message.keys() == {"role", "content"} for message in messages):   # messages is a list of dictionaries with "role" and "content" keys
            raise ValueError("Messages are not in the correct format.")
        if not all(message["role"] in {"system", "user", "assistant"} for message in messages):   # messages is a list of dictionaries with "role" in {"system", "user", "assistant"}
            raise ValueError("Messages are not in the correct format.")

    def _convert_to_specific_chat_template(self, prompt: str|tuple[str, str]) -> torch.Tensor:
        """Convert the messages to the specific chat template for the model."""
        messages = self._convert_text_to_messages(prompt)
        self._verify_messages(messages)
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to("cuda")
        return inputs
    
    def _generate_from_specific_chat_template(self, inputs: torch.Tensor, generation_config: GenerationConfig) -> List[str]:
        """Generate from the specific chat template for the model."""
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=generation_config.max_new_tokens,
            do_sample=generation_config.do_sample,
            temperature=generation_config.temperature,
            top_p=generation_config.top_p,
            top_k=generation_config.top_k,
        )
        return outputs
    

    def _decode_output(self, output: torch.Tensor) -> str:
        """Decode the output to a string."""
        return self.tokenizer.decode(output, skip_special_tokens=True)
    
    def _post_process_output(self, decoded_text: str) -> str:
        """Default post-processing logic. Subclasses can override."""
        return decoded_text.split("assistant")[-1].strip().strip("`").strip('json')

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "model_family": self.__class__.__name__,
            "model_name": self.config.model_name,
            "is_loaded": self.model is not None,
        }

    def cleanup(self) -> None:
        """Clean up model and tokenizer resources."""
        self.model = None
        self.tokenizer = None
        torch.cuda.empty_cache()
        logger.debug("Model resources cleaned up.")

class Llama3ModelManager(BaseHuggingFaceManager):
    """Model manager specifically for Meta's Llama models."""
    _model_name_patterns = [
        "meta-llama/Llama-3",
        "unsloth/Llama-3",
    ]

    def _generate_from_specific_chat_template(self, inputs: torch.Tensor, generation_config: GenerationConfig) -> List[str]:
        """Generate from the specific chat template for the model."""
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=generation_config.max_new_tokens,
            do_sample=generation_config.do_sample,
            temperature=generation_config.temperature,
            top_p=generation_config.top_p,
            top_k=generation_config.top_k,
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
    
    def _post_process_output(self, decoded_text: str) -> str:
        """Qwen3-specific post-processing logic."""
        return decoded_text.split("assistant")[-1].strip().strip("`").strip('json')
    
class QwQModelManager(BaseHuggingFaceManager):
    """Model manager specifically for QwQ models."""
    _model_name_patterns = [
        "Qwen/QwQ-",
        "unsloth/QwQ-",
    ]
    
    def _post_process_output(self, decoded_text: str) -> str:
        """Qwen3-specific post-processing logic."""
        processed_text = decoded_text.split("assistant")[-1].strip().strip("`").strip('json')
        return processed_text

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


    def _load_specific_model_and_tokenizer(self) -> Tuple[Any, Any]:
        """Load model and tokenizer using huggingface."""

        model_kwargs = {
            "pretrained_model_name_or_path": self.config.model_name,
            "cache_dir": self.config.cache_dir,
            "dtype": "auto",
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

        # (quantization removed)

        model = AutoModelForImageTextToText.from_pretrained(**model_kwargs)

        processor = AutoProcessor.from_pretrained(
            self.config.model_name,
            cache_dir=self.config.cache_dir,
        )

        return model, processor
    
    def generate(self, data_or_list, generation_config: GenerationConfig,) -> List[str]:
        """Generate responses for one or many raw items (supports multimodal)."""
        if not self.model or not self.processor:
            raise RuntimeError("Model and processor must be loaded before generation.")

        items = data_or_list if isinstance(data_or_list, list) else [data_or_list]
        processed_texts: List[str] = []

        for item in items:
            inputs, input_len = self._convert_to_specific_chat_template(item)
            outputs = self._generate_from_specific_chat_template(inputs, input_len, generation_config)
            for output in outputs:
                decoded = self.processor.decode(output, skip_special_tokens=True)
                processed = self._post_process_output(decoded)
                processed_texts.append(processed)

        return processed_texts
    
    def _convert_text_to_messages(self, data: str|tuple[str, str]|tuple[str, str | List[str|Image.Image]]) -> List[Dict[str, str]]:
        """Convert the text and image data (PIL.Image) to the specific chat template for the model."""
        if isinstance(data, str):   # data is a single string
            return [{"role": "user", "content": [{"type": "text", "text": data}]}]
        elif isinstance(data, tuple) and isinstance(data[1], list):
            messages = [{"role": "system", "content": [{"type": "text", "text": data[0]}]}, {"role": "user", "content": []}]
            for item in data[1]:
                if isinstance(item, str):
                    messages[1]["content"].append({"type": "text", "text": item})
                elif isinstance(item, Image.Image):
                    messages[1]["content"].append({"type": "image", "image": item})
                else:
                    raise ValueError(f"Invalid item type: {type(item)}")
            return messages
        elif isinstance(data, tuple):   # data is a tuple of (user_message, assistant_message)
            return [{"role": "system", "content": [{"type": "text", "text": data[0]}]}, {"role": "user", "content": [{"type": "text", "text": data[1]}]}]
        else:
            raise ValueError(f"Invalid text type: {type(data)}")

    def _verify_messages(self, messages: List[Dict[str, str]]) -> None:
        """Verify the messages are in the correct format for MedGemma."""
        if not messages:
            raise ValueError("Messages are empty.")
        if not isinstance(messages, list):   # messages is a list of dictionaries
            raise ValueError("Messages are not a list.")
        if not all(isinstance(message, dict) for message in messages):   # messages is a list of dictionaries
            raise ValueError("Messages are not a list of dictionaries.")
        if not all(message.keys() == {"role", "content"} for message in messages):   # messages is a list of dictionaries with "role" and "content" keys
            raise ValueError("Messages are not in the correct format.")
        if not all(message["role"] in {"system", "user", "assistant"} for message in messages):   # messages is a list of dictionaries with "role" in {"system", "user", "assistant"}
            raise ValueError("Messages are not in the correct format.")
        if not all(isinstance(message["content"], list) for message in messages):   # messages is a list of dictionaries with "content" as a list
            raise ValueError("Messages are not in the correct format.")
        if not all(isinstance(item, dict) for message in messages for item in message["content"]):   # messages is a list of dictionaries with "content" as a list of dictionaries
            raise ValueError("Messages are not in the correct format.")
        if not all(item.keys() == {"type", "text", "image"} for message in messages for item in message["content"]):   # messages is a list of dictionaries with "content" as a list of dictionaries with "type" in {"text", "image"}
            raise ValueError("Messages are not in the correct format.")
        if not all(item["type"] in {"text", "image"} for message in messages for item in message["content"]):   # messages is a list of dictionaries with "content" as a list of dictionaries with "type" in {"text", "image"}
            raise ValueError("Messages are not in the correct format.")
    
    def _convert_to_specific_chat_template(self, data: str|tuple[str, str]|tuple[str, str | List[str|Image.Image]]) -> torch.Tensor:
        """Convert the messages to the specific chat template for the model."""
        messages = self._convert_text_to_messages(data)
        self._verify_messages(messages)
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to("cuda")

        input_len = inputs["input_ids"].shape[-1]
        print(input_len) # debug

        return inputs, input_len
    
    def _generate_from_specific_chat_template(self, inputs: torch.Tensor, input_len: int, generation_config: GenerationConfig) -> List[str]:
        """Generate from the specific chat template for the model."""

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=generation_config.max_new_tokens,
                do_sample=generation_config.do_sample,
                temperature=generation_config.temperature,
                top_p=generation_config.top_p,
                top_k=generation_config.top_k,
            )
        
        # Slice the output to get only the generated tokens
        generated_tokens = outputs[:, input_len:]
        return generated_tokens
    
    def _post_process_output(self, decoded_text: str) -> str:
        """MedGemma-specific post-processing logic."""
        return decoded_text.strip()

class GptOssModelManager(BaseHuggingFaceManager):
    """Model manager for GPT-OSS text-only models."""
    _model_name_patterns = [
        "gpt-oss",
        "gptoss",
        "openai-community/gpt-4o-mini-oss",
    ]

    def _post_process_output(self, decoded_text: str) -> str:
        return decoded_text.split("assistant")[-1].strip().strip("`").strip("json")


ComponentFactory.register_model_manager_for_patterns(
    "llama", Llama3ModelManager, Llama3ModelManager._model_name_patterns
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
ComponentFactory.register_model_manager_for_patterns(
    "gptoss", GptOssModelManager, GptOssModelManager._model_name_patterns
)

# Register some specific model names for better support
ComponentFactory.register_model_for_manager("meta-llama/Llama-3.1-8B-Instruct", "llama")
ComponentFactory.register_model_for_manager("unsloth/Llama-3.3-70B-Instruct-bnb-4bit", "llama")
ComponentFactory.register_model_for_manager("unsloth/Qwen3-30B-A3B-bnb-4bit", "qwen3")
ComponentFactory.register_model_for_manager("unsloth/QwQ-32B-unsloth-bnb-4bit", "qwq")
ComponentFactory.register_model_for_manager("Qwen/QwQ-32B", "qwq")
ComponentFactory.register_model_for_manager("unsloth/medgemma-27b-it", "medgemma")
ComponentFactory.register_model_for_manager("unsloth/medgemma-27b-text-it-unsloth-bnb-4bit", "qwq")
ComponentFactory.register_model_for_manager("google/medgemma-27b-it", "medgemma")
ComponentFactory.register_model_for_manager("google/medgemma-27b-text-it", "qwq")
ComponentFactory.register_model_for_manager("openai/gpt-oss-20b", "gptoss")
ComponentFactory.register_model_for_manager("openai/gpt-oss-120b", "gptoss")




