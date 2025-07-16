"""
Concrete implementation of the PromptManager.
"""
from typing import List, Dict, Any, Callable
import logging

from core.interfaces import PromptManager, ComponentFactory
from core.config import InferenceConfig
import helpers.llm_prompts as prompts

logger = logging.getLogger(__name__)
class DefaultPromptManager(PromptManager):
    """Manages prompt templates and creation."""

    # Initialize the prompt manager with the templates
    def __init__(self, config: InferenceConfig):
        self.config = config

        # Make a dictionary of all the templates from the prompts module
        self._templates = {
            name: getattr(prompts, name)
            for name in dir(prompts)
            if name.startswith("prompt_") and callable(getattr(prompts, name))
        }
        logger.info(f"Initialized DefaultPromptManager with {len(self._templates)} templates.")

    # Get a prompt template function by name
    def get_prompt_template(self, template_name: str) -> Callable:
        """Get a prompt template function by name."""
        if template_name not in self._templates:
            raise ValueError(f"Prompt template '{template_name}' not found.")
        return self._templates[template_name]
    
    # Use the template function to create a specific prompt for the experiment
    def _create_specific_prompt(self, template_func: Callable, data: Dict[str, Any]) -> List[Dict[str, str]]:
        """Create a specific prompt for the model."""
        # The data from the dataloader is expected to be a tuple/list of lists
        # We need to extract the report lists to pass to the template function.
        input_data = data

        # Create the completed prompts
        completed_prompts = template_func(*input_data)

        # Ensure completed_prompts is a flat list of dictionaries.
        # It seems some template functions might return a list of lists of messages,
        # where each inner list corresponds to a batch item's messages.
        if completed_prompts and isinstance(completed_prompts[0], list):
            # Flatten the list of lists if necessary
            flattened_prompts = []
            for sublist in completed_prompts:
                flattened_prompts.extend(sublist)
            completed_prompts = flattened_prompts

        return completed_prompts

    def _convert_to_multimodal_format(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Converts string content to multimodal format for models like MedGemma."""
        converted_messages = []
        for message in messages:
            if isinstance(message.get("content"), str):
                converted_message = message.copy()
                converted_message["content"] = [{"type": "text", "text": message["content"]}]
                converted_messages.append(converted_message)
            else:
                converted_messages.append(message)
        return converted_messages

    # Create messages for the model
    def create_messages(self, template_name: str, data: Dict[str, Any]) -> List[Dict[str, str]]:
        """Create messages for the model using a batch of data."""
        template_func = self.get_prompt_template(template_name)
        completed_prompts = self._create_specific_prompt(template_func, data)

        # Convert the prompts to the multimodal format if the model is a multimodal model
        if self.config.multimodal:
            completed_prompts = self._convert_to_multimodal_format(completed_prompts)
        return completed_prompts

    def list_available_templates(self) -> List[str]:
        """List all available prompt templates."""
        return list(self._templates.keys())

    def validate_template(self, template_name: str) -> bool:
        """Validate that a template exists."""
        return template_name in self._templates

class RadPathPromptManager(DefaultPromptManager):
    """Manages prompt templates and creation."""
    # Use the template function to create a specific prompt for the experiment
    def _create_specific_prompt(self, template_func: Callable, data: Dict[str, Any]) -> List[Dict[str, str]]:
        """Create a specific prompt for the model."""
        # The data from the dataloader is expected to be a tuple/list of lists
        # We need to extract the report lists to pass to the template function.
        radreports_batch = data[1]  
        bxreports_batch = data[2]

        # Pass the report lists to the template function
        input_data = (bxreports_batch, radreports_batch)

        # Create the completed prompts
        completed_prompts = template_func(*input_data)

        # Ensure completed_prompts is a flat list of dictionaries.
        if completed_prompts and isinstance(completed_prompts[0], list):
            # Flatten the list of lists if necessary
            flattened_prompts = []
            for sublist in completed_prompts:
                flattened_prompts.extend(sublist)
            completed_prompts = flattened_prompts

        return completed_prompts

# Register the prompt manager with the factory
ComponentFactory.register_prompt_manager("RadPath", RadPathPromptManager)