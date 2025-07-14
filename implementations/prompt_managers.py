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

    def __init__(self, config: InferenceConfig):
        self.config = config
        self._templates = {
            name: getattr(prompts, name)
            for name in dir(prompts)
            if name.startswith("prompt_") and callable(getattr(prompts, name))
        }
        logger.info(f"Initialized DefaultPromptManager with {len(self._templates)} templates.")

    def get_prompt_template(self, template_name: str) -> Callable:
        """Get a prompt template function by name."""
        if template_name not in self._templates:
            raise ValueError(f"Prompt template '{template_name}' not found.")
        return self._templates[template_name]

    def create_messages(
        self,
        template_name: str,
        data: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """Create messages for the model using a batch of data."""
        template_func = self.get_prompt_template(template_name)
        
        # The data from the dataloader is expected to be a tuple/list of lists
        # (mrns_batch, radreports_batch, bxreports_batch)
        # We need to extract the report lists to pass to the template function.
        radreports_batch = data[1]
        bxreports_batch = data[2]

        return template_func(bxreports_batch, radreports_batch)

    def list_available_templates(self) -> List[str]:
        """List all available prompt templates."""
        return list(self._templates.keys())

    def validate_template(self, template_name: str) -> bool:
        """Validate that a template exists."""
        return template_name in self._templates

# Register the prompt manager with the factory
ComponentFactory.register_prompt_manager("default", DefaultPromptManager) 