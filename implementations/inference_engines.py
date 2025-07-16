"""
Concrete implementation of the InferenceEngine.
"""
import time
import logging
from typing import List, Dict, Any
from torch.utils.data import DataLoader

from core.interfaces import InferenceEngine, ComponentFactory, ModelManager, PromptManager
from core.config import InferenceConfig

logger = logging.getLogger(__name__)

class DefaultInferenceEngine(InferenceEngine):
    """
    Default inference engine with 
    """

    def __init__(self, config: InferenceConfig, model_manager: ModelManager, prompt_manager: PromptManager):
        self.config = config
        self.model_manager = model_manager
        self.prompt_manager = prompt_manager
        self.tokenizer = None
        self.model = None
        self.stats = {"batches_processed": 0, "total_time": 0}
        logger.info("Initialized DefaultInferenceEngine")

    def setup(self):
        """Load the model and tokenizer."""
        self.tokenizer, self.model = self.model_manager.load_model()

    def process_dataset(self, dataset: Any) -> Dict[str, Any]:
        """Process an entire dataset."""
        if not self.model or not self.tokenizer:
            self.setup()

        dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=self.config.shuffle)
        
        results = {}
        start_time = time.time()

        for batch in dataloader:
            mrns_batch = batch[0]
            
            # Create prompts
            messages = self.prompt_manager.create_messages(
                self.config.prompt_template,
                batch
            )

            # Generate text
            generated_texts = self.model_manager.generate(
                messages,
                self.config.generation
            )
            
            for mrn, gen_text in zip(mrns_batch, generated_texts):
                results[mrn] = gen_text
            
            self.stats["batches_processed"] += 1

        end_time = time.time()
        self.stats["total_time"] += (end_time - start_time)
        
        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return self.stats

    # Note: process_batch and process_single are not implemented as
    # this engine works on a full dataset. They could be implemented
    # if a more granular processing API is needed.
    def process_batch(self, batch: List[Any]) -> List[Any]:
        raise NotImplementedError("process_batch is not implemented in this engine.")

    def process_single(self, request: Any) -> Any:
        raise NotImplementedError("process_single is not implemented in this engine.")

# Register the inference engine with the factory
ComponentFactory.register_inference_engine("default", DefaultInferenceEngine) 