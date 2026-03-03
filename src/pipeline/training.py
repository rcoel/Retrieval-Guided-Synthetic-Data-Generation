"""
LoRA Fine-Tuning Module.

Uses PEFT (Parameter-Efficient Fine-Tuning) with LoRA to adapt a
causal language model on private data with retrieved public context.

Structural improvement: imports prompts from prompts.py (no circular import)
and uses shared model_loader (no duplicate BitsAndBytesConfig).
"""

from __future__ import annotations
import torch
from typing import List
from datasets import Dataset
from transformers import TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from .. import config
from ..logger import get_logger
from ..model_loader import load_causal_model, load_tokenizer
from .prompts import create_prompt  # ← Now imported from prompts.py (no circular import)

logger = get_logger("rag_pipeline.training")


def create_training_dataset(private_dataset, public_passages: List[str], indexer) -> Dataset:
    """
    Creates the training dataset by retrieving context and formatting prompts.
    
    Args:
        private_dataset: HuggingFace Dataset with 'text' column.
        public_passages: List of public text passages.
        indexer: SemanticIndexer for retrieval.
        
    Returns:
        HuggingFace Dataset with 'text' column containing formatted prompts.
    """
    training_data = []
    for example in private_dataset:
        private_text = example['text']
        retrieved_indices = indexer.retrieve([private_text], k=config.NUM_RETRIEVED_DOCS_K)[0]
        context_docs = [public_passages[idx] for idx in retrieved_indices]
        
        prompt = create_prompt(private_text, context_docs)
        training_data.append({"text": prompt})
        
    logger.info(f"Created training dataset with {len(training_data)} examples")
    return Dataset.from_list(training_data)


class LoRAFineTuner:
    """Handles LoRA-based fine-tuning of causal language models."""

    def __init__(self, base_model_name: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Use shared model loader (eliminates duplicate BitsAndBytesConfig)
        self.model = load_causal_model(base_model_name, quantize=True)
        self.tokenizer = load_tokenizer(base_model_name)
        
        logger.info(f"Model loaded: {base_model_name}")

    def setup_peft(self) -> None:
        """Sets up the PEFT configuration for LoRA."""
        self.model = prepare_model_for_kbit_training(self.model)
        
        lora_config = LoraConfig(
            r=config.LORA_R,
            lora_alpha=config.LORA_ALPHA,
            target_modules=config.LORA_TARGET_MODULES,
            lora_dropout=config.LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        logger.info("PEFT/LoRA configured")

    def train(self, training_dataset: Dataset) -> None:
        """
        Runs the fine-tuning process.
        
        Args:
            training_dataset: Dataset with 'text' column containing prompts.
        """
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"], truncation=True,
                padding="max_length", max_length=512,
            )

        tokenized_dataset = training_dataset.map(tokenize_function, batched=True)

        training_args = TrainingArguments(
            output_dir=config.MODEL_OUTPUT_DIR,
            num_train_epochs=config.NUM_EPOCHS,
            per_device_train_batch_size=config.BATCH_SIZE,
            gradient_accumulation_steps=4,
            learning_rate=config.LEARNING_RATE,
            lr_scheduler_type="cosine",
            warmup_steps=config.WARMUP_STEPS,
            logging_dir='./logs',
            logging_steps=10,
            save_strategy="epoch",
            fp16=torch.cuda.is_available(),
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            tokenizer=self.tokenizer,
        )

        logger.info("Starting fine-tuning...")
        trainer.train()
        logger.info("Fine-tuning complete")
        
        self.model.save_pretrained(config.ADAPTER_PATH)
        logger.info(f"LoRA adapter saved to {config.ADAPTER_PATH}")