"""
Downstream Task Evaluation.

Trains a BERT classifier on synthetic data and evaluates
on original data to measure utility preservation.
"""

from __future__ import annotations
import numpy as np
from typing import List, Dict, Any
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset
from sklearn.metrics import accuracy_score
from .. import config
from ..logger import get_logger

logger = get_logger("rag_pipeline.downstream")


def compute_metrics(p) -> Dict[str, float]:
    """Compute accuracy from trainer predictions."""
    preds = np.argmax(p.predictions, axis=1)
    return {"accuracy": accuracy_score(p.label_ids, preds)}


def train_and_evaluate_classifier(
    train_texts: List[str],
    train_labels: List[int],
    test_texts: List[str],
    test_labels: List[int],
) -> Dict[str, Any]:
    """
    Fine-tunes and evaluates a BERT-base classifier.
    
    Trains on synthetic data, evaluates on original data.
    This measures how well synthetic data preserves downstream utility.
    
    Args:
        train_texts: Synthetic texts for training.
        train_labels: Labels for training data.
        test_texts: Original texts for evaluation.
        test_labels: Labels for evaluation data.
        
    Returns:
        Dictionary with 'eval_accuracy' and other Trainer metrics.
    """
    logger.info(f"Training classifier: {len(train_texts)} train, {len(test_texts)} test")
    
    train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
    test_dataset = Dataset.from_dict({"text": test_texts, "label": test_labels})

    tokenizer = AutoTokenizer.from_pretrained(config.CLASSIFIER_MODEL)
    num_labels = max(max(train_labels), max(test_labels)) + 1
    model = AutoModelForSequenceClassification.from_pretrained(
        config.CLASSIFIER_MODEL, num_labels=num_labels
    )

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)

    training_args = TrainingArguments(
        output_dir=config.MODEL_OUTPUT_DIR + "classifier",
        num_train_epochs=3,
        per_device_train_batch_size=config.EVAL_BATCH_SIZE,
        per_device_eval_batch_size=config.EVAL_BATCH_SIZE,
        evaluation_strategy="epoch",
        logging_dir='./logs',
        logging_steps=50,
        load_best_model_at_end=True,
        save_strategy="epoch",
        metric_for_best_model="accuracy",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    results = trainer.evaluate()
    
    logger.info(f"Downstream accuracy: {results.get('eval_accuracy', 'N/A')}")
    return results