import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments
)
from datasets import Dataset
from sklearn.metrics import accuracy_score
from .. import config

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return {"accuracy": accuracy_score(p.label_ids, preds)}

def train_and_evaluate_classifier(train_texts, train_labels, test_texts, test_labels):
    """Fine-tunes and evaluates a BERT-base classifier."""
    
    train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
    test_dataset = Dataset.from_dict({"text": test_texts, "label": test_labels})

    tokenizer = AutoTokenizer.from_pretrained(config.CLASSIFIER_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(config.CLASSIFIER_MODEL, num_labels=max(train_labels) + 1)

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
    
    return results