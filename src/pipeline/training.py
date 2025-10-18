import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from .. import config
from .generation import create_prompt # Re-use prompt creation logic

def create_training_dataset(private_dataset, public_passages, indexer):
    """Creates the training dataset by retrieving context and formatting prompts."""
    training_data = []
    for example in tqdm(private_dataset, desc="Creating Training Dataset"):
        private_text = example['text']
        retrieved_indices = indexer.retrieve([private_text], k=config.NUM_RETRIEVED_DOCS_K)[0]
        context_docs = [public_passages[idx] for idx in retrieved_indices]
        
        # The prompt is the input, and the model learns to predict it
        # In Causal LM, the labels are the inputs shifted by one token.
        prompt = create_prompt(private_text, context_docs)
        training_data.append({"text": prompt})
        
    return Dataset.from_list(training_data)


class LoRAFineTuner:
    def __init__(self, base_model_name):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Quantization config for memory efficiency
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map="auto", # Automatically uses available GPUs
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def setup_peft(self):
        """Sets up the PEFT configuration for LoRA."""
        self.model = prepare_model_for_kbit_training(self.model)
        
        lora_config = LoraConfig(
            r=config.LORA_R,
            lora_alpha=config.LORA_ALPHA,
            target_modules=config.LORA_TARGET_MODULES,
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

    def train(self, training_dataset):
        """Runs the fine-tuning process."""
        
        def tokenize_function(examples):
            return self.tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

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
            fp16=True, # Use mixed precision
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            tokenizer=self.tokenizer
        )

        print("Starting fine-tuning...")
        trainer.train()
        print("Fine-tuning complete.")
        
        self.model.save_pretrained(config.ADAPTER_PATH)
        print(f"LoRA adapter saved to {config.ADAPTER_PATH}")