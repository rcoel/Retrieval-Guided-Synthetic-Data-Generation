import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm
from .. import config
from ..utils import get_device

def create_prompt(private_example, context_docs):
    """Creates the structured prompt for generation."""
    context_str = "\n".join([f"{i+1}. {doc}" for i, doc in enumerate(context_docs)])
    
    prompt = f"""[SYSTEM] You are an AI assistant that generates a high-quality, semantically equivalent variant of a given text example. The new variant should retain the original's intent, meaning, and key information but should not be an exact copy.

[USER]
### CONTEXT FROM PUBLIC DOCUMENTS:
{context_str}

### TASK:
Generate a synthetic variant of the following example.

### ORIGINAL PRIVATE EXAMPLE:
{private_example}

[ASSISTANT]"""
    return prompt

class SyntheticDataGenerator:
    def __init__(self, base_model_name, lora_adapter_path):
        self.device = get_device()
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        self.model = PeftModel.from_pretrained(base_model, lora_adapter_path)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, private_data, public_passages, retriever):
        """Generates synthetic data for the entire private dataset."""
        synthetic_samples = []
        
        print("Generating synthetic data...")
        for example in tqdm(private_data, desc="Generating Samples"):
            private_text = example['text']
            
            retrieved_indices = retriever.retrieve([private_text], k=config.NUM_RETRIEVED_DOCS_K)[0]
            context_docs = [public_passages[idx] for idx in retrieved_indices]

            prompt = create_prompt(private_text, context_docs)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=config.MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=config.GENERATION_TEMP,
                    top_p=config.GENERATION_TOP_P,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            full_generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            assistant_response = full_generated_text.split("[ASSISTANT]")[-1].strip()

            synthetic_samples.append({
                "original_text": private_text,
                "synthetic_text": assistant_response,
                "label": example['label']
            })
            
        return synthetic_samples