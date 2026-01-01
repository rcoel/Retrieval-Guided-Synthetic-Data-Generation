import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .. import config
from ..utils import get_device

class PrivacyAttacker:
    """
    Adversarial 'Red Team' module that attempts to infer private information 
    from synthetic samples. If it succeeds, the sample is deemed unsafe.
    """
    def __init__(self, model_name=None):
        self.device = get_device()
        # Use a flexible model, default to the one in config or a small dedicated one
        self.model_name = model_name if model_name else config.BASE_LLM_MODEL
        print(f"Loading Red Team Attacker model: {self.model_name}...")
        
        # In a real scenario, this might be a different, more powerful model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, 
            device_map="auto", 
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        self.model.eval()

    def attack(self, synthetic_text, original_label=None):
        """
        Attempts to guess the original label or sensitive entity from the synthetic text.
        Returns (bool, str): (True if privacy is compromised/attack successful, reasoning)
        """
        # Scenario: We want to see if the model can guess the specific "label" or topic
        # simply from the rewritten text, which might imply it's too specific.
        # OR: extracting PII. 
        
        # For this project's context (often text classification data like SST2 or medical),
        # an attack might be: "Can you recover the specific entity name?" 
        # Since we don't have the entity list hardcoded, we'll try a generic PII extration attack.
        
        prompt = f"""[SYSTEM] You are a privacy auditor. Your goal is to identify if the following text contains specific, sensitive real-world entities that were likely just copied from a private source, or if it has successfully generalized them.

[USER]
Text: "{synthetic_text}"

Does this text contain any specific proper nouns (names, specific locations, organizations) that look like they might be real private data rather than generic placeholders? 
Answer "YES" or "NO" and explain.
[ASSISTANT]
"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=50, 
                temperature=0.1 # Low temp for deterministic auditing
            )
            
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_tail = response.split("[ASSISTANT]")[-1].strip().upper()
        
        # Heuristic: If the attacker says YES, it found something suspicious.
        is_compromised = "YES" in response_tail[:10] # Look at the start of the answer
        
        return is_compromised, response_tail
