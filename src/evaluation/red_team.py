"""
Adversarial Red Team Privacy Attacker.

Probes synthetic text for PII leakage by using an LLM to attempt
entity extraction. If the attacker can identify specific private
entities, the sample is deemed compromised and rejected.

Structural improvement: uses shared model_loader and centralized prompts.
"""

from __future__ import annotations
import torch
from typing import Tuple
from .. import config
from ..utils import get_device
from ..logger import get_logger
from ..model_loader import load_causal_model, load_tokenizer
from ..pipeline.prompts import create_red_team_prompt

logger = get_logger("rag_pipeline.red_team")


class PrivacyAttacker:
    """
    Adversarial 'Red Team' module that attempts to infer private information
    from synthetic samples. If it succeeds, the sample is deemed unsafe.
    """

    def __init__(self, model_name: str | None = None):
        self.device = get_device()
        self.model_name = model_name or config.BASE_LLM_MODEL
        
        logger.info(f"Loading Red Team Attacker model: {self.model_name}")
        
        # Use shared model loader (eliminates duplicate BitsAndBytesConfig)
        self.model = load_causal_model(self.model_name, quantize=False)
        self.tokenizer = load_tokenizer(self.model_name)
        self.model.eval()

    def attack(self, synthetic_text: str, original_label: int | None = None) -> Tuple[bool, str]:
        """
        Attempts to extract private information from synthetic text.
        
        Args:
            synthetic_text: The synthetic text to audit.
            original_label: Optional original label (unused, for interface consistency).
            
        Returns:
            Tuple of (is_compromised, reasoning_text).
        """
        prompt = create_red_team_prompt(synthetic_text)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.1,  # Low temp for deterministic auditing
            )
            
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_tail = response.split("[ASSISTANT]")[-1].strip().upper()
        
        is_compromised = "YES" in response_tail[:10]
        
        if is_compromised:
            logger.warning(f"Red Team ATTACK SUCCESSFUL: {response_tail[:100]}")
        
        return is_compromised, response_tail
