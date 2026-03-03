"""
Critic Agent — Extracted from generation.py for separation of concerns.

Responsible for analyzing failed synthetic samples and producing
structured Chain-of-Thought feedback for the Generator to refine.
"""

from __future__ import annotations
import torch
from typing import Tuple
from ..model_loader import load_causal_model, load_tokenizer
from ..logger import get_logger
from .prompts import create_critic_prompt

logger = get_logger("rag_pipeline.critic")


class CriticAgent:
    """
    Analyzes failed synthetic samples and produces structured feedback.
    
    The Critic uses Chain-of-Thought prompting to:
    1. Identify problematic spans
    2. Explain the failure
    3. Provide specific fix instructions as JSON
    """

    def __init__(self, model, tokenizer, device: str):
        """
        Initialize with a pre-loaded model and tokenizer.
        
        Using pre-loaded model avoids redundant loading when the Generator
        and Critic share the same model instance.
        
        Args:
            model: A loaded causal language model.
            tokenizer: The corresponding tokenizer.
            device: Compute device string.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        logger.info("Critic Agent initialized")

    def critique(
        self, 
        original_text: str, 
        generated_text: str, 
        issue_type: str,
        max_new_tokens: int = 60,
    ) -> str:
        """
        Generate structured critique for a failed sample.
        
        Args:
            original_text: The original private text.
            generated_text: The synthetic text that failed checks.
            issue_type: Description of the detected issue.
            max_new_tokens: Maximum tokens for the critique response.
            
        Returns:
            Natural language critique with fix instructions.
        """
        critic_prompt = create_critic_prompt(original_text, generated_text, issue_type)
        inputs = self.tokenizer(critic_prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        
        critique_text = self.tokenizer.decode(
            outputs[0], skip_special_tokens=True
        ).split("[ASSISTANT]")[-1].strip()
        
        logger.debug(f"Critique generated for issue '{issue_type}': {critique_text[:80]}...")
        return critique_text

    def generate_coherence_feedback(self, perplexity: float) -> str:
        """
        Generate feedback for coherence (perplexity) failures.
        
        This doesn't require model inference — it's a templated response
        since the issue is clear (incoherent text).
        
        Args:
            perplexity: The measured perplexity score.
            
        Returns:
            Feedback string for the Generator.
        """
        return (
            f"The generated text is incoherent (perplexity={perplexity:.1f}). "
            "Produce a more fluent, grammatically correct variant that closely "
            "follows the structure of the original."
        )

    def generate_red_team_feedback(self, reasoning: str) -> str:
        """
        Generate feedback for Red Team failures.
        
        Args:
            reasoning: The Red Team's explanation of the leak.
            
        Returns:
            Feedback string for the Generator.
        """
        return (
            f"Privacy Leak Detected by Red Team: {reasoning}. "
            "Obfuscate entities better — replace proper nouns with generic terms."
        )
