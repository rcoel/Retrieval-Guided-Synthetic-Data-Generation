"""
Prompt templates for the Adaptive RAG pipeline.

Extracted from generation.py to break the circular import
between training.py and generation.py, and to provide a
single source of truth for all prompt engineering.
"""

from __future__ import annotations
from typing import List, Optional


def create_prompt(
    private_example: str, 
    context_docs: List[str], 
    feedback: Optional[str] = None
) -> str:
    """
    Creates the structured prompt for generation, optionally 
    including critique feedback from a previous failed attempt.
    
    Args:
        private_example: The original private text to rewrite.
        context_docs: Retrieved public documents for context.
        feedback: Optional critique from the Critic Agent.
        
    Returns:
        Formatted prompt string.
    """
    context_str = "\n".join([f"{i+1}. {doc}" for i, doc in enumerate(context_docs)])
    
    base_instruction = "Generate a synthetic variant of the following example."
    if feedback:
        base_instruction += (
            f"\n\nCRITIQUE FROM PREVIOUS ATTEMPT:\n{feedback}\n\n"
            "FIX INSTRUCTION: Apply the feedback above to fix the issue."
        )

    prompt = f"""[SYSTEM] You are an AI assistant that generates a high-quality, semantically equivalent variant of a given text example. The new variant should retain the original's intent, meaning, and key information but should not be an exact copy.

[USER]
### CONTEXT FROM PUBLIC DOCUMENTS:
{context_str}

### TASK:
{base_instruction}

### ORIGINAL PRIVATE EXAMPLE:
{private_example}

[ASSISTANT]"""
    return prompt


def create_critic_prompt(
    original_text: str, 
    generated_text: str, 
    issue_type: str
) -> str:
    """
    Chain-of-Thought Critic prompt that produces structured JSON feedback.
    
    The Critic is instructed to:
    1. IDENTIFY problematic spans
    2. EXPLAIN why the issue occurred
    3. PROVIDE specific fix instructions
    
    Args:
        original_text: The original private text.
        generated_text: The synthetic text that failed quality checks.
        issue_type: Description of the detected issue.
        
    Returns:
        Formatted critic prompt string.
    """
    prompt = f"""[SYSTEM] You are a strict Data Privacy and Quality Assurance Critic. You MUST respond with a structured analysis.

[USER]
Original Text: "{original_text}"
Generated Text: "{generated_text}"

Issue Detected: {issue_type}

Task: Analyze this issue step-by-step:
1. IDENTIFY the specific problematic spans in the generated text
2. EXPLAIN why this is a privacy/quality issue
3. PROVIDE a concrete fix instruction

Respond in this JSON format:
{{"issue": "<issue type>", "severity": "<low|medium|high>", "problematic_spans": ["<span1>", "<span2>"], "fix_instruction": "<specific instruction>"}}

[ASSISTANT]"""
    return prompt


def create_red_team_prompt(synthetic_text: str) -> str:
    """
    Creates the Red Team (Privacy Attacker) prompt.
    
    Args:
        synthetic_text: The synthetic text to audit for privacy leaks.
        
    Returns:
        Formatted red team audit prompt.
    """
    prompt = f"""[SYSTEM] You are a privacy auditor. Your goal is to identify if the following text contains specific, sensitive real-world entities that were likely just copied from a private source, or if it has successfully generalized them.

[USER]
Text: "{synthetic_text}"

Does this text contain any specific proper nouns (names, specific locations, organizations) that look like they might be real private data rather than generic placeholders? 
Answer "YES" or "NO" and explain.
[ASSISTANT]
"""
    return prompt
