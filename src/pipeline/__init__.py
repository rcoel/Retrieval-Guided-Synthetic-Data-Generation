"""
src.pipeline — Core pipeline modules.

Modules:
    prompts     — All prompt templates (single source of truth)
    indexing    — FAISS semantic index with DP-noisy retrieval
    training    — LoRA/PEFT fine-tuning
    generation  — Adaptive agentic generation loop
    critic      — Standalone Critic Agent (CoT feedback)
"""

from .prompts import create_prompt, create_critic_prompt, create_red_team_prompt
from .indexing import SemanticIndexer, chunk_text

__all__ = [
    "create_prompt",
    "create_critic_prompt",
    "create_red_team_prompt",
    "SemanticIndexer",
    "chunk_text",
]
