"""
Shared model loading utilities.

Eliminates duplicated BitsAndBytesConfig blocks across generation.py,
training.py, and red_team.py. Provides a single factory function
for loading quantized causal language models.
"""

from __future__ import annotations
import torch
from typing import Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from .utils import get_device


def get_quantization_config() -> Optional[BitsAndBytesConfig]:
    """
    Returns BitsAndBytesConfig for 4-bit quantization if CUDA is available.
    
    Returns:
        BitsAndBytesConfig or None (CPU fallback).
    """
    if torch.cuda.is_available():
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    return None


def load_causal_model(
    model_name: str,
    quantize: bool = True,
    trust_remote_code: bool = True,
) -> AutoModelForCausalLM:
    """
    Load a causal language model with optional quantization.
    
    Args:
        model_name: HuggingFace model identifier.
        quantize: Whether to apply 4-bit quantization (requires CUDA).
        trust_remote_code: Allow custom model code from HuggingFace.
        
    Returns:
        Loaded AutoModelForCausalLM.
    """
    quant_config = get_quantization_config() if quantize else None
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=trust_remote_code,
    )
    return model


def load_tokenizer(
    model_name: str,
    trust_remote_code: bool = True,
    set_pad_token: bool = True,
) -> AutoTokenizer:
    """
    Load a tokenizer with sensible defaults.
    
    Args:
        model_name: HuggingFace model identifier.
        trust_remote_code: Allow custom tokenizer code.
        set_pad_token: Set pad_token to eos_token if not set.
        
    Returns:
        Configured AutoTokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=trust_remote_code
    )
    if set_pad_token and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model_and_tokenizer(
    model_name: str,
    quantize: bool = True,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Convenience function to load both model and tokenizer.
    
    Args:
        model_name: HuggingFace model identifier.
        quantize: Whether to apply 4-bit quantization.
        
    Returns:
        Tuple of (model, tokenizer).
    """
    model = load_causal_model(model_name, quantize=quantize)
    tokenizer = load_tokenizer(model_name)
    return model, tokenizer
