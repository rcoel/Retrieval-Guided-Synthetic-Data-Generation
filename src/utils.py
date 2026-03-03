"""
Utility functions for the RAG pipeline.

Provides device selection, data I/O, NLTK setup, and reproducibility.
"""

from __future__ import annotations
import torch
import json
import random
import nltk
import numpy as np
from typing import List, Dict, Any
from .logger import get_logger

logger = get_logger("rag_pipeline.utils")


def get_device() -> str:
    """Returns the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_seed(seed: int = 42) -> None:
    """
    Set seeds for reproducibility across all libraries.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to {seed}")


def save_synthetic_data(data: List[Dict[str, Any]], path: str) -> None:
    """
    Saves generated synthetic data to a JSONL file.
    
    Args:
        data: List of sample dictionaries.
        path: Output file path.
    """
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, default=str) + '\n')
    logger.info(f"Synthetic data saved to {path} ({len(data)} samples)")


def load_synthetic_data(path: str) -> List[Dict[str, Any]]:
    """
    Loads synthetic data from a JSONL file.
    
    Args:
        path: Input file path.
        
    Returns:
        List of sample dictionaries.
    """
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    logger.info(f"Loaded {len(data)} samples from {path}")
    return data


def setup_nltk() -> None:
    """Downloads required NLTK data if not present."""
    for resource in ['tokenizers/punkt', 'tokenizers/punkt_tab']:
        try:
            nltk.data.find(resource)
        except LookupError:
            resource_name = resource.split('/')[-1]
            logger.info(f"Downloading NLTK '{resource_name}'...")
            nltk.download(resource_name, quiet=True)