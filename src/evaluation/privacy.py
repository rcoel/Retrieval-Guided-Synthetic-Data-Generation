"""
Privacy Metrics for Synthetic Data Evaluation.

Provides:
- Exact Match Ratio: percentage of verbatim copies
- N-gram Overlap: percentage of shared n-grams between original and synthetic
- Single-pair overlap: used in the generation feedback loop
"""

from __future__ import annotations
from typing import List
from nltk import ngrams
from nltk.tokenize import word_tokenize
from tqdm import tqdm
from ..logger import get_logger

logger = get_logger("rag_pipeline.privacy")


def calculate_exact_match_ratio(
    original_texts: List[str],
    synthetic_texts: List[str],
) -> float:
    """
    Calculates the percentage of synthetic texts that are exact copies.
    
    Args:
        original_texts: Original texts.
        synthetic_texts: Synthetic texts.
        
    Returns:
        Exact match percentage (0.0 to 100.0).
    """
    if not synthetic_texts:
        return 0.0
    original_set = set(o.strip() for o in original_texts)
    match_count = sum(1 for s in synthetic_texts if s.strip() in original_set)
    ratio = (match_count / len(synthetic_texts)) * 100
    logger.info(f"Exact Match Ratio: {ratio:.2f}% ({match_count}/{len(synthetic_texts)})")
    return ratio


def calculate_ngram_overlap(
    original_texts: List[str],
    synthetic_texts: List[str],
    n: int = 5,
) -> float:
    """
    Calculates the percentage of n-grams in synthetic data that appear in original.
    
    Lower overlap = better privacy preservation.
    
    Args:
        original_texts: Original texts.
        synthetic_texts: Synthetic texts.
        n: N-gram size (default 5).
        
    Returns:
        Overlap percentage (0.0 to 100.0).
    """
    original_ngrams = set()
    for text in tqdm(original_texts, desc=f"Extracting {n}-grams", leave=False):
        tokens = word_tokenize(text)
        original_ngrams.update(ngrams(tokens, n))
    
    if not original_ngrams:
        return 0.0

    synthetic_ngrams_count = 0
    overlap_count = 0
    for text in tqdm(synthetic_texts, desc=f"Analyzing {n}-gram overlap", leave=False):
        tokens = word_tokenize(text)
        current_ngrams = list(ngrams(tokens, n))
        synthetic_ngrams_count += len(current_ngrams)
        overlap_count += sum(1 for ng in current_ngrams if ng in original_ngrams)
    
    if synthetic_ngrams_count == 0:
        return 0.0
    
    ratio = (overlap_count / synthetic_ngrams_count) * 100
    logger.info(f"{n}-gram Overlap: {ratio:.2f}% ({overlap_count}/{synthetic_ngrams_count})")
    return ratio


def calculate_single_pair_ngram_overlap(
    original_text: str,
    synthetic_text: str,
    n: int = 5,
) -> float:
    """
    Calculates n-gram overlap between a single pair (used in generation loop).
    
    Args:
        original_text: Original text.
        synthetic_text: Synthetic text.
        n: N-gram size.
        
    Returns:
        Overlap ratio (0.0 to 1.0).
    """
    tokens_orig = word_tokenize(original_text.lower())
    tokens_synth = word_tokenize(synthetic_text.lower())
    
    grams_orig = set(ngrams(tokens_orig, n))
    grams_synth = list(ngrams(tokens_synth, n))
    
    if not grams_orig or not grams_synth:
        return 0.0
        
    overlap_count = sum(1 for g in grams_synth if g in grams_orig)
    return overlap_count / len(grams_synth)