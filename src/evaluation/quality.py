"""
Quality Metrics for Synthetic Data Evaluation.

Provides:
- Self-BLEU: diversity measurement (lower = more diverse)
- Type-Token Ratio (TTR): lexical richness
- Semantic Similarity: meaning preservation vs original
"""

from __future__ import annotations
import torch
from typing import List, Optional
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import evaluate
from .. import config
from ..logger import get_logger

logger = get_logger("rag_pipeline.quality")


def calculate_self_bleu(texts: List[str]) -> float:
    """
    Calculates Self-BLEU score for diversity.
    
    Lower Self-BLEU = more diverse outputs (less mode collapse).
    
    Args:
        texts: List of generated texts.
        
    Returns:
        Average Self-BLEU score (0.0 to 1.0).
    """
    if len(texts) < 2:
        logger.warning("Self-BLEU requires ≥2 texts, returning 0.0")
        return 0.0
    
    bleu = evaluate.load("bleu")
    total_bleu = 0.0
    
    for i in tqdm(range(len(texts)), desc="Self-BLEU", leave=False):
        hypothesis = texts[i]
        references = [texts[j] for j in range(len(texts)) if i != j]
        results = bleu.compute(predictions=[hypothesis], references=[references])
        total_bleu += results['bleu']
        
    score = total_bleu / len(texts)
    logger.info(f"Self-BLEU: {score:.4f}")
    return score


def calculate_ttr(texts: List[str]) -> float:
    """
    Calculates Type-Token Ratio (TTR) for lexical richness.
    
    Higher TTR = richer vocabulary in the synthetic dataset.
    
    Args:
        texts: List of generated texts.
        
    Returns:
        TTR score (0.0 to 1.0).
    """
    all_tokens = [token for text in texts for token in word_tokenize(text.lower())]
    if not all_tokens:
        return 0.0
    score = len(set(all_tokens)) / len(all_tokens)
    logger.info(f"TTR: {score:.4f} ({len(set(all_tokens))} unique / {len(all_tokens)} total)")
    return score


def calculate_semantic_similarity(
    original_texts: List[str],
    synthetic_texts: List[str],
    model: Optional[SentenceTransformer] = None,
) -> float:
    """
    Measures cosine similarity between original and synthetic embeddings.
    
    Higher value = better meaning preservation.
    
    Args:
        original_texts: Original texts.
        synthetic_texts: Generated synthetic texts.
        model: Optional pre-loaded SentenceTransformer.
        
    Returns:
        Mean cosine similarity (0.0 to 1.0).
    """
    if model is None:
        model = SentenceTransformer(config.EMBEDDING_MODEL)
    
    original_embeddings = model.encode(original_texts, convert_to_tensor=True)
    synthetic_embeddings = model.encode(synthetic_texts, convert_to_tensor=True)
    
    cosine_scores = util.cos_sim(original_embeddings, synthetic_embeddings)
    score = torch.diag(cosine_scores).mean().item()
    logger.info(f"Semantic Similarity: {score:.4f}")
    return score


def measure_similarity_batch(
    original_texts: List[str],
    synthetic_texts: List[str],
    model: SentenceTransformer,
) -> List[float]:
    """
    Returns a list of similarity scores for each pair in the batch.
    
    Args:
        original_texts: List of original texts.
        synthetic_texts: List of synthetic texts.
        model: Pre-loaded SentenceTransformer.
        
    Returns:
        List of cosine similarity scores, one per pair.
    """
    original_embeddings = model.encode(original_texts, convert_to_tensor=True)
    synthetic_embeddings = model.encode(synthetic_texts, convert_to_tensor=True)
    
    cosine_scores = util.pairwise_cos_sim(original_embeddings, synthetic_embeddings)
    return cosine_scores.cpu().tolist()