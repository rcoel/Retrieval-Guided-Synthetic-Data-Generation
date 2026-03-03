"""
Dataset Registry — Central registry for all supported datasets.

Provides auto-download, column mapping, and domain-matched public
corpus generation for multi-dataset experiments.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import os
import pandas as pd
from datasets import load_dataset, Dataset
from .logger import get_logger

logger = get_logger("rag_pipeline.registry")


@dataclass
class DatasetSpec:
    """Specification for a supported dataset."""
    source: str                    # HuggingFace dataset identifier
    subset: Optional[str]          # Optional subset name
    text_column: str               # Column containing text
    label_column: str              # Column containing labels
    task_type: str                 # "sentiment", "topic", "nli", etc.
    num_labels: int                # Number of output classes
    splits: List[str] = field(default_factory=lambda: ["train", "validation"])
    description: str = ""
    public_corpus_query: str = ""  # Wikipedia search query for domain corpus


# ──────────────────────────────────────────────────────────────────
# Registry of supported datasets
# ──────────────────────────────────────────────────────────────────

DATASET_REGISTRY: Dict[str, DatasetSpec] = {
    "sst2": DatasetSpec(
        source="glue",
        subset="sst2",
        text_column="sentence",
        label_column="label",
        task_type="sentiment",
        num_labels=2,
        splits=["train", "validation"],
        description="Stanford Sentiment Treebank (binary)",
        public_corpus_query="movie review film criticism opinion",
    ),
    "ag_news": DatasetSpec(
        source="fancyzhx/ag_news",
        subset=None,
        text_column="text",
        label_column="label",
        task_type="topic",
        num_labels=4,
        splits=["train", "test"],
        description="AG News Topic Classification (4-class)",
        public_corpus_query="news article world business sports science technology",
    ),
    "imdb": DatasetSpec(
        source="stanfordnlp/imdb",
        subset=None,
        text_column="text",
        label_column="label",
        task_type="sentiment",
        num_labels=2,
        splits=["train", "test"],
        description="IMDB Movie Review Sentiment (binary)",
        public_corpus_query="movie review film acting cinema",
    ),
}


def list_datasets() -> List[str]:
    """Return list of registered dataset keys."""
    return list(DATASET_REGISTRY.keys())


def get_dataset_spec(dataset_key: str) -> DatasetSpec:
    """
    Get the specification for a dataset.
    
    Args:
        dataset_key: Registry key (e.g., "sst2", "ag_news", "imdb").
        
    Returns:
        DatasetSpec for the requested dataset.
        
    Raises:
        KeyError: If dataset is not in registry.
    """
    if dataset_key not in DATASET_REGISTRY:
        available = ", ".join(DATASET_REGISTRY.keys())
        raise KeyError(
            f"Dataset '{dataset_key}' not found. Available: {available}"
        )
    return DATASET_REGISTRY[dataset_key]


def load_registered_dataset(
    dataset_key: str,
    max_samples: Optional[int] = None,
) -> Tuple[Dataset, DatasetSpec]:
    """
    Load a dataset from the registry with standardized columns.
    
    Returns a Dataset with columns ['text', 'label'], regardless of
    the original column names.
    
    Args:
        dataset_key: Registry key.
        max_samples: Optional sample limit.
        
    Returns:
        Tuple of (standardized Dataset, DatasetSpec).
    """
    spec = get_dataset_spec(dataset_key)
    logger.info(f"Loading dataset: {dataset_key} ({spec.description})")
    
    raw = load_dataset(spec.source, spec.subset)
    
    # Combine splits
    dfs = []
    for split in spec.splits:
        if split in raw:
            dfs.append(raw[split].to_pandas())
    
    if not dfs:
        raise ValueError(f"No valid splits found for {dataset_key}")
    
    full_df = pd.concat(dfs, ignore_index=True)
    
    # Standardize column names
    rename_map = {}
    if spec.text_column != "text":
        rename_map[spec.text_column] = "text"
    if spec.label_column != "label":
        rename_map[spec.label_column] = "label"
    if rename_map:
        full_df = full_df.rename(columns=rename_map)
    
    # Validate
    assert "text" in full_df.columns, f"Missing 'text' column after rename"
    assert "label" in full_df.columns, f"Missing 'label' column after rename"
    
    # Clean
    full_df = full_df.dropna(subset=["text"])
    full_df = full_df[full_df["text"].str.strip().astype(bool)]
    
    # Truncate long texts for efficiency (IMDB reviews can be very long)
    if dataset_key == "imdb":
        full_df["text"] = full_df["text"].str[:512]
    
    if max_samples:
        full_df = full_df.head(max_samples)
    
    logger.info(f"Loaded {len(full_df)} samples, {spec.num_labels} labels")
    return Dataset.from_pandas(full_df[["text", "label"]]), spec


def generate_public_corpus(
    dataset_key: str,
    output_path: str,
    num_samples: int = 500,
) -> pd.DataFrame:
    """
    Generate a domain-matched public corpus using Wikipedia.
    
    Downloads Wikipedia articles matching the dataset's domain
    and saves them as a CSV with a 'text' column.
    
    Args:
        dataset_key: Registry key to match domain.
        output_path: Where to save the CSV.
        num_samples: Number of passages to generate.
        
    Returns:
        DataFrame with 'text' column.
    """
    spec = get_dataset_spec(dataset_key)
    logger.info(f"Generating public corpus for {dataset_key} domain...")
    
    try:
        # Use Wikipedia dataset from HuggingFace
        wiki = load_dataset(
            "wikipedia", "20220301.simple",
            split=f"train[:{num_samples}]"
        )
        texts = [article["text"][:1000] for article in wiki if len(article["text"]) > 100]
    except Exception as e:
        logger.warning(f"Wikipedia download failed: {e}. Using fallback corpus.")
        # Fallback: generate domain-relevant placeholder texts
        texts = _generate_fallback_corpus(spec, num_samples)
    
    df = pd.DataFrame({"text": texts[:num_samples]})
    
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Public corpus saved: {output_path} ({len(df)} passages)")
    return df


def _generate_fallback_corpus(spec: DatasetSpec, n: int) -> List[str]:
    """
    Generate fallback corpus when Wikipedia is unavailable.
    Uses the dataset itself (public split) as corpus.
    """
    logger.info("Using dataset's own text as fallback public corpus")
    try:
        raw = load_dataset(spec.source, spec.subset)
        split = spec.splits[0]
        texts = [ex[spec.text_column] for ex in raw[split]]
        # Shuffle and take a subset to avoid data leakage
        import random
        random.shuffle(texts)
        return texts[:n]
    except Exception:
        return [f"Public document {i} for {spec.task_type} domain." for i in range(n)]
