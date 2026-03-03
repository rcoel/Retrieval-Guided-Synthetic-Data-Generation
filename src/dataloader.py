"""
Data loading utilities.

Handles loading public corpora (CSV) and private datasets (HuggingFace),
with proper error handling and data validation.
"""

from __future__ import annotations
import pandas as pd
from typing import Optional
from datasets import load_dataset, Dataset
from .logger import get_logger

logger = get_logger("rag_pipeline.dataloader")


class DataLoadError(Exception):
    """Raised when data loading fails."""
    pass


def load_public_corpus(path: str) -> pd.DataFrame:
    """
    Loads the public corpus from a CSV file.
    
    Args:
        path: Path to CSV file with a 'text' column.
        
    Returns:
        DataFrame with validated 'text' column.
        
    Raises:
        DataLoadError: If file is missing or malformed.
    """
    logger.info(f"Loading public corpus from {path}...")
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        raise DataLoadError(
            f"Public corpus not found at '{path}'. "
            "Create a CSV file with a 'text' column."
        )
    except pd.errors.ParserError as e:
        raise DataLoadError(f"Failed to parse CSV at '{path}': {e}")

    if 'text' not in df.columns:
        raise DataLoadError(
            f"CSV at '{path}' must contain a 'text' column. "
            f"Found columns: {list(df.columns)}"
        )

    # Drop NaN rows and empty strings
    original_len = len(df)
    df = df.dropna(subset=['text'])
    df = df[df['text'].str.strip().astype(bool)]
    
    if len(df) == 0:
        raise DataLoadError(f"Public corpus at '{path}' contains no valid text rows.")
    
    if len(df) < original_len:
        logger.warning(
            f"Dropped {original_len - len(df)} invalid rows from corpus "
            f"({len(df)} remaining)"
        )

    logger.info(f"Loaded {len(df)} passages from public corpus")
    return df


def load_private_dataset(
    dataset_name: str, 
    subset: Optional[str] = None
) -> Dataset:
    """
    Loads a private dataset from HuggingFace datasets.
    
    Args:
        dataset_name: HuggingFace dataset identifier (e.g., "glue").
        subset: Optional dataset subset (e.g., "sst2").
        
    Returns:
        HuggingFace Dataset with standardized 'text' and 'label' columns.
        
    Raises:
        DataLoadError: If dataset cannot be loaded or processed.
    """
    logger.info(f"Loading private dataset: {dataset_name} ({subset or 'default'})")
    
    try:
        dataset = load_dataset(dataset_name, subset)
    except Exception as e:
        raise DataLoadError(f"Failed to load dataset '{dataset_name}/{subset}': {e}")

    # GLUE tasks: combine train + validation and standardize columns
    if dataset_name == "glue":
        train_df = dataset["train"].to_pandas()
        val_df = dataset["validation"].to_pandas()
        full_df = pd.concat([train_df, val_df], ignore_index=True)
        
        # Standardize text column name
        text_column_map = {
            "sentence": "text",
            "question": "text",
            "question1": "text",
            "premise": "text",
        }
        for old_name, new_name in text_column_map.items():
            if old_name in full_df.columns:
                full_df = full_df.rename(columns={old_name: new_name})
                break
    else:
        full_df = dataset["train"].to_pandas()

    # Validate required columns
    if 'text' not in full_df.columns:
        raise DataLoadError(
            f"Dataset '{dataset_name}/{subset}' has no 'text' column after standardization. "
            f"Available columns: {list(full_df.columns)}"
        )

    logger.info(f"Loaded {len(full_df)} examples from private dataset")
    return Dataset.from_pandas(full_df)