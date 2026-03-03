"""
Semantic Indexing with FAISS and DP-Noisy Retrieval.

Handles text chunking, embedding, FAISS index management,
and privacy-preserving retrieval with calibrated Laplacian noise.
"""

from __future__ import annotations
import faiss
import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer
from ..logger import get_logger

logger = get_logger("rag_pipeline.indexing")


def chunk_text(
    documents: List[str], 
    chunk_size: int, 
    chunk_overlap: int,
) -> List[str]:
    """
    Chunks documents into smaller overlapping passages.
    
    Args:
        documents: List of raw document strings.
        chunk_size: Character length of each chunk.
        chunk_overlap: Overlap between consecutive chunks.
        
    Returns:
        List of text passages.
    """
    passages = []
    for doc in documents:
        if not isinstance(doc, str):
            continue
        for i in range(0, len(doc), chunk_size - chunk_overlap):
            passages.append(doc[i : i + chunk_size])
    
    logger.info(f"Chunked {len(documents)} documents into {len(passages)} passages")
    return passages


class SemanticIndexer:
    """FAISS-based semantic indexer for retrieval-augmented generation."""

    def __init__(self, embedding_model_name: str, embedding_dim: int):
        self.model = SentenceTransformer(embedding_model_name)
        self.dimension = embedding_dim
        self.index: faiss.Index | None = None

    def build_index(self, corpus_passages: List[str], use_hnsw: bool = True) -> None:
        """
        Builds the FAISS index from a list of text passages.
        
        Args:
            corpus_passages: List of text passages to index.
            use_hnsw: Use HNSW index (fast) vs Flat L2 (exact).
        """
        logger.info("Encoding passages for FAISS index...")
        embeddings = self.model.encode(
            corpus_passages, show_progress_bar=True, convert_to_numpy=True
        )
        embeddings = np.float32(embeddings)

        if use_hnsw:
            logger.info("Building HNSW FAISS index...")
            self.index = faiss.IndexHNSWFlat(self.dimension, 32, faiss.METRIC_INNER_PRODUCT)
        else:
            logger.info("Building Flat L2 FAISS index...")
            self.index = faiss.IndexFlatL2(self.dimension)

        self.index.add(embeddings)
        logger.info(f"Index built: {self.index.ntotal} vectors")

    def save_index(self, path: str) -> None:
        """Saves the FAISS index to disk."""
        faiss.write_index(self.index, path)
        logger.info(f"Index saved to {path}")

    def load_index(self, path: str) -> None:
        """Loads a FAISS index from disk."""
        self.index = faiss.read_index(path)
        logger.info(f"Index loaded from {path} ({self.index.ntotal} vectors)")

    def retrieve(
        self, 
        query_texts: List[str], 
        k: int, 
        privacy_epsilon: float = 0.0,
    ) -> np.ndarray:
        """
        Retrieves top-k passages, optionally adding calibrated DP noise.
        
        Args:
            query_texts: List of query strings.
            k: Number of neighbors to retrieve.
            privacy_epsilon: Per-query epsilon for Laplacian noise (0 = no noise).
            
        Returns:
            Array of shape (n_queries, k) with passage indices.
        """
        if self.index is None:
            raise RuntimeError("Index has not been built or loaded.")

        query_embeddings = self.model.encode(query_texts, convert_to_numpy=True)
        query_embeddings = np.float32(query_embeddings)

        if privacy_epsilon > 0.0:
            from ..privacy_budget import add_calibrated_noise
            query_embeddings = add_calibrated_noise(
                query_embeddings, epsilon=privacy_epsilon, sensitivity=2.0
            )

        _distances, indices = self.index.search(query_embeddings, k)
        return indices