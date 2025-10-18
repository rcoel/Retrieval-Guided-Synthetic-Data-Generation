import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

def chunk_text(documents, chunk_size, chunk_overlap):
    """Chunks documents into smaller passages."""
    passages = []
    for doc in documents:
        if not isinstance(doc, str): continue
        for i in range(0, len(doc), chunk_size - chunk_overlap):
            passages.append(doc[i:i + chunk_size])
    return passages

class SemanticIndexer:
    def __init__(self, embedding_model_name, embedding_dim):
        self.model = SentenceTransformer(embedding_model_name)
        self.dimension = embedding_dim
        self.index = None

    def build_index(self, corpus_passages, use_hnsw=True):
        """Builds the FAISS index from a list of text passages."""
        print("Encoding passages for FAISS index...")
        embeddings = self.model.encode(corpus_passages, show_progress_bar=True, convert_to_numpy=True)
        embeddings = np.float32(embeddings)

        if use_hnsw:
            print("Building HNSW FAISS index...")
            self.index = faiss.IndexHNSWFlat(self.dimension, 32, faiss.METRIC_INNER_PRODUCT)
        else:
            print("Building Flat L2 FAISS index...")
            self.index = faiss.IndexFlatL2(self.dimension)

        self.index.add(embeddings)
        print(f"Index built successfully with {self.index.ntotal} vectors.")

    def save_index(self, path):
        """Saves the FAISS index to disk."""
        faiss.write_index(self.index, path)
        print(f"Index saved to {path}")

    def load_index(self, path):
        """Loads a FAISS index from disk."""
        self.index = faiss.read_index(path)
        print(f"Index loaded from {path} with {self.index.ntotal} vectors.")

    def retrieve(self, query_texts, k):
        """Retrieves the top-k most similar document indices for each query."""
        if self.index is None:
            raise RuntimeError("Index has not been built or loaded.")

        query_embeddings = self.model.encode(query_texts, convert_to_numpy=True)
        query_embeddings = np.float32(query_embeddings)

        _distances, indices = self.index.search(query_embeddings, k)
        return indices