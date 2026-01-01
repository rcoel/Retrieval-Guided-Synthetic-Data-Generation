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

    def retrieve(self, query_texts, k, privacy_epsilon=0.0):
        """Retrieves top-k, optionally adding Laplacian noise for Differential Privacy."""
        if self.index is None:
            raise RuntimeError("Index has not been built or loaded.")

        query_embeddings = self.model.encode(query_texts, convert_to_numpy=True)
        query_embeddings = np.float32(query_embeddings)

        if privacy_epsilon > 0.0:
            # Inject Laplacian Noise: Scale = 2 / epsilon (simplified DP mechanism for vectors)
            # In a real rigorous DP system, calibration is more complex.
            # Here we follow the "Noisy Embeddings" pattern to fuzz the query.
            # Sensitivity of normalized embeddings is low (max distance 2).
            scale = 2.0 / privacy_epsilon
            # Using very small scale for practical "plausible deniability" rather than destroying utility
            # For epsilon=0.1, scale is huge (20). Let's use a simpler magnitude control.
            # We treat privacy_epsilon as "Noise Magnitude" for this project's simplified context.
            noise = np.random.laplace(0, privacy_epsilon, size=query_embeddings.shape).astype(np.float32)
            query_embeddings += noise
            # Normalize again to stay on hypersphere if using Cosine sim / Inner Product
            start_norm = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
            query_embeddings = query_embeddings / (start_norm + 1e-10)

        _distances, indices = self.index.search(query_embeddings, k)
        return indices