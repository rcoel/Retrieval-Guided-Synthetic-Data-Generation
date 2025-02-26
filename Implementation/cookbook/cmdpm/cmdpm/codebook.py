import torch
import numpy as np
from sklearn.cluster import KMeans

class Codebook:
    def __init__(self, codebook_size, embedding_dim):
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.codebook_vectors = None
        self._is_trained = False

    def train(self, embeddings, random_state=42):
        kmeans = KMeans(n_clusters=self.codebook_size, random_state=random_state, n_init=10)
        kmeans.fit(embeddings.cpu().numpy() if isinstance(embeddings, torch.Tensor) else embeddings)
        self.codebook_vectors = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
        self._is_trained = True

    def quantize(self, embeddings):
        if not self._is_trained:
            raise RuntimeError("Codebook not trained")
        distances = torch.cdist(embeddings, self.codebook_vectors)
        return torch.argmin(distances, dim=-1)

    def dequantize(self, indices):
        return self.codebook_vectors[indices]

    @property
    def is_trained(self):
        return self._is_trained