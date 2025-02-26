import torch
from sklearn.cluster import KMeans
import torch.distributions.laplace as laplace_dist

class Codebook:
    def __init__(self, codebook_size, embedding_dim):
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.codebook_vectors = None

    def train_codebook(self, embeddings):
        kmeans = KMeans(n_clusters=self.codebook_size, random_state=42, n_init=10)
        kmeans.fit(embeddings)
        self.codebook_vectors = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)

    def quantize(self, embedding):
        if self.codebook_vectors is None:
            raise ValueError("Codebook not trained yet. Call train_codebook first.")
        distances = torch.cdist(embedding.unsqueeze(0), self.codebook_vectors.unsqueeze(0))[0]
        closest_index = torch.argmin(distances)
        return closest_index

    def dequantize(self, index):
        if self.codebook_vectors is None:
            raise ValueError("Codebook not trained yet. Call train_codebook first.")
        return self.codebook_vectors[index]

    def add_dp_noise(self, noise_scale):
        if self.codebook_vectors is None:
            raise ValueError("Codebook not trained yet. Call train_codebook first.")
        laplace = laplace_dist.Laplace(torch.tensor([0.0]), torch.tensor([noise_scale]))
        noise = laplace.sample(self.codebook_vectors.shape)
        self.codebook_vectors += noise
