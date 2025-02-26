import torch
import torch.nn as nn
import numpy as np
import torch.distributions.laplace as laplace_dist

class CMDPEncoder(nn.Module):
    def __init__(self, base_encoder, codebook, k, dp_noise_scale=0.0):
        super().__init__()
        self.base_encoder = base_encoder
        self.codebook = codebook
        self.k = k
        self.dp_noise_scale = dp_noise_scale

    def forward(self, input_ids, attention_mask):
        batch_size, seq_len = input_ids.shape
        base_embeddings = self.base_encoder(input_ids, attention_mask=attention_mask).last_hidden_state

        quantized_embeddings_indices = []
        for b in range(batch_size):
            seq_indices = []
            for s in range(seq_len):
                seq_indices.append(self.codebook.quantize(base_embeddings[b, s]))
            quantized_embeddings_indices.append(torch.stack(seq_indices))
        quantized_embeddings_indices = torch.stack(quantized_embeddings_indices)

        mixed_codebook_vectors = []
        for b in range(batch_size):
            seq_mixed_vectors = []
            for s in range(seq_len):
                mixed_vector = torch.zeros(self.codebook.embedding_dim)
                for _ in range(self.k):
                    random_index = np.random.randint(batch_size)
                    codebook_index = quantized_embeddings_indices[random_index, s]
                    mixed_vector += self.codebook.dequantize(codebook_index)
                mixed_vector /= self.k
                seq_mixed_vectors.append(mixed_vector)
            mixed_codebook_vectors.append(torch.stack(seq_mixed_vectors))
        mixed_codebook_vectors = torch.stack(mixed_codebook_vectors)

        if self.dp_noise_scale > 0:
            laplace = laplace_dist.Laplace(torch.tensor(0.0), torch.tensor(self.dp_noise_scale))
            noise = laplace.sample(mixed_codebook_vectors.shape)
            mixed_codebook_vectors += noise

        return mixed_codebook_vectors
