import torch
from cmdpm import Codebook

def test_codebook_training():
    codebook = Codebook(codebook_size=10, embedding_dim=8)
    embeddings = torch.randn(100, 8)  # 100 samples, 8-dimensional embeddings
    codebook.train(embeddings)
    assert codebook.is_trained, "Codebook should be trained after calling train()"

def test_codebook_quantization():
    codebook = Codebook(codebook_size=10, embedding_dim=8)
    embeddings = torch.randn(100, 8)
    codebook.train(embeddings)
    quantized_indices = codebook.quantize(embeddings)
    assert quantized_indices.shape == (100,), "Quantized indices should have shape (100,)"