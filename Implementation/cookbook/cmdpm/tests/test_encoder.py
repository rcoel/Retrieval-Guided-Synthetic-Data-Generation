import torch
from cmdpm import Codebook, CMDPEncoder

def test_encoder_forward():
    codebook = Codebook(codebook_size=10, embedding_dim=8)
    embeddings = torch.randn(100, 8)
    codebook.train(embeddings)
    
    encoder = CMDPEncoder(codebook, k=4)
    input_ids = torch.randint(0, 100, (32, 16))  # Batch of 32 sequences, each of length 16
    attention_mask = torch.ones_like(input_ids)
    
    output = encoder(input_ids, attention_mask)
    assert output.shape == (32, 16, 8), "Encoder output should have shape (batch_size, seq_len, embedding_dim)"