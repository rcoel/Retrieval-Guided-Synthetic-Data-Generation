import torch
import torch.nn as nn
from transformers import BertModel

class CMDPEncoder(nn.Module):
    def __init__(self, codebook, k=4, dp_epsilon=0.1):
        super().__init__()
        self.base_encoder = BertModel.from_pretrained('bert-base-uncased')
        self.codebook = codebook
        self.k = k
        self.dp_epsilon = dp_epsilon
        self._freeze_encoder()

    def _freeze_encoder(self):
        for param in self.base_encoder.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        base_embeddings = self.base_encoder(input_ids, attention_mask=attention_mask).last_hidden_state
        quantized_indices = self.codebook.quantize(base_embeddings)
        batch_size, seq_len = quantized_indices.shape

        mixed_vectors = []
        for b in range(batch_size):
            seq_vectors = []
            for s in range(seq_len):
                random_indices = torch.randint(0, batch_size, (self.k,))
                mixed_vector = self.codebook.dequantize(quantized_indices[random_indices, s]).mean(dim=0)
                seq_vectors.append(mixed_vector)
            mixed_vectors.append(torch.stack(seq_vectors))
        mixed_vectors = torch.stack(mixed_vectors)

        if self.dp_epsilon > 0:
            noise = torch.randn_like(mixed_vectors) * self.dp_epsilon
            mixed_vectors += noise

        return mixed_vectors