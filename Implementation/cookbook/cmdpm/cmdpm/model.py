import torch.nn as nn

class CMDPModel(nn.Module):
    def __init__(self, encoder, num_labels):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(encoder.codebook.embedding_dim, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        encoded = self.encoder(input_ids, attention_mask)
        pooled = encoded[:, 0, :]  # CLS token
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        return {'loss': loss, 'logits': logits}