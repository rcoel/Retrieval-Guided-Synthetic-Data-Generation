import torch.nn as nn

class CMDPModelForClassification(nn.Module):
    def __init__(self, cmdp_encoder, num_labels):
        super().__init__()
        self.cmdp_encoder = cmdp_encoder
        self.classifier = nn.Linear(cmdp_encoder.codebook.embedding_dim, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        cmdp_representations = self.cmdp_encoder(input_ids, attention_mask)
        pooled_output = cmdp_representations[:, 0, :]
        logits = self.classifier(pooled_output)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return {"loss": loss, "logits": logits}

    @property
    def num_labels(self):
        return self.classifier.out_features
