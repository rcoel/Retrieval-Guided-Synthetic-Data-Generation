import torch
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from codebook import Codebook
from cmdp_encoder import CMDPEncoder
from model import CMDPModelForClassification
from utils import create_dataloader

def train(model, train_dataloader, optimizer, num_epochs, device):
    model.to(device)
    model.train()
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, labels=labels)
            loss = outputs['loss']
            loss.backward()
            optimizer.step()
            print(f"Epoch: {epoch+1}, Loss: {loss.item():.4f}")

if __name__ == "__main__":
    # Hyperparameters
    codebook_size = 256
    embedding_dim = 768
    k_mixing_factor = 4
    dp_noise_scale = 0.1
    num_labels = 2
    num_epochs = 3
    learning_rate = 2e-5
    batch_size = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare Data and Tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_texts = ["This movie was great!", "The plot was terrible.", "Excellent acting.", "Predictable and boring."]
    train_labels = [1, 0, 1, 0]
    train_dataloader = create_dataloader(train_texts, train_labels, tokenizer, batch_size)

    # Train Codebook
    bert_model_for_codebook = BertModel.from_pretrained('bert-base-uncased')
    all_train_embeddings = []
    with torch.no_grad():
        for batch in train_dataloader:
            input_ids = batch[0]
            attention_mask = batch[1]
            batch_embeddings = bert_model_for_codebook(input_ids, attention_mask=attention_mask).last_hidden_state
            all_train_embeddings.append(batch_embeddings.view(-1, embedding_dim))
    all_train_embeddings = torch.cat(all_train_embeddings, dim=0).numpy()
    codebook = Codebook(codebook_size=codebook_size, embedding_dim=embedding_dim)
    codebook.train_codebook(all_train_embeddings)
    codebook.add_dp_noise(noise_scale=dp_noise_scale)

    # Initialize CMDP Encoder and Model
    base_bert_encoder = BertModel.from_pretrained('bert-base-uncased')
    cmdp_encoder = CMDPEncoder(base_encoder=base_bert_encoder, codebook=codebook, k=k_mixing_factor, dp_noise_scale=dp_noise_scale)
    model = CMDPModelForClassification(cmdp_encoder=cmdp_encoder, num_labels=num_labels)

    # Training
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    train(model, train_dataloader, optimizer, num_epochs, device)
