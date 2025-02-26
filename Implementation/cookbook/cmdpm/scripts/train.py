import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Add project root to path

import torch
import yaml
from transformers import BertTokenizer, BertModel
from cmdpm import Codebook, CMDPEncoder, CMDPModel, CMDDataset, create_dataloader, load_dataset

def main():
    # Load config
    with open("configs/train_config.yaml") as f:
        config = yaml.safe_load(f)

    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Load and split data
    train_texts, train_labels, val_texts, val_labels = load_dataset(config['data']['train_path'])
    train_dataset = CMDDataset(train_texts, train_labels, tokenizer)
    val_dataset = CMDDataset(val_texts, val_labels, tokenizer)
    train_loader = create_dataloader(train_dataset, config['training']['batch_size'])
    val_loader = create_dataloader(val_dataset, config['training']['batch_size'])

    # Initialize Codebook and train it
    codebook = Codebook(config['codebook']['size'], config['codebook']['embedding_dim'])

    # Use a separate BERT model to generate embeddings for codebook training
    bert_for_codebook = BertModel.from_pretrained('bert-base-uncased')
    bert_for_codebook.eval()  # Set to evaluation mode

    # Generate embeddings for codebook training
    all_embeddings = []
    with torch.no_grad():
        for batch in train_loader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            embeddings = bert_for_codebook(input_ids, attention_mask=attention_mask).last_hidden_state
            all_embeddings.append(embeddings.view(-1, config['codebook']['embedding_dim']))  # Flatten embeddings
    all_embeddings = torch.cat(all_embeddings, dim=0)

    # Train the Codebook
    codebook.train(all_embeddings)

    # Initialize CMDPEncoder and CMDPModel
    encoder = CMDPEncoder(codebook, k=config['model']['k_mixing'])
    model = CMDPModel(encoder, config['model']['num_labels'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['learning_rate'])

    # Train the model
    model.train()
    for epoch in range(config['training']['epochs']):
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
            )
            loss = outputs['loss']
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader)}")

    # Save the trained model and codebook
    os.makedirs("models", exist_ok=True)  # Create the "models" directory if it doesn't exist
    model_save_path = os.path.join("models", "cmdp_model.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'codebook_vectors': codebook.codebook_vectors,  # Save the trained codebook vectors
    }, model_save_path)
    print(f"Model and codebook saved to {model_save_path}")

if __name__ == "__main__":
    main()