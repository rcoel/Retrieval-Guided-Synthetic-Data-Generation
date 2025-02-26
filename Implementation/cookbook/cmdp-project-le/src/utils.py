import torch

def create_dataloader(texts, labels, tokenizer, batch_size):
    encodings = tokenizer(texts, truncation=True, padding=True, return_tensors='pt')
    dataset = TensorDataset(encodings['input_ids'], encodings['attention_mask'], torch.tensor(labels))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader
