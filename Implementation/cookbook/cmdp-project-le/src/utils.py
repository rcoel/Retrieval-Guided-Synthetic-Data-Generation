import torch
from torch.utils.data import TensorDataset, DataLoader

def create_dataloader(texts, labels, tokenizer, batch_size):
    encodings = tokenizer(texts, truncation=True, padding=True, return_tensors='pt')
    dataset = TensorDataset(encodings['input_ids'], encodings['attention_mask'], torch.tensor(labels))

    def collate_fn(batch):
        input_ids, attention_mask, labels = zip(*batch)
        return {
            'input_ids': torch.stack(input_ids),
            'attention_mask': torch.stack(attention_mask),
            'labels': torch.stack(labels)
        }

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    return dataloader