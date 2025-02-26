import torch
from tqdm import tqdm
from .utils.logger import TrainingLogger

class CMDTrainer:
    def __init__(self, model, optimizer, device='cuda'):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.logger = TrainingLogger()
        
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(dataloader, desc="Training"):
            self.optimizer.zero_grad()
            
            inputs = {
                'input_ids': batch['input_ids'].to(self.device),
                'attention_mask': batch['attention_mask'].to(self.device),
                'labels': batch['labels'].to(self.device)
            }
            
            outputs = self.model(**inputs)
            loss = outputs['loss']
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            self.logger.log_step(loss=loss.item())
            
        return total_loss / len(dataloader)

    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                inputs = {
                    'input_ids': batch['input_ids'].to(self.device),
                    'attention_mask': batch['attention_mask'].to(self.device),
                    'labels': batch['labels'].to(self.device)
                }
                
                outputs = self.model(**inputs)
                loss = outputs['loss']
                total_loss += loss.item()
                
                _, predicted = torch.max(outputs['logits'], 1)
                correct += (predicted == inputs['labels']).sum().item()
                total += inputs['labels'].size(0)
                
        return {
            'loss': total_loss / len(dataloader),
            'accuracy': correct / total
        }

    def train(self, train_loader, val_loader, epochs):
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)
            
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}")
            print(f"Val Accuracy: {val_metrics['accuracy']:.4f}\n")
            
            self.logger.log_epoch(
                epoch,
                train_loss,
                val_metrics['loss'],
                val_metrics['accuracy']
            )