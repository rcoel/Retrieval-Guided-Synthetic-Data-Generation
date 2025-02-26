import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Add project root to path

import torch
import yaml
from transformers import BertTokenizer
from cmdpm import Codebook, CMDPEncoder, CMDPModel
from rouge import Rouge

def load_model(config_path, model_path, device="cpu"):
    """
    Load the trained CMDPModel and its components.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Initialize Codebook
    codebook = Codebook(config['codebook']['size'], config['codebook']['embedding_dim'])

    # Initialize CMDPEncoder and CMDPModel
    encoder = CMDPEncoder(codebook, k=config['model']['k_mixing'])
    model = CMDPModel(encoder, config['model']['num_labels'])

    # Load trained model and codebook
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    codebook.codebook_vectors = checkpoint['codebook_vectors']  # Load the trained codebook vectors
    codebook._is_trained = True  # Mark the codebook as trained

    model.to(device)
    model.eval()

    return model, tokenizer

def predict(text, model, tokenizer, device="cpu"):
    """
    Predict the class of a given text.
    """
    # Tokenize the input text
    inputs = tokenizer(
        text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    ).to(device)

    # Perform inference
    with torch.no_grad():
        outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        logits = outputs['logits']
        predicted_class = torch.argmax(logits, dim=-1).item()

    return predicted_class, logits

def rouge_l_score(predictions, references):
    """
    Compute ROUGE-L score between predictions and references.
    
    Args:
        predictions (list): List of predicted texts.
        references (list): List of reference texts.
    
    Returns:
        float: ROUGE-L F1 score.
    """
    rouge = Rouge()
    scores = rouge.get_scores(predictions, references, avg=True)
    return scores['rouge-l']['f']

def top_k_accuracy(logits, labels, k=1):
    """
    Compute Top-K Accuracy.
    
    Args:
        logits (torch.Tensor): Model output logits (batch_size, num_classes).
        labels (torch.Tensor): Ground truth labels (batch_size,).
        k (int): Number of top predictions to consider.
    
    Returns:
        float: Top-K Accuracy.
    """
    num_classes = logits.size(1)
    if k > num_classes:
        k = num_classes  # Adjust k to be no larger than the number of classes
    _, top_k_preds = torch.topk(logits, k, dim=1)
    correct = top_k_preds.eq(labels.view(-1, 1)).sum().item()
    return correct / len(labels)

def token_hit_rate(predictions, references):
    """
    Compute Token Hit Rate.
    
    Args:
        predictions (list): List of predicted texts.
        references (list): List of reference texts.
    
    Returns:
        float: Token Hit Rate.
    """
    hit_rates = []
    for pred, ref in zip(predictions, references):
        pred_tokens = set(pred.split())
        ref_tokens = set(ref.split())
        hit_rate = len(pred_tokens.intersection(ref_tokens)) / len(ref_tokens)
        hit_rates.append(hit_rate)
    return sum(hit_rates) / len(hit_rates)

def evaluate_metrics(model, tokenizer, test_texts, test_labels, device="cpu"):
    """
    Evaluate the model using ROUGE-L, Top-K Accuracy, and Token Hit Rate.
    """
    predictions = []
    all_logits = []
    all_labels = []

    for text, label in zip(test_texts, test_labels):
        pred_class, logits = predict(text, model, tokenizer, device)
        predictions.append(str(pred_class))  # Convert to string for ROUGE-L
        all_logits.append(logits)
        all_labels.append(label)

    # ROUGE-L
    rouge_l = rouge_l_score(predictions, [str(label) for label in test_labels])

    # Top-K Accuracy
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.tensor(all_labels, device=device)
    top_k_acc = top_k_accuracy(all_logits, all_labels, k=3)  # Change k as needed

    # Token Hit Rate
    token_hit = token_hit_rate(predictions, [str(label) for label in test_labels])

    return {
        'rouge_l': rouge_l,
        'top_k_accuracy': top_k_acc,
        'token_hit_rate': token_hit,
    }

def main():
    # Configuration and paths
    config_path = "configs/train_config.yaml"
    model_path = os.path.join("models", "cmdp_model.pth")  # Correct path to the saved model
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the trained model
    model, tokenizer = load_model(config_path, model_path, device)

    # Example test data (replace with your actual test data)
    test_texts = [
        "This movie was absolutely fantastic! I loved every moment of it.",
        "The film was a complete waste of time. Terrible acting and plot.",
    ]
    test_labels = [1, 0]  # Corresponding labels

    # Evaluate metrics
    metrics = evaluate_metrics(model, tokenizer, test_texts, test_labels, device)
    print(f"ROUGE-L: {metrics['rouge_l']:.4f}")
    print(f"Top-K Accuracy: {metrics['top_k_accuracy']:.4f}")
    print(f"Token Hit Rate: {metrics['token_hit_rate']:.4f}")

if __name__ == "__main__":
    main()