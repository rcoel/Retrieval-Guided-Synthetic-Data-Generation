import torch
import faiss
import numpy as np
from transformers import BertTokenizer, BertModel
from rouge import Rouge

class InvBERTKNN:
    def __init__(self, model_name='bert-base-uncased', k=5):
        """
        Initialize the InvBERT-KNN model.

        Args:
            model_name (str): Name of the pre-trained BERT model.
            k (int): Number of nearest neighbors for KNN.
        """
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.eval()  # Set to evaluation mode
        self.k = k
        self.index = None  # FAISS index for KNN
        self.references = []  # Store reference texts
        self.labels = []  # Store reference labels

    def encode(self, texts):
        """
        Encode a list of texts into BERT embeddings.

        Args:
            texts (list): List of input texts.

        Returns:
            numpy.ndarray: Array of embeddings (num_texts, embedding_dim).
        """
        inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # Use [CLS] token embeddings
        return embeddings

    def build_index(self, reference_texts, reference_labels):
        """
        Build a FAISS index for KNN search.

        Args:
            reference_texts (list): List of reference texts.
            reference_labels (list): List of reference labels.
        """
        self.references = reference_texts
        self.labels = reference_labels
        embeddings = self.encode(reference_texts)
        d = embeddings.shape[1]  # Dimension of embeddings
        self.index = faiss.IndexFlatL2(d)  # L2 distance for KNN
        self.index.add(embeddings.astype(np.float32))

    def predict(self, query_texts):
        """
        Predict labels for query texts using KNN.

        Args:
            query_texts (list): List of query texts.

        Returns:
            list: Predicted labels.
        """
        query_embeddings = self.encode(query_texts)
        distances, indices = self.index.search(query_embeddings.astype(np.float32), self.k)
        predictions = []
        for idx in indices:
            neighbor_labels = [self.labels[i] for i in idx]
            predicted_label = max(set(neighbor_labels), key=neighbor_labels.count)  # Majority voting
            predictions.append(predicted_label)
        return predictions

    def evaluate_metrics(self, query_texts, query_labels):
        """
        Evaluate the model using Top-K Accuracy, ROUGE-L, and Token Hit Rate.

        Args:
            query_texts (list): List of query texts.
            query_labels (list): List of ground truth labels.

        Returns:
            dict: Dictionary containing evaluation metrics.
        """
        predictions = self.predict(query_texts)

        # Top-K Accuracy
        top_k_acc = top_k_accuracy(predictions, query_labels, k=self.k)

        # ROUGE-L
        rouge_l = rouge_l_score([str(p) for p in predictions], [str(l) for l in query_labels])

        # Token Hit Rate
        token_hit = token_hit_rate([str(p) for p in predictions], [str(l) for l in query_labels])

        return {
            'top_k_accuracy': top_k_acc,
            'rouge_l': rouge_l,
            'token_hit_rate': token_hit,
        }

def top_k_accuracy(predictions, labels, k=1):
    """
    Compute Top-K Accuracy.

    Args:
        predictions (list): List of predicted labels.
        labels (list): List of ground truth labels.
        k (int): Number of top predictions to consider.

    Returns:
        float: Top-K Accuracy.
    """
    correct = sum(1 for p, l in zip(predictions, labels) if p == l)
    return correct / len(labels)

def rouge_l_score(predictions, references):
    """
    Compute ROUGE-L score.

    Args:
        predictions (list): List of predicted texts.
        references (list): List of reference texts.

    Returns:
        float: ROUGE-L F1 score.
    """
    rouge = Rouge()
    scores = rouge.get_scores(predictions, references, avg=True)
    return scores['rouge-l']['f']

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

def main():
    # Example data
    reference_texts = [
        "This movie was fantastic! I loved it.",
        "The film was terrible. I hated it.",
        "Amazing acting and a great plot.",
        "Boring and predictable. Waste of time."
    ]
    reference_labels = [1, 0, 1, 0]  # 1: Positive, 0: Negative

    query_texts = [
        "I really enjoyed this film. It was great!",
        "The movie was awful. I didn't like it."
    ]
    query_labels = [1, 0]  # Ground truth labels for evaluation

    # Initialize InvBERT-KNN
    invbert_knn = InvBERTKNN(k=3)

    # Build the FAISS index with reference data
    invbert_knn.build_index(reference_texts, reference_labels)

    # Evaluate on query data
    metrics = invbert_knn.evaluate_metrics(query_texts, query_labels)
    print(f"Top-K Accuracy: {metrics['top_k_accuracy']:.4f}")
    print(f"ROUGE-L: {metrics['rouge_l']:.4f}")
    print(f"Token Hit Rate: {metrics['token_hit_rate']:.4f}")

if __name__ == "__main__":
    main()