import torch
import json
import nltk

def get_device():
    """Returns the appropriate device (CUDA or CPU)."""
    return "cuda" if torch.cuda.is_available() else "cpu"

def save_synthetic_data(data, path):
    """Saves generated synthetic data to a JSONL file."""
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
    print(f"Synthetic data saved to {path}")

def load_synthetic_data(path):
    """Loads synthetic data from a JSONL file."""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def setup_nltk():
    """Downloads NLTK data if not present."""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Downloading NLTK's 'punkt' tokenizer...")
        nltk.download('punkt')