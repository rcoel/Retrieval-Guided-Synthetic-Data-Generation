import yaml
from src.train import train
from src.inference import inference
from transformers import BertTokenizer
import torch

if __name__ == "__main__":
    # Load configuration
    with open('config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Initialize tokenizer and device
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Example training and inference
    # train(...)  # Call the train function with appropriate arguments
    # inference(...)  # Call the inference function with appropriate arguments
