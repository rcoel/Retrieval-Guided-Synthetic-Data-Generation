import torch
from transformers import BertTokenizer
from model import CMDPModelForClassification

def inference(model, tokenizer, text, device):
    model.to(device)
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs['logits']
        predicted_class_id = torch.argmax(logits, dim=-1).item()
    return predicted_class_id

if __name__ == "__main__":
    # Example usage
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    example_text = "This was an amazing film, I loved it!"
    # Load your trained model here
    # model = ...
    predicted_class = inference(model, tokenizer, example_text, device)
    print(f"Example Text: '{example_text}'")
    print(f"Predicted Class ID: {predicted_class} (1: Positive, 0: Negative)")
