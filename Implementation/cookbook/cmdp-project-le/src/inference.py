import torch
from transformers import BertTokenizer, BertModel
from model import CMDPModelForClassification
from cmdp_encoder import CMDPEncoder
from codebook import Codebook

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

    # Load the trained model
    codebook_size = 28
    embedding_dim = 768
    k_mixing_factor = 4
    dp_noise_scale = 0.1
    num_labels = 2

    codebook = Codebook(codebook_size=codebook_size, embedding_dim=embedding_dim)
    codebook.codebook_vectors = torch.load('codebook_vectors.pth', map_location=device)

    base_bert_encoder = BertModel.from_pretrained('bert-base-uncased')
    cmdp_encoder = CMDPEncoder(base_encoder=base_bert_encoder, codebook=codebook, k=k_mixing_factor, dp_noise_scale=dp_noise_scale)
    model = CMDPModelForClassification(cmdp_encoder=cmdp_encoder, num_labels=num_labels)

    # Load the model weights
    model.load_state_dict(torch.load('cmdp_model.pth', map_location=device))

    predicted_class = inference(model, tokenizer, example_text, device)
    print(f"Example Text: '{example_text}'")
    print(f"Predicted Class ID: {predicted_class} (1: Positive, 0: Negative)")
