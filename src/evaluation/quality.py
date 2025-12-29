import torch
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import evaluate
from .. import config

def calculate_self_bleu(texts):
    """Calculates Self-BLEU score for diversity using the evaluate library."""
    bleu = evaluate.load("bleu")
    total_bleu = 0.0
    
    for i in tqdm(range(len(texts)), desc="Calculating Self-BLEU"):
        hypothesis = texts[i]
        references = [texts[j] for j in range(len(texts)) if i != j]
        # The evaluate library expects a list of lists for references
        results = bleu.compute(predictions=[hypothesis], references=[references])
        total_bleu += results['bleu']
        
    return total_bleu / len(texts)

def calculate_ttr(texts):
    """Calculates Type-Token Ratio (TTR) for lexical richness."""
    all_tokens = [token for text in texts for token in word_tokenize(text.lower())]
    if not all_tokens:
        return 0
    return len(set(all_tokens)) / len(all_tokens)

def calculate_semantic_similarity(original_texts, synthetic_texts, model=None):
    """Measures cosine similarity between original and synthetic embeddings."""
    if model is None:
        model = SentenceTransformer(config.EMBEDDING_MODEL)
    
    original_embeddings = model.encode(original_texts, convert_to_tensor=True)
    synthetic_embeddings = model.encode(synthetic_texts, convert_to_tensor=True)
    
    cosine_scores = util.cos_sim(original_embeddings, synthetic_embeddings)
    
    return torch.diag(cosine_scores).mean().item()

def measure_similarity_batch(original_texts, synthetic_texts, model):
    """Returns a list of similarity scores for each pair in the batch."""
    original_embeddings = model.encode(original_texts, convert_to_tensor=True)
    synthetic_embeddings = model.encode(synthetic_texts, convert_to_tensor=True)
    
    # Calculate pair-wise similarity
    # util.cos_sim returns matrix, we want diagonal (one-to-one)
    cosine_scores = util.pairwise_cos_sim(original_embeddings, synthetic_embeddings)
    return cosine_scores.cpu().tolist()