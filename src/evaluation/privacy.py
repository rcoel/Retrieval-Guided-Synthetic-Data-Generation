from nltk import ngrams
from tqdm import tqdm
from nltk.tokenize import word_tokenize

def calculate_exact_match_ratio(original_texts, synthetic_texts):
    """Calculates the percentage of exact matches."""
    original_set = set(o.strip() for o in original_texts)
    match_count = sum(1 for s_text in synthetic_texts if s_text.strip() in original_set)
    return (match_count / len(synthetic_texts)) * 100

def calculate_ngram_overlap(original_texts, synthetic_texts, n=5):
    """Calculates the percentage of n-grams in synthetic data that appeared in original data."""
    original_ngrams = set()
    for text in tqdm(original_texts, desc=f"Extracting {n}-grams from original data"):
        tokens = word_tokenize(text)
        original_ngrams.update(ngrams(tokens, n))
    
    if not original_ngrams: return 0.0

    synthetic_ngrams_count, overlap_count = 0, 0
    for text in tqdm(synthetic_texts, desc=f"Analyzing {n}-grams in synthetic data"):
        tokens = word_tokenize(text)
        current_ngrams = list(ngrams(tokens, n))
        synthetic_ngrams_count += len(current_ngrams)
        for ng in current_ngrams:
            if ng in original_ngrams:
                overlap_count += 1
    
    if synthetic_ngrams_count == 0: return 0.0
    return (overlap_count / synthetic_ngrams_count) * 100

def calculate_single_pair_ngram_overlap(original_text, synthetic_text, n=5):
    """Calculates n-gram overlap between a single pair of original and synthetic strings."""
    tokens_orig = word_tokenize(original_text.lower())
    tokens_synth = word_tokenize(synthetic_text.lower())
    
    grams_orig = set(ngrams(tokens_orig, n))
    grams_synth = list(ngrams(tokens_synth, n))
    
    if not grams_orig or not grams_synth:
        return 0.0
        
    overlap_count = sum(1 for g in grams_synth if g in grams_orig)
    return overlap_count / len(grams_synth)