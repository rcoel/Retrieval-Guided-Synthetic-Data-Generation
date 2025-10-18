import os
import pandas as pd
from src import config, data_loader, utils
from src.pipeline import indexing, training, generation
from src.evaluation import downstream_task, quality, privacy

def main():
    """Main function to run the entire RAG-based synthetic data pipeline."""
    
    # --- 0. Setup ---
    utils.setup_nltk()
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.MODEL_OUTPUT_DIR, exist_ok=True)
    
    print("\n" + "="*80)
    print("      STARTING SYNTHETIC DATA GENERATION PIPELINE      ")
    print("="*80 + "\n")

    # --- 1. Load Data ---
    private_dataset = data_loader.load_private_dataset(config.PRIVATE_DATA_NAME, config.PRIVATE_DATA_SUBSET)
    public_corpus_df = data_loader.load_public_corpus(config.PUBLIC_CORPUS_PATH)
    
    public_passages = indexing.chunk_text(
        public_corpus_df['text'].tolist(),
        config.CHUNK_SIZE,
        config.CHUNK_OVERLAP
    )

    # --- 2. Build or Load Semantic Index ---
    indexer = indexing.SemanticIndexer(config.EMBEDDING_MODEL, config.EMBEDDING_DIM)
    if not os.path.exists(config.FAISS_INDEX_PATH):
        print("Building and saving a new FAISS index...")
        indexer.build_index(public_passages)
        indexer.save_index(config.FAISS_INDEX_PATH)
    else:
        print("Loading existing FAISS index...")
        indexer.load_index(config.FAISS_INDEX_PATH)

    # --- 3. Fine-tuning (PEFT LoRA) ---
    print("\n" + "-"*30 + " STAGE 3: FINE-TUNING " + "-"*30)
    
    # Use a subset for training demonstration
    train_subset = private_dataset.select(range(config.MAX_TRAIN_SAMPLES or len(private_dataset)))
    
    training_dataset = training.create_training_dataset(train_subset, public_passages, indexer)
    
    tuner = training.LoRAFineTuner(config.BASE_LLM_MODEL)
    tuner.setup_peft()
    tuner.train(training_dataset)

    # --- 4. Generate Synthetic Data ---
    print("\n" + "-"*30 + " STAGE 4: GENERATION " + "-"*30)
    generator = generation.SyntheticDataGenerator(config.BASE_LLM_MODEL, config.ADAPTER_PATH)
    
    # Use a subset for generation demonstration
    generation_subset = private_dataset.select(range(config.MAX_GENERATION_SAMPLES or len(private_dataset)))
    
    synthetic_data = generator.generate(generation_subset, public_passages, indexer)
    utils.save_synthetic_data(synthetic_data, config.SYNTHETIC_DATA_PATH)
    
    # --- 5. Evaluation ---
    print("\n" + "-"*30 + " STAGE 5: EVALUATION " + "-"*30)
    
    # Use a subset for evaluation
    eval_data = synthetic_data[:config.MAX_EVAL_SAMPLES]
    
    original_texts = [d['original_text'] for d in eval_data]
    synthetic_texts = [d['synthetic_text'] for d in eval_data]
    labels = [d['label'] for d in eval_data]

    # 5.1. Downstream Performance
    print("\n--- Evaluating Downstream Performance ---")
    # A proper eval would use a held-out test set from the original data
    # Here we train on synthetic and test on original for a proxy measure
    split_idx = int(len(labels) * 0.8)
    downstream_results = downstream_task.train_and_evaluate_classifier(
        train_texts=synthetic_texts[:split_idx], train_labels=labels[:split_idx],
        test_texts=original_texts[split_idx:], test_labels=labels[split_idx:]
    )
    print(f"  Downstream Accuracy on Original Data: {downstream_results['eval_accuracy']:.4f}")

    # 5.2. Data Quality Metrics
    print("\n--- Evaluating Synthetic Data Quality ---")
    semantic_sim = quality.calculate_semantic_similarity(original_texts, synthetic_texts)
    ttr = quality.calculate_ttr(synthetic_texts)
    self_bleu = quality.calculate_self_bleu(synthetic_texts)
    print(f"  Semantic Similarity (Orig vs. Synth): {semantic_sim:.4f}")
    print(f"  Type-Token Ratio (TTR): {ttr:.4f}")
    print(f"  Self-BLEU (Diversity): {self_bleu:.4f}")

    # 5.3. Privacy Leakage Evaluation
    print("\n--- Evaluating Privacy Leakage ---")
    exact_match = privacy.calculate_exact_match_ratio(original_texts, synthetic_texts)
    ngram_overlap = privacy.calculate_ngram_overlap(original_texts, synthetic_texts, n=5)
    print(f"  Exact Match Ratio: {exact_match:.2f}%")
    print(f"  5-gram Overlap Ratio: {ngram_overlap:.2f}%")
    
    print("\n" + "="*80)
    print("      PIPELINE COMPLETED SUCCESSFULLY      ")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()