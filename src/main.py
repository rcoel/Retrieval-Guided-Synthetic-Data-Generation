"""
Main Pipeline Orchestrator — Enhanced Adaptive RAG Synthetic Data Generation.

Orchestrates the full pipeline with proper error handling and stage isolation:
1. Data Loading
2. FAISS Indexing
3. LoRA Fine-Tuning
4. Adaptive Agentic Generation (DP, Perplexity Gate, CoT Critic)
5. Comprehensive Evaluation (Quality + Privacy + MIA + DP Audit)
"""

from __future__ import annotations
import os
import json
import sys
from typing import Dict, Any, Optional

from src import config
from src.config import PipelineConfig
from src.dataloader import load_private_dataset, load_public_corpus, DataLoadError
from src.utils import setup_nltk, save_synthetic_data, set_seed
from src.logger import setup_logger, get_logger
from src.pipeline.indexing import SemanticIndexer, chunk_text
from src.pipeline import training, generation
from src.evaluation import quality, privacy, downstream_task
from src.evaluation.membership_inference import run_mia_evaluation

logger = get_logger("rag_pipeline.main")


def run_pipeline() -> Dict[str, Any]:
    """
    Run the entire RAG-based synthetic data pipeline.
    
    Returns:
        Dictionary of all collected metrics.
        
    Raises:
        DataLoadError: If data files are missing or malformed.
    """
    # --- 0. Setup ---
    setup_logger()
    setup_nltk()
    set_seed(config.SEED)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.MODEL_OUTPUT_DIR, exist_ok=True)
    
    all_metrics: Dict[str, Any] = {"config": PipelineConfig().to_dict()}
    
    logger.info("=" * 70)
    logger.info("  ENHANCED ADAPTIVE RAG — SYNTHETIC DATA GENERATION PIPELINE")
    logger.info("=" * 70)

    # --- 1. Load Data ---
    logger.info("STAGE 1: Loading Data")
    private_dataset = load_private_dataset(config.PRIVATE_DATA_NAME, config.PRIVATE_DATA_SUBSET)
    public_corpus_df = load_public_corpus(config.PUBLIC_CORPUS_PATH)
    
    public_passages = chunk_text(
        public_corpus_df['text'].tolist(),
        config.CHUNK_SIZE,
        config.CHUNK_OVERLAP,
    )

    # --- 2. Build or Load Semantic Index ---
    logger.info("STAGE 2: Semantic Indexing")
    indexer = SemanticIndexer(config.EMBEDDING_MODEL, config.EMBEDDING_DIM)
    if not os.path.exists(config.FAISS_INDEX_PATH):
        indexer.build_index(public_passages)
        indexer.save_index(config.FAISS_INDEX_PATH)
    else:
        indexer.load_index(config.FAISS_INDEX_PATH)

    # --- 3. Fine-tuning (PEFT LoRA) ---
    logger.info("STAGE 3: LoRA Fine-Tuning")
    try:
        train_subset = private_dataset.select(
            range(config.MAX_TRAIN_SAMPLES or len(private_dataset))
        )
        training_dataset = training.create_training_dataset(
            train_subset, public_passages, indexer
        )
        
        tuner = training.LoRAFineTuner(config.BASE_LLM_MODEL)
        tuner.setup_peft()
        tuner.train(training_dataset)
    except Exception as e:
        logger.error(f"Fine-tuning failed: {e}")
        logger.warning("Attempting to use existing LoRA adapter if available...")
        if not os.path.exists(config.ADAPTER_PATH):
            raise

    # --- 4. Generate Synthetic Data ---
    logger.info("STAGE 4: Enhanced Agentic Generation")
    generator = generation.SyntheticDataGenerator(
        config.BASE_LLM_MODEL, config.ADAPTER_PATH
    )
    
    gen_subset = private_dataset.select(
        range(config.MAX_GENERATION_SAMPLES or len(private_dataset))
    )
    synthetic_data = generator.generate(gen_subset, public_passages, indexer)
    save_synthetic_data(synthetic_data, config.SYNTHETIC_DATA_PATH)
    
    all_metrics["generation"] = generator.get_metrics()

    # --- 5. Evaluation ---
    logger.info("STAGE 5: Comprehensive Evaluation")
    eval_data = synthetic_data[:config.MAX_EVAL_SAMPLES]
    
    if not eval_data:
        logger.warning("No synthetic data to evaluate. Skipping evaluation.")
        return all_metrics

    original_texts = [d['original_text'] for d in eval_data]
    synthetic_texts = [d['synthetic_text'] for d in eval_data]
    labels = [d['label'] for d in eval_data]

    # 5.1 Downstream Performance
    all_metrics["downstream"] = _evaluate_downstream(
        synthetic_texts, original_texts, labels
    )

    # 5.2 Quality Metrics
    all_metrics["quality"] = _evaluate_quality(original_texts, synthetic_texts)

    # 5.3 Privacy Metrics
    all_metrics["privacy"] = _evaluate_privacy(original_texts, synthetic_texts)

    # 5.4 Membership Inference Attack
    all_metrics["membership_inference"] = _evaluate_mia(
        synthetic_texts, original_texts
    )

    # 5.5 DP Budget Audit
    dp_report = generator.get_dp_report()
    all_metrics["dp_audit"] = dp_report
    if dp_report.get("status") != "DP accounting disabled":
        logger.info(f"DP Budget: spent={dp_report['epsilon_spent']:.4f}, "
                     f"remaining={dp_report['budget_remaining']:.4f}")

    # --- 6. Save Metrics ---
    if config.ENABLE_METRICS_LOGGING:
        with open(config.METRICS_LOG_PATH, 'w', encoding='utf-8') as f:
            json.dump(all_metrics, f, indent=2, default=str)
        logger.info(f"Metrics saved to {config.METRICS_LOG_PATH}")

    logger.info("=" * 70)
    logger.info("  PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("=" * 70)

    return all_metrics


def _evaluate_downstream(
    synth_texts: list, orig_texts: list, labels: list,
) -> Dict[str, Any]:
    """Run downstream classifier evaluation."""
    logger.info("Evaluating downstream performance...")
    split_idx = int(len(labels) * 0.8)
    if split_idx > 0 and split_idx < len(labels):
        try:
            results = downstream_task.train_and_evaluate_classifier(
                train_texts=synth_texts[:split_idx],
                train_labels=labels[:split_idx],
                test_texts=orig_texts[split_idx:],
                test_labels=labels[split_idx:],
            )
            acc = results['eval_accuracy']
            logger.info(f"  Downstream Accuracy: {acc:.4f}")
            return {"accuracy": acc}
        except Exception as e:
            logger.error(f"  Downstream eval failed: {e}")
            return {"accuracy": "error", "error": str(e)}
    
    logger.warning("  Insufficient samples for downstream split")
    return {"accuracy": "N/A"}


def _evaluate_quality(
    orig_texts: list, synth_texts: list,
) -> Dict[str, float]:
    """Run quality metrics evaluation."""
    logger.info("Evaluating synthetic data quality...")
    sem_sim = quality.calculate_semantic_similarity(orig_texts, synth_texts)
    ttr = quality.calculate_ttr(synth_texts)
    self_bleu = quality.calculate_self_bleu(synth_texts)
    logger.info(f"  Sem-Sim={sem_sim:.4f}, TTR={ttr:.4f}, Self-BLEU={self_bleu:.4f}")
    return {"semantic_similarity": sem_sim, "ttr": ttr, "self_bleu": self_bleu}


def _evaluate_privacy(
    orig_texts: list, synth_texts: list,
) -> Dict[str, float]:
    """Run privacy leakage evaluation."""
    logger.info("Evaluating privacy leakage...")
    exact = privacy.calculate_exact_match_ratio(orig_texts, synth_texts)
    ngram = privacy.calculate_ngram_overlap(orig_texts, synth_texts, n=5)
    logger.info(f"  Exact Match: {exact:.2f}%, 5-gram Overlap: {ngram:.2f}%")
    return {"exact_match_pct": exact, "ngram_overlap_pct": ngram}


def _evaluate_mia(
    synth_texts: list, orig_texts: list,
) -> Dict[str, Any]:
    """Run membership inference attack evaluation."""
    logger.info("Running Membership Inference Attack...")
    n = len(orig_texts)
    if n >= 2:
        try:
            return run_mia_evaluation(
                synthetic_texts=synth_texts,
                member_texts=orig_texts[:n // 2],
                non_member_texts=orig_texts[n // 2:],
            )
        except Exception as e:
            logger.error(f"  MIA failed: {e}")
            return {"status": "error", "error": str(e)}
    
    logger.warning("  Insufficient samples for MIA")
    return {"status": "skipped"}


if __name__ == "__main__":
    try:
        run_pipeline()
    except DataLoadError as e:
        logger.error(f"Data loading failed: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)