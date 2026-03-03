#!/usr/bin/env python3
"""
Experiment Runner — CLI for Reproducible Experiments.

Usage:
    # Single experiment
    python run_experiment.py --config experiments/configs/full_pipeline.yaml --dataset sst2

    # Run all ablations
    python run_experiment.py --ablation all --dataset sst2 --seed 42

    # Dry run (small samples for testing)
    python run_experiment.py --config experiments/configs/full_pipeline.yaml --dry-run

    # Generate results table
    python run_experiment.py --results-table --results-dir experiments/results/
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import glob
import yaml
from typing import Dict, Any, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import PipelineConfig
from src import config as config_module
from src.utils import setup_nltk, save_synthetic_data, set_seed
from src.logger import setup_logger, get_logger
from src.dataset_registry import (
    load_registered_dataset,
    get_dataset_spec,
    generate_public_corpus,
    list_datasets,
)
from src.pipeline.indexing import SemanticIndexer, chunk_text
from src.pipeline import training, generation
from src.evaluation import quality, privacy, downstream_task
from src.evaluation.membership_inference import run_mia_evaluation
from src.evaluation.shadow_model_mia import ShadowModelMIA
from src.evaluation.results_table import (
    load_experiment_results,
    generate_latex_table,
    generate_markdown_table,
)

logger = get_logger("rag_pipeline.experiment")


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Load experiment config from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def apply_config_overrides(yaml_config: Dict[str, Any], args: argparse.Namespace) -> None:
    """Apply YAML config and CLI args to the module-level config."""
    cfg = PipelineConfig()
    
    # Map YAML keys to config attributes (case-insensitive)
    for key, value in yaml_config.items():
        attr = key.upper()
        if hasattr(cfg, attr) and value is not None:
            setattr(cfg, attr, value)
    
    # CLI overrides take priority
    if args.dataset:
        yaml_config["dataset"] = args.dataset
    if args.seed:
        cfg.SEED = args.seed
    if args.dry_run:
        cfg.MAX_TRAIN_SAMPLES = 4
        cfg.MAX_GENERATION_SAMPLES = 4
        cfg.MAX_EVAL_SAMPLES = 4
    
    # Apply to module-level config
    for field_name in cfg.__dataclass_fields__:
        setattr(config_module, field_name, getattr(cfg, field_name))


def run_single_experiment(
    config_path: str,
    dataset_key: str,
    seed: int,
    dry_run: bool = False,
    output_dir: str = "experiments/results",
) -> Dict[str, Any]:
    """
    Run a single experiment with the given config.
    
    Args:
        config_path: Path to YAML config.
        dataset_key: Dataset registry key.
        seed: Random seed.
        dry_run: If True, use tiny sample sizes.
        output_dir: Where to save results.
        
    Returns:
        Dictionary of all metrics.
    """
    config_name = os.path.splitext(os.path.basename(config_path))[0]
    logger.info(f"{'='*60}")
    logger.info(f"  EXPERIMENT: {config_name}")
    logger.info(f"  Dataset: {dataset_key} | Seed: {seed}")
    logger.info(f"{'='*60}")
    
    # Load config
    yaml_config = load_yaml_config(config_path)
    
    # Create argparse-like namespace for apply_config_overrides
    args = argparse.Namespace(
        dataset=dataset_key, seed=seed, dry_run=dry_run,
    )
    apply_config_overrides(yaml_config, args)
    
    # Setup
    setup_logger()
    setup_nltk()
    set_seed(seed)
    os.makedirs(config_module.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config_module.MODEL_OUTPUT_DIR, exist_ok=True)
    
    all_metrics: Dict[str, Any] = {
        "config_name": config_name,
        "dataset": dataset_key,
        "seed": seed,
        "config": yaml_config,
    }
    
    # --- 1. Load Dataset ---
    logger.info("Stage 1: Loading dataset from registry")
    dataset, spec = load_registered_dataset(
        dataset_key,
        max_samples=config_module.MAX_TRAIN_SAMPLES,
    )
    
    # Load or generate public corpus
    corpus_path = config_module.PUBLIC_CORPUS_PATH
    if not os.path.exists(corpus_path):
        logger.info("Generating domain-matched public corpus...")
        generate_public_corpus(dataset_key, corpus_path)
    
    import pandas as pd
    public_df = pd.read_csv(corpus_path)
    public_passages = chunk_text(
        public_df['text'].tolist(),
        config_module.CHUNK_SIZE,
        config_module.CHUNK_OVERLAP,
    )
    
    # --- 2. Build Index ---
    logger.info("Stage 2: Semantic Indexing")
    indexer = SemanticIndexer(config_module.EMBEDDING_MODEL, config_module.EMBEDDING_DIM)
    if not os.path.exists(config_module.FAISS_INDEX_PATH):
        indexer.build_index(public_passages)
        indexer.save_index(config_module.FAISS_INDEX_PATH)
    else:
        indexer.load_index(config_module.FAISS_INDEX_PATH)
    
    # --- 3. Fine-tuning ---
    logger.info("Stage 3: LoRA Fine-Tuning")
    try:
        training_dataset = training.create_training_dataset(
            dataset, public_passages, indexer
        )
        tuner = training.LoRAFineTuner(config_module.BASE_LLM_MODEL)
        tuner.setup_peft()
        tuner.train(training_dataset)
    except Exception as e:
        logger.error(f"Fine-tuning failed: {e}")
        if not os.path.exists(config_module.ADAPTER_PATH):
            raise
    
    # --- 4. Generate ---
    logger.info("Stage 4: Agentic Generation")
    gen = generation.SyntheticDataGenerator(
        config_module.BASE_LLM_MODEL, config_module.ADAPTER_PATH
    )
    synthetic_data = gen.generate(dataset, public_passages, indexer)
    save_synthetic_data(synthetic_data, config_module.SYNTHETIC_DATA_PATH)
    all_metrics["generation"] = gen.get_metrics()
    
    # --- 5. Evaluate ---
    logger.info("Stage 5: Evaluation")
    eval_data = synthetic_data[:config_module.MAX_EVAL_SAMPLES]
    
    if eval_data:
        orig = [d['original_text'] for d in eval_data]
        synth = [d['synthetic_text'] for d in eval_data]
        labels = [d['label'] for d in eval_data]
        
        # Quality
        all_metrics["quality"] = {
            "semantic_similarity": quality.calculate_semantic_similarity(orig, synth),
            "ttr": quality.calculate_ttr(synth),
            "self_bleu": quality.calculate_self_bleu(synth),
        }
        
        # Privacy
        all_metrics["privacy"] = {
            "exact_match_pct": privacy.calculate_exact_match_ratio(orig, synth),
            "ngram_overlap_pct": privacy.calculate_ngram_overlap(orig, synth, n=5),
        }
        
        # Simple MIA
        n = len(orig)
        if n >= 4:
            all_metrics["simple_mia"] = run_mia_evaluation(
                synth, orig[:n//2], orig[n//2:]
            )
        
        # Shadow Model MIA
        if n >= 4:
            shadow_mia = ShadowModelMIA()
            all_metrics["shadow_mia"] = shadow_mia.run_attack(
                synth, orig[:n//2], orig[n//2:]
            )
        
        # Downstream
        split_idx = int(len(labels) * 0.8)
        if split_idx > 0 and split_idx < len(labels):
            try:
                result = downstream_task.train_and_evaluate_classifier(
                    synth[:split_idx], labels[:split_idx],
                    orig[split_idx:], labels[split_idx:],
                )
                all_metrics["downstream"] = {"accuracy": result.get("eval_accuracy")}
            except Exception as e:
                logger.error(f"Downstream eval failed: {e}")
                all_metrics["downstream"] = {"accuracy": "error"}
        
        # DP Audit
        all_metrics["dp_audit"] = gen.get_dp_report()
    
    # --- 6. Save Results ---
    os.makedirs(output_dir, exist_ok=True)
    result_path = os.path.join(output_dir, f"{config_name}_{dataset_key}_s{seed}.json")
    with open(result_path, 'w') as f:
        json.dump(all_metrics, f, indent=2, default=str)
    logger.info(f"Results saved: {result_path}")
    
    return all_metrics


def run_ablation_suite(
    dataset_key: str,
    seed: int,
    configs_dir: str = "experiments/configs",
    dry_run: bool = False,
) -> Dict[str, Dict]:
    """Run all configs in the configs directory."""
    config_files = sorted(glob.glob(os.path.join(configs_dir, "*.yaml")))
    logger.info(f"Running ablation suite: {len(config_files)} configs on {dataset_key}")
    
    all_results = {}
    for config_path in config_files:
        name = os.path.splitext(os.path.basename(config_path))[0]
        try:
            results = run_single_experiment(
                config_path, dataset_key, seed, dry_run=dry_run,
            )
            all_results[name] = results
        except Exception as e:
            logger.error(f"Experiment {name} failed: {e}")
            all_results[name] = {"error": str(e)}
    
    return all_results


def generate_tables(results_dir: str) -> None:
    """Generate publication-ready tables from saved results."""
    results = load_experiment_results(results_dir)
    if not results:
        logger.warning("No results found to tabulate")
        return
    
    metrics = [
        "semantic_similarity", "ttr", "self_bleu",
        "exact_match_pct", "ngram_overlap_pct",
        "attack_success_rate", "auc_roc",
    ]
    
    # LaTeX
    latex = generate_latex_table(results, metrics)
    latex_path = os.path.join(results_dir, "results_table.tex")
    with open(latex_path, 'w') as f:
        f.write(latex)
    logger.info(f"LaTeX table saved: {latex_path}")
    
    # Markdown
    md = generate_markdown_table(results, metrics)
    md_path = os.path.join(results_dir, "results_table.md")
    with open(md_path, 'w') as f:
        f.write(md)
    logger.info(f"Markdown table saved: {md_path}")
    
    print("\n" + md)


def main():
    parser = argparse.ArgumentParser(
        description="PrivaSyn — Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python run_experiment.py --config experiments/configs/full_pipeline.yaml --dataset sst2
  python run_experiment.py --ablation all --dataset ag_news --seed 42
  python run_experiment.py --dry-run --config experiments/configs/full_pipeline.yaml
  python run_experiment.py --results-table --results-dir experiments/results/
  python run_experiment.py --list-datasets
"""
    )
    
    parser.add_argument("--config", type=str, help="Path to YAML experiment config")
    parser.add_argument("--dataset", type=str, default="sst2",
                       choices=list_datasets(),
                       help="Dataset key from registry")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dry-run", action="store_true",
                       help="Use tiny sample sizes for testing")
    parser.add_argument("--ablation", type=str, choices=["all"],
                       help="Run all ablation configs")
    parser.add_argument("--results-table", action="store_true",
                       help="Generate results table from saved results")
    parser.add_argument("--results-dir", type=str, default="experiments/results",
                       help="Directory for results (default: experiments/results/)")
    parser.add_argument("--list-datasets", action="store_true",
                       help="List available datasets")
    
    args = parser.parse_args()
    
    setup_logger()
    
    if args.list_datasets:
        for key in list_datasets():
            spec = get_dataset_spec(key)
            print(f"  {key:12s}  {spec.description}")
        return
    
    if args.results_table:
        generate_tables(args.results_dir)
        return
    
    if args.ablation:
        run_ablation_suite(args.dataset, args.seed, dry_run=args.dry_run)
        generate_tables(args.results_dir)
        return
    
    if args.config:
        run_single_experiment(
            args.config, args.dataset, args.seed, dry_run=args.dry_run,
        )
        return
    
    parser.print_help()


if __name__ == "__main__":
    main()
