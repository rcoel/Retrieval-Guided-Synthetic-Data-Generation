"""
Results Table Generator — Publication-Ready Output.

Generates LaTeX and Markdown tables from experiment results,
with automatic bolding of best values and proper formatting.
"""

from __future__ import annotations
import json
import os
from typing import Dict, List, Optional, Tuple
from ..logger import get_logger

logger = get_logger("rag_pipeline.results_table")


# Metrics where HIGHER is better
HIGHER_IS_BETTER = {
    "semantic_similarity", "ttr", "downstream_accuracy",
    "accuracy", "eval_accuracy",
}

# Metrics where LOWER is better
LOWER_IS_BETTER = {
    "exact_match_pct", "ngram_overlap_pct", "self_bleu",
    "attack_success_rate", "auc_roc", "tpr_at_1pct_fpr",
    "advantage", "privacy_gap",
}


def load_experiment_results(results_dir: str) -> Dict[str, Dict]:
    """
    Load all experiment result JSONs from a directory.
    
    Args:
        results_dir: Path to directory containing result JSON files.
        
    Returns:
        {experiment_name: result_dict}
    """
    results = {}
    if not os.path.exists(results_dir):
        logger.warning(f"Results directory not found: {results_dir}")
        return results
    
    for fname in sorted(os.listdir(results_dir)):
        if fname.endswith(".json"):
            path = os.path.join(results_dir, fname)
            with open(path, 'r') as f:
                data = json.load(f)
            name = fname.replace(".json", "")
            results[name] = data
            logger.info(f"Loaded: {name}")
    
    return results


def generate_latex_table(
    results: Dict[str, Dict],
    metrics: List[str],
    caption: str = "Experimental Results",
    label: str = "tab:results",
) -> str:
    """
    Generate a publication-ready LaTeX table.
    
    Args:
        results: {method_name: {metric: value}}
        metrics: List of metric names to include as columns.
        caption: Table caption.
        label: LaTeX label.
        
    Returns:
        LaTeX table string.
    """
    methods = list(results.keys())
    
    # Find best value per metric
    best_vals = {}
    for metric in metrics:
        values = []
        for method in methods:
            val = _extract_metric(results[method], metric)
            if val is not None:
                values.append(val)
        
        if values:
            if metric in HIGHER_IS_BETTER:
                best_vals[metric] = max(values)
            elif metric in LOWER_IS_BETTER:
                best_vals[metric] = min(values)
            else:
                best_vals[metric] = max(values)  # Default: higher is better

    # Header
    col_str = "l" + "c" * len(metrics)
    header_cells = [_format_metric_name(m) for m in metrics]
    header = " & ".join(["Method"] + header_cells) + " \\\\"
    
    lines = [
        f"\\begin{{table}}[t]",
        f"\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{col_str}}}",
        "\\toprule",
        header,
        "\\midrule",
    ]

    for method in methods:
        cells = [_format_method_name(method)]
        for metric in metrics:
            val = _extract_metric(results[method], metric)
            if val is None:
                cells.append("--")
            else:
                formatted = f"{val:.4f}"
                if metric in best_vals and abs(val - best_vals[metric]) < 1e-6:
                    formatted = f"\\textbf{{{formatted}}}"
                cells.append(formatted)
        lines.append(" & ".join(cells) + " \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    table = "\n".join(lines)
    logger.info(f"Generated LaTeX table with {len(methods)} methods × {len(metrics)} metrics")
    return table


def generate_markdown_table(
    results: Dict[str, Dict],
    metrics: List[str],
) -> str:
    """
    Generate a Markdown table from experiment results.
    
    Args:
        results: {method_name: {metric: value}}
        metrics: Metric names for columns.
        
    Returns:
        Markdown table string.
    """
    methods = list(results.keys())
    
    # Find best values
    best_vals = {}
    for metric in metrics:
        values = [_extract_metric(results[m], metric) for m in methods]
        values = [v for v in values if v is not None]
        if values:
            if metric in LOWER_IS_BETTER:
                best_vals[metric] = min(values)
            else:
                best_vals[metric] = max(values)

    header_cells = ["Method"] + [_format_metric_name(m) for m in metrics]
    header = "| " + " | ".join(header_cells) + " |"
    sep = "| " + " | ".join(["---"] * len(header_cells)) + " |"
    
    rows = [header, sep]
    for method in methods:
        cells = [_format_method_name(method)]
        for metric in metrics:
            val = _extract_metric(results[method], metric)
            if val is None:
                cells.append("--")
            else:
                formatted = f"{val:.4f}"
                if metric in best_vals and abs(val - best_vals[metric]) < 1e-6:
                    formatted = f"**{formatted}**"
                cells.append(formatted)
        rows.append("| " + " | ".join(cells) + " |")
    
    return "\n".join(rows)


def _extract_metric(result: Dict, metric: str) -> Optional[float]:
    """Extract a metric value from nested result dict."""
    # Direct lookup
    if metric in result:
        val = result[metric]
        return float(val) if isinstance(val, (int, float)) else None
    
    # Search in sub-dictionaries
    for section in ["quality", "privacy", "downstream", "membership_inference",
                     "shadow_mia", "dp_audit", "generation"]:
        if section in result and isinstance(result[section], dict):
            if metric in result[section]:
                val = result[section][metric]
                return float(val) if isinstance(val, (int, float)) else None
    return None


def _format_metric_name(metric: str) -> str:
    """Format metric name for display."""
    names = {
        "semantic_similarity": "Sem-Sim ↑",
        "ttr": "TTR ↑",
        "self_bleu": "Self-BLEU ↓",
        "exact_match_pct": "Exact Match ↓",
        "ngram_overlap_pct": "5-gram ↓",
        "attack_success_rate": "MIA ASR ↓",
        "auc_roc": "MIA AUC ↓",
        "tpr_at_1pct_fpr": "TPR@1\\%FPR ↓",
        "advantage": "Advantage ↓",
        "downstream_accuracy": "Acc ↑",
        "eval_accuracy": "Acc ↑",
        "accuracy": "Acc ↑",
    }
    return names.get(metric, metric)


def _format_method_name(method: str) -> str:
    """Format method/config name for display."""
    names = {
        "full_pipeline": "Ours (Full)",
        "baseline_vanilla": "Vanilla",
        "ablation_no_dp": "w/o DP",
        "ablation_no_critic": "w/o Critic",
        "ablation_no_perplexity": "w/o Perplexity",
        "ablation_no_redteam": "w/o Red Team",
        "ablation_fixed_temp": "w/o Adaptive Temp",
    }
    return names.get(method, method)
