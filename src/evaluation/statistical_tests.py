"""
Statistical Significance Testing for Experiment Results.

Provides bootstrap confidence intervals, paired t-tests, and
effect size calculations for rigorous comparison of methods.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats
from ..logger import get_logger

logger = get_logger("rag_pipeline.stats")


def bootstrap_confidence_interval(
    scores: List[float],
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    statistic: str = "mean",
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a metric.
    
    Args:
        scores: List of metric values (e.g., from multiple runs or samples).
        n_bootstrap: Number of bootstrap iterations.
        confidence_level: CI level (default 0.95 for 95%).
        statistic: "mean" or "median".
        seed: Random seed for reproducibility.
        
    Returns:
        Tuple of (point_estimate, ci_lower, ci_upper).
    """
    rng = np.random.RandomState(seed)
    arr = np.array(scores)
    n = len(arr)
    
    stat_fn = np.mean if statistic == "mean" else np.median
    point = stat_fn(arr)
    
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        sample = rng.choice(arr, size=n, replace=True)
        bootstrap_stats.append(stat_fn(sample))
    
    bootstrap_stats = np.array(bootstrap_stats)
    alpha = (1 - confidence_level) / 2
    ci_lower = float(np.percentile(bootstrap_stats, alpha * 100))
    ci_upper = float(np.percentile(bootstrap_stats, (1 - alpha) * 100))
    
    return float(point), ci_lower, ci_upper


def paired_t_test(
    scores_a: List[float],
    scores_b: List[float],
) -> Dict[str, float]:
    """
    Paired t-test between two methods' scores.
    
    Args:
        scores_a: Metric values from method A.
        scores_b: Metric values from method B.
        
    Returns:
        Dictionary with t-statistic, p-value, and significance flag.
    """
    if len(scores_a) != len(scores_b):
        raise ValueError("Score lists must have equal length for paired test")
    
    t_stat, p_value = stats.ttest_rel(scores_a, scores_b)
    
    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "significant_at_005": p_value < 0.05,
        "significant_at_001": p_value < 0.01,
    }


def cohens_d(
    scores_a: List[float],
    scores_b: List[float],
) -> float:
    """
    Compute Cohen's d effect size between two groups.
    
    Interpretation:
        |d| < 0.2: negligible
        0.2 ≤ |d| < 0.5: small
        0.5 ≤ |d| < 0.8: medium
        |d| ≥ 0.8: large
    """
    a, b = np.array(scores_a), np.array(scores_b)
    pooled_std = np.sqrt((a.std()**2 + b.std()**2) / 2)
    if pooled_std == 0:
        return 0.0
    return float((a.mean() - b.mean()) / pooled_std)


def compute_all_statistics(
    baseline_scores: Dict[str, List[float]],
    method_scores: Dict[str, List[float]],
    method_name: str = "Ours",
) -> Dict[str, Dict]:
    """
    Compute comprehensive statistics comparing a method against baseline.
    
    Args:
        baseline_scores: {metric_name: [score1, score2, ...]} for baseline.
        method_scores: {metric_name: [score1, score2, ...]} for our method.
        method_name: Name for reporting.
        
    Returns:
        {metric_name: {mean, ci_lower, ci_upper, p_value, cohens_d, ...}}
    """
    results = {}
    
    for metric in method_scores:
        ours = method_scores[metric]
        mean, ci_low, ci_high = bootstrap_confidence_interval(ours)
        
        entry = {
            "mean": mean,
            "ci_lower": ci_low,
            "ci_upper": ci_high,
            "ci_str": f"{mean:.4f} [{ci_low:.4f}, {ci_high:.4f}]",
        }
        
        # If baseline exists for this metric, add comparison
        if metric in baseline_scores and len(baseline_scores[metric]) == len(ours):
            base = baseline_scores[metric]
            base_mean, base_ci_low, base_ci_high = bootstrap_confidence_interval(base)
            
            entry["baseline_mean"] = base_mean
            entry["baseline_ci_str"] = f"{base_mean:.4f} [{base_ci_low:.4f}, {base_ci_high:.4f}]"
            entry["improvement"] = mean - base_mean
            entry["improvement_pct"] = ((mean - base_mean) / abs(base_mean) * 100) if base_mean != 0 else 0.0
            
            t_result = paired_t_test(base, ours)
            entry.update(t_result)
            entry["effect_size"] = cohens_d(base, ours)
        
        results[metric] = entry
        logger.info(f"  {metric}: {entry['ci_str']}")
    
    return results
