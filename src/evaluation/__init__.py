"""
Evaluation Package — Quality, Privacy, and Statistical Analysis.

Provides comprehensive evaluation of synthetic data across three axes:
1. Quality: Semantic similarity, TTR, Self-BLEU
2. Privacy: Exact match, n-gram overlap, MIA (simple + shadow model)
3. Statistical: Bootstrap CI, paired t-tests, effect sizes
"""

from .quality import (
    calculate_self_bleu,
    calculate_ttr,
    calculate_semantic_similarity,
    measure_similarity_batch,
)
from .privacy import (
    calculate_exact_match_ratio,
    calculate_ngram_overlap,
    calculate_single_pair_ngram_overlap,
)
from .membership_inference import (
    MembershipInferenceAttack,
    run_mia_evaluation,
)
from .shadow_model_mia import ShadowModelMIA
from .statistical_tests import (
    bootstrap_confidence_interval,
    paired_t_test,
    cohens_d,
    compute_all_statistics,
)
from .results_table import (
    generate_latex_table,
    generate_markdown_table,
    load_experiment_results,
)

__all__ = [
    # Quality
    "calculate_self_bleu",
    "calculate_ttr",
    "calculate_semantic_similarity",
    "measure_similarity_batch",
    # Privacy
    "calculate_exact_match_ratio",
    "calculate_ngram_overlap",
    "calculate_single_pair_ngram_overlap",
    # MIA
    "MembershipInferenceAttack",
    "run_mia_evaluation",
    "ShadowModelMIA",
    # Statistics
    "bootstrap_confidence_interval",
    "paired_t_test",
    "cohens_d",
    "compute_all_statistics",
    # Results
    "generate_latex_table",
    "generate_markdown_table",
    "load_experiment_results",
]
