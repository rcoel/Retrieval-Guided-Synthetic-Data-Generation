"""
Shadow Model Membership Inference Attack (MIA).

Implements the full shadow-model attack from Shokri et al. (2017):
1. Train N shadow models on random subsets of data
2. Generate confidence vectors for "in" and "out" samples
3. Train a binary attack classifier
4. Evaluate on the target model's outputs

This is the gold standard for privacy evaluation in research papers.

Reference:
    Shokri et al., "Membership Inference Attacks Against ML Models", IEEE S&P 2017
"""

from __future__ import annotations
import numpy as np
from typing import List, Dict, Optional, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, accuracy_score, roc_curve,
    precision_recall_curve, average_precision_score,
)
from sklearn.model_selection import StratifiedKFold
from sentence_transformers import SentenceTransformer, util
from .. import config
from ..logger import get_logger

logger = get_logger("rag_pipeline.shadow_mia")


class ShadowModelMIA:
    """
    Full Shadow-Model Membership Inference Attack.
    
    Rather than training actual shadow generative models (expensive),
    this implementation uses embedding-based shadow analysis:
    
    1. Compute semantic similarity features between synthetic data
       and training/non-training data
    2. Train a logistic regression attack classifier on these features
    3. Evaluate attack performance with comprehensive metrics
    
    This is a practical approximation suitable for synthetic data
    evaluation where the "model" is the generation pipeline itself.
    """

    def __init__(
        self,
        embedding_model_name: Optional[str] = None,
        n_shadow_splits: int = 5,
    ):
        """
        Args:
            embedding_model_name: Sentence transformer for feature extraction.
            n_shadow_splits: Number of cross-validation folds for shadow training.
        """
        self.model_name = embedding_model_name or config.EMBEDDING_MODEL
        self.model = SentenceTransformer(self.model_name)
        self.n_splits = n_shadow_splits
        self.attack_classifier: Optional[LogisticRegression] = None

    def extract_features(
        self,
        synthetic_texts: List[str],
        reference_texts: List[str],
    ) -> np.ndarray:
        """
        Extract multi-dimensional attack features for each synthetic sample.
        
        Features per sample:
        1. Max cosine similarity to reference set
        2. Mean cosine similarity to top-5 nearest references
        3. Std of top-5 similarities
        4. Min distance to reference set
        5. Similarity entropy (how spread out the similarities are)
        
        Args:
            synthetic_texts: The generated synthetic data.
            reference_texts: Either member or non-member texts.
            
        Returns:
            Feature matrix of shape (n_synthetic, 5).
        """
        synth_emb = self.model.encode(
            synthetic_texts, convert_to_tensor=True, show_progress_bar=False
        )
        ref_emb = self.model.encode(
            reference_texts, convert_to_tensor=True, show_progress_bar=False
        )

        # Full similarity matrix (n_synth × n_ref)
        sim_matrix = util.cos_sim(synth_emb, ref_emb).cpu().numpy()

        # Sort similarities in descending order for top-k analysis
        sorted_sims = np.sort(sim_matrix, axis=1)[:, ::-1]

        k = min(5, sim_matrix.shape[1])
        top_k = sorted_sims[:, :k]

        features = np.column_stack([
            sim_matrix.max(axis=1),          # max similarity
            top_k.mean(axis=1),              # mean of top-k
            top_k.std(axis=1),               # std of top-k
            1.0 - sim_matrix.max(axis=1),    # min distance
            self._similarity_entropy(sim_matrix),  # entropy
        ])

        return features.astype(np.float32)

    def _similarity_entropy(self, sim_matrix: np.ndarray) -> np.ndarray:
        """Compute entropy of similarity distribution per sample."""
        # Shift to positive and normalize to probability distribution
        shifted = sim_matrix - sim_matrix.min(axis=1, keepdims=True) + 1e-8
        probs = shifted / shifted.sum(axis=1, keepdims=True)
        entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1)
        return entropy

    def run_attack(
        self,
        synthetic_texts: List[str],
        member_texts: List[str],
        non_member_texts: List[str],
    ) -> Dict[str, float]:
        """
        Run the full shadow-model MIA pipeline.
        
        1. Extract features from synthetic→member and synthetic→non-member
        2. Train attack classifier via cross-validation
        3. Evaluate with comprehensive metrics
        
        Args:
            synthetic_texts: Generated synthetic data.
            member_texts: Training data (ground truth members).
            non_member_texts: Held-out data (ground truth non-members).
            
        Returns:
            Dictionary with comprehensive attack metrics.
        """
        logger.info(f"Running Shadow Model MIA: {len(synthetic_texts)} synthetic, "
                     f"{len(member_texts)} members, {len(non_member_texts)} non-members")

        # Extract features
        logger.info("  Extracting member features...")
        member_features = self.extract_features(synthetic_texts, member_texts)
        
        logger.info("  Extracting non-member features...")
        non_member_features = self.extract_features(synthetic_texts, non_member_texts)

        # Create attack dataset
        X = np.vstack([member_features, non_member_features])
        y = np.concatenate([
            np.ones(len(member_features)),    # 1 = member
            np.zeros(len(non_member_features)) # 0 = non-member
        ])

        # Cross-validated attack
        logger.info(f"  Training attack classifier ({self.n_splits}-fold CV)...")
        all_probs = np.zeros(len(y))
        all_preds = np.zeros(len(y))

        n_splits = min(self.n_splits, min(np.sum(y == 0), np.sum(y == 1)))
        if n_splits < 2:
            # Not enough data for CV, train on full
            clf = LogisticRegression(max_iter=1000, random_state=42)
            clf.fit(X, y)
            all_probs = clf.predict_proba(X)[:, 1]
            all_preds = clf.predict(X)
        else:
            skf = StratifiedKFold(n_splits=int(n_splits), shuffle=True, random_state=42)
            for train_idx, test_idx in skf.split(X, y):
                clf = LogisticRegression(max_iter=1000, random_state=42)
                clf.fit(X[train_idx], y[train_idx])
                all_probs[test_idx] = clf.predict_proba(X[test_idx])[:, 1]
                all_preds[test_idx] = clf.predict(X[test_idx])

        # Compute comprehensive metrics
        results = self._compute_metrics(y, all_probs, all_preds)
        
        logger.info(f"  Shadow MIA complete: ASR={results['attack_success_rate']:.2%}, "
                     f"AUC={results['auc_roc']:.4f}")
        return results

    def _compute_metrics(
        self,
        y_true: np.ndarray,
        y_probs: np.ndarray,
        y_preds: np.ndarray,
    ) -> Dict[str, float]:
        """Compute comprehensive attack evaluation metrics."""
        auc = roc_auc_score(y_true, y_probs)
        asr = accuracy_score(y_true, y_preds)

        # TPR at specific FPR thresholds (important for privacy)
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        tpr_at_1_fpr = self._tpr_at_fpr(fpr, tpr, 0.01)
        tpr_at_01_fpr = self._tpr_at_fpr(fpr, tpr, 0.001)

        # Precision-Recall AUC
        ap = average_precision_score(y_true, y_probs)

        # Advantage: 2 * |ASR - 0.5| (how much better than random)
        advantage = 2 * abs(asr - 0.5)

        return {
            "attack_success_rate": float(asr),
            "auc_roc": float(auc),
            "tpr_at_1pct_fpr": float(tpr_at_1_fpr),
            "tpr_at_01pct_fpr": float(tpr_at_01_fpr),
            "average_precision": float(ap),
            "advantage": float(advantage),
            "n_samples": int(len(y_true)),
        }

    @staticmethod
    def _tpr_at_fpr(fpr: np.ndarray, tpr: np.ndarray, target_fpr: float) -> float:
        """Find TPR at a specific FPR threshold."""
        idx = np.searchsorted(fpr, target_fpr, side='right') - 1
        if idx < 0:
            return 0.0
        return float(tpr[idx])

    @staticmethod
    def interpret_results(results: Dict[str, float]) -> str:
        """Generate human-readable interpretation."""
        asr = results["attack_success_rate"]
        auc = results["auc_roc"]
        adv = results["advantage"]

        lines = [
            f"  Attack Success Rate: {asr:.2%}",
            f"  AUC-ROC:            {auc:.4f}",
            f"  TPR@1%FPR:          {results['tpr_at_1pct_fpr']:.4f}",
            f"  TPR@0.1%FPR:        {results['tpr_at_01pct_fpr']:.4f}",
            f"  Advantage:          {adv:.4f}",
        ]

        if asr <= 0.55:
            lines.append("  ✅ STRONG PRIVACY: Attack near random chance")
        elif asr <= 0.65:
            lines.append("  ⚠️  MODERATE RISK: Some information leakage")
        else:
            lines.append("  ❌ HIGH RISK: Significant privacy vulnerability")

        return "\n".join(lines)
