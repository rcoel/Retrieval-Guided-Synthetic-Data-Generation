"""
Membership Inference Attack (MIA) Evaluation Module.

Implements a loss-based membership inference attack to quantitatively
assess whether synthetic data leaks information about training samples.
This is a standard requirement for privacy-related publications.

Reference:
    Shokri et al., "Membership Inference Attacks Against ML Models", IEEE S&P 2017
    Yeom et al., "Privacy Risk in Machine Learning", CSF 2018
"""

from __future__ import annotations
import numpy as np
from typing import List, Dict, Optional
from sklearn.metrics import roc_auc_score, accuracy_score
from sentence_transformers import SentenceTransformer, util
from .. import config
from ..logger import get_logger

logger = get_logger("rag_pipeline.mia")


class MembershipInferenceAttack:
    """
    Loss-based Membership Inference Attack.
    
    Intuition: If a synthetic sample is "too close" to a specific training
    sample, an attacker can infer that the training sample was in the 
    original dataset. We measure this via semantic similarity thresholding.
    """

    def __init__(self, embedding_model_name: Optional[str] = None):
        """
        Args:
            embedding_model_name: Sentence transformer model for similarity.
        """
        self.model_name = embedding_model_name or config.EMBEDDING_MODEL
        self.model = SentenceTransformer(self.model_name)

    def compute_attack_scores(
        self,
        synthetic_texts: List[str],
        member_texts: List[str],
        non_member_texts: List[str],
    ) -> Dict[str, float]:
        """
        Perform the membership inference attack.
        
        For each synthetic sample, compute max similarity to members vs
        non-members. If the model memorized training data, synthetic
        samples will be much closer to members.
        
        Args:
            synthetic_texts: Generated synthetic data.
            member_texts: Original training data (ground truth members).
            non_member_texts: Held-out data not used in training.
            
        Returns:
            Dictionary with ASR, AUC-ROC, privacy gap, and similarity stats.
        """
        logger.info("Running MIA: encoding texts...")
        synth_emb = self.model.encode(
            synthetic_texts, convert_to_tensor=True, show_progress_bar=False
        )
        member_emb = self.model.encode(
            member_texts, convert_to_tensor=True, show_progress_bar=False
        )
        non_member_emb = self.model.encode(
            non_member_texts, convert_to_tensor=True, show_progress_bar=False
        )

        # Max similarity to members vs non-members
        member_sims = util.cos_sim(synth_emb, member_emb)
        non_member_sims = util.cos_sim(synth_emb, non_member_emb)

        max_member_sim = member_sims.max(dim=1).values.cpu().numpy()
        max_non_member_sim = non_member_sims.max(dim=1).values.cpu().numpy()

        n_synth = len(synthetic_texts)
        
        # Binary classification: member scores vs non-member scores
        all_scores = np.concatenate([max_member_sim, max_non_member_sim])
        all_labels = np.concatenate([np.ones(n_synth), np.zeros(n_synth)])

        auc_roc = roc_auc_score(all_labels, all_scores)

        threshold = np.median(all_scores)
        predictions = (all_scores >= threshold).astype(int)
        asr = accuracy_score(all_labels, predictions)

        privacy_gap = float(np.mean(max_member_sim) - np.mean(max_non_member_sim))

        results = {
            "attack_success_rate": float(asr),
            "auc_roc": float(auc_roc),
            "privacy_gap": privacy_gap,
            "mean_member_similarity": float(np.mean(max_member_sim)),
            "mean_non_member_similarity": float(np.mean(max_non_member_sim)),
            "threshold_used": float(threshold),
        }

        logger.info(f"MIA results: ASR={asr:.2%}, AUC={auc_roc:.4f}, gap={privacy_gap:.4f}")
        return results

    @staticmethod
    def interpret_results(results: Dict[str, float]) -> str:
        """
        Generate a human-readable interpretation of MIA results.
        
        Guidelines:
            ASR ≈ 50% → Ideal (random guessing)
            ASR > 60% → Concerning
            ASR > 70% → Significant leakage
        """
        asr = results["attack_success_rate"]
        auc = results["auc_roc"]
        gap = results["privacy_gap"]

        lines = [
            f"  Attack Success Rate: {asr:.2%}",
            f"  AUC-ROC: {auc:.4f}",
            f"  Privacy Gap: {gap:.4f}",
        ]

        if asr <= 0.55:
            lines.append("  ✅ STRONG PRIVACY: Attack near random chance")
        elif asr <= 0.65:
            lines.append("  ⚠️  MODERATE RISK: Some information leakage detected")
        else:
            lines.append("  ❌ HIGH RISK: Significant membership inference vulnerability")

        if auc <= 0.55:
            lines.append("  ✅ AUC near 0.5: Hard to distinguish members from non-members")
        elif auc <= 0.7:
            lines.append("  ⚠️  AUC suggests partial distinguishability")
        else:
            lines.append("  ❌ AUC > 0.7: Attacker can reliably identify training data")

        return "\n".join(lines)


def run_mia_evaluation(
    synthetic_texts: List[str],
    member_texts: List[str],
    non_member_texts: List[str],
    embedding_model: Optional[str] = None,
) -> Dict[str, float]:
    """
    Convenience function to run the full MIA evaluation.
    
    Args:
        synthetic_texts: Generated synthetic samples.
        member_texts: Training data used to generate synthetic data.
        non_member_texts: Held-out data NOT used in generation.
        embedding_model: Optional model name override.
        
    Returns:
        Dictionary of MIA metrics.
    """
    attacker = MembershipInferenceAttack(embedding_model)
    results = attacker.compute_attack_scores(
        synthetic_texts, member_texts, non_member_texts
    )
    
    logger.info("MIA Evaluation Complete:\n" + MembershipInferenceAttack.interpret_results(results))
    return results
