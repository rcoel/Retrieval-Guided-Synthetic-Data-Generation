"""
Privacy Budget Manager with Rényi Differential Privacy Accounting.

Tracks cumulative privacy expenditure across all queries and generation
steps, providing formal DP guarantees rather than ad-hoc noise injection.
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
from . import config


@dataclass
class PrivacyLedgerEntry:
    """A single record of privacy consumption."""
    step_name: str
    epsilon_consumed: float
    alpha: float
    cumulative_epsilon: float


class RenyiDPAccountant:
    """
    Rényi Differential Privacy (RDP) accountant for tracking privacy budget.
    
    Uses Rényi divergence of order α to tightly compose DP guarantees
    across multiple queries. Converts RDP guarantees to (ε, δ)-DP.
    
    Reference:
        Mironov, "Rényi Differential Privacy", CSF 2017
        Balle et al., "Hypothesis Testing Interpretations...", AISTATS 2020
    """

    def __init__(self, total_budget: float, delta: float = 1e-5, alpha: float = 5.0):
        """
        Args:
            total_budget: Maximum ε allowed before halting.
            delta: Failure probability for (ε, δ)-DP conversion.
            alpha: Rényi divergence order (α > 1). Higher α → tighter for Gaussian.
        """
        self.total_budget = total_budget
        self.delta = delta
        self.alpha = alpha
        self.rdp_consumed = 0.0  # cumulative RDP (Rényi divergence)
        self.ledger: List[PrivacyLedgerEntry] = []

    @property
    def epsilon_spent(self) -> float:
        """Convert accumulated RDP to (ε, δ)-DP."""
        return self._rdp_to_eps_delta(self.rdp_consumed, self.alpha, self.delta)

    @property
    def budget_remaining(self) -> float:
        return max(0.0, self.total_budget - self.epsilon_spent)

    def can_query(self, epsilon_step: Optional[float] = None) -> bool:
        """Check if a query can be made without exceeding the budget."""
        if epsilon_step is not None:
            # Simulate adding this step
            projected = self._rdp_to_eps_delta(
                self.rdp_consumed + self._eps_to_rdp(epsilon_step, self.alpha),
                self.alpha,
                self.delta
            )
            return projected <= self.total_budget
        return self.epsilon_spent < self.total_budget

    def consume(self, epsilon_step: float, step_name: str = "query") -> float:
        """
        Record privacy consumption for one step.
        
        Uses RDP composition: RDP values add linearly.
        
        Args:
            epsilon_step: The ε consumed by this single step.
            step_name: Human-readable label for the ledger.
            
        Returns:
            Current cumulative ε after this step.
            
        Raises:
            PrivacyBudgetExhaustedError: If budget would be exceeded.
        """
        rdp_step = self._eps_to_rdp(epsilon_step, self.alpha)
        projected_epsilon = self._rdp_to_eps_delta(
            self.rdp_consumed + rdp_step, self.alpha, self.delta
        )

        if projected_epsilon > self.total_budget:
            raise PrivacyBudgetExhaustedError(
                f"Budget exhausted: consuming ε={epsilon_step:.4f} would bring "
                f"total to {projected_epsilon:.4f} > {self.total_budget:.4f}"
            )

        self.rdp_consumed += rdp_step
        current_eps = self.epsilon_spent

        self.ledger.append(PrivacyLedgerEntry(
            step_name=step_name,
            epsilon_consumed=epsilon_step,
            alpha=self.alpha,
            cumulative_epsilon=current_eps
        ))

        return current_eps

    def get_report(self) -> dict:
        """Generate a summary report of privacy expenditure."""
        return {
            "total_budget": self.total_budget,
            "epsilon_spent": self.epsilon_spent,
            "budget_remaining": self.budget_remaining,
            "delta": self.delta,
            "alpha": self.alpha,
            "num_queries": len(self.ledger),
            "is_exhausted": not self.can_query(),
            "ledger_summary": [
                {"step": e.step_name, "eps": e.epsilon_consumed, "cumulative": e.cumulative_epsilon}
                for e in self.ledger[-10:]  # Last 10 entries
            ]
        }

    @staticmethod
    def _eps_to_rdp(epsilon: float, alpha: float) -> float:
        """
        Convert pure ε-DP to RDP of order α.
        RDP(α) ≤ ε for pure ε-DP mechanisms.
        For Laplace mechanism: RDP(α) = (1/(α-1)) * log((α-1)/(2α-1) * exp((α-1)*ε) + α/(2α-1) * exp(-(α)*ε))
        Simplified: for small ε, RDP ≈ α * ε² / 2
        """
        if alpha <= 1:
            return epsilon
        # Tight RDP bound for Laplace mechanism
        if epsilon == 0:
            return 0.0
        term1 = (alpha - 1) / (2 * alpha - 1) * math.exp((alpha - 1) * epsilon)
        term2 = alpha / (2 * alpha - 1) * math.exp(-alpha * epsilon)
        if term1 + term2 <= 0:
            return epsilon  # Fallback for numerical issues
        return max(0.0, (1.0 / (alpha - 1)) * math.log(term1 + term2))

    @staticmethod
    def _rdp_to_eps_delta(rdp: float, alpha: float, delta: float) -> float:
        """
        Convert RDP of order α to (ε, δ)-DP.
        ε = RDP(α) - log(δ) / (α - 1)
        """
        if alpha <= 1 or delta <= 0:
            return rdp
        return rdp + math.log(1.0 / delta) / (alpha - 1)

    def reset(self):
        """Reset the accountant for a new run."""
        self.rdp_consumed = 0.0
        self.ledger.clear()


class PrivacyBudgetExhaustedError(Exception):
    """Raised when the privacy budget is exceeded."""
    pass


def calibrate_noise_scale(epsilon: float, sensitivity: float = 2.0) -> float:
    """
    Calibrate Laplacian noise scale for a given epsilon and sensitivity.
    
    For Laplace mechanism: scale = sensitivity / epsilon
    For normalized embeddings, sensitivity ≈ 2 (max L2 distance).
    
    Args:
        epsilon: Privacy parameter for this query.
        sensitivity: L2 sensitivity of the query (default 2.0 for normalized embeddings).
        
    Returns:
        Scale parameter for np.random.laplace.
    """
    if epsilon <= 0:
        raise ValueError("Epsilon must be positive")
    return sensitivity / epsilon


def add_calibrated_noise(embeddings: np.ndarray, epsilon: float, sensitivity: float = 2.0) -> np.ndarray:
    """
    Add calibrated Laplacian noise to embeddings with proper DP guarantees.
    
    Args:
        embeddings: Query embeddings array of shape (n, d).
        epsilon: Privacy parameter.
        sensitivity: L2 sensitivity.
        
    Returns:
        Noisy embeddings, re-normalized to unit sphere.
    """
    scale = calibrate_noise_scale(epsilon, sensitivity)
    noise = np.random.laplace(0, scale, size=embeddings.shape).astype(np.float32)
    noisy = embeddings + noise

    # Re-normalize to unit sphere for cosine similarity / inner product
    norms = np.linalg.norm(noisy, axis=1, keepdims=True)
    noisy = noisy / (norms + 1e-10)

    return noisy
