# PrivaSyn — Formal Differential Privacy Guarantee

## Theorem 1: Privacy Composition

**Statement.** Let $\mathcal{M}_1, \mathcal{M}_2, \ldots, \mathcal{M}_T$ be $T$ adaptive mechanisms applied sequentially during synthetic data generation, where each mechanism $\mathcal{M}_t$ satisfies $(\alpha, \rho_t)$-Rényi Differential Privacy (RDP). Then the composed mechanism $\mathcal{M} = (\mathcal{M}_1, \ldots, \mathcal{M}_T)$ satisfies $(\varepsilon, \delta)$-DP where:

$$\varepsilon = \sum_{t=1}^{T} \rho_t + \frac{\log(1/\delta)}{\alpha - 1}$$

for any $\delta > 0$ and $\alpha > 1$.

**Proof sketch.** By the composition theorem for Rényi divergence (Mironov, 2017), RDP guarantees compose linearly:

$$D_\alpha(\mathcal{M}(D) \| \mathcal{M}(D')) \leq \sum_{t=1}^{T} \rho_t$$

The conversion from $(\alpha, \rho)$-RDP to $(\varepsilon, \delta)$-DP follows from Proposition 3 of Mironov (2017):

$$\varepsilon = \rho + \frac{\log(1/\delta)}{\alpha - 1}$$

This yields tighter bounds than standard (ε, δ)-DP composition. ∎

---

## Corollary 1: Per-Query Privacy

**Statement.** Each retrieval query in the Noisy Retrieval phase uses the Laplace mechanism with sensitivity $\Delta_2 = 2$ (for unit-normalized embeddings) and scale $b = \Delta_2 / \varepsilon_q$. This provides $\varepsilon_q$-DP per query.

**Derivation.** For the Laplace mechanism with parameter $b$, the RDP guarantee of order $\alpha$ is:

$$\rho_q(\alpha) = \frac{1}{\alpha - 1} \cdot \log\left(\frac{\alpha - 1}{2\alpha - 1} e^{(\alpha-1)\varepsilon_q} + \frac{\alpha}{2\alpha - 1} e^{-\alpha \varepsilon_q}\right)$$

where $\varepsilon_q = \Delta_2 / b$ is the per-query privacy parameter.

---

## Budget Tracking in the Pipeline

The `RenyiDPAccountant` class tracks privacy expenditure across all pipeline stages:

| Stage | Mechanism | ε per step |
|-------|-----------|-----------|
| Noisy Retrieval | Laplace noise on query embeddings | `PRIVACY_EPSILON` (default 0.1) |
| Each retry | Additional retrieval query | `PRIVACY_EPSILON` |
| Vocabulary anonymization | Name/date masking in critic | — (post-processing, no DP cost) |

**Total budget after T queries:**

$$\varepsilon_{\text{total}} = \text{RDP-to-DP}\left(\sum_{t=1}^{T} \rho(\varepsilon_q, \alpha),\; \alpha,\; \delta\right) \leq \varepsilon_{\text{budget}}$$

Generation **halts** when projected $\varepsilon_{\text{total}}$ would exceed $\varepsilon_{\text{budget}}$ (default: 10.0).

---

## Implementation Details

```python
# src/privacy_budget.py — Key methods

class RenyiDPAccountant:
    def consume(self, epsilon_step, step_name):
        # 1. Convert ε to RDP: ρ = _eps_to_rdp(ε, α)
        # 2. Accumulate: rdp_consumed += ρ  (linear composition)
        # 3. Convert back: ε_total = _rdp_to_eps_delta(rdp_consumed, α, δ)
        # 4. Check: if ε_total > budget → PrivacyBudgetExhaustedError
```

**Parameters used (defaults):**

- $\alpha = 5.0$ (Rényi divergence order)
- $\delta = 10^{-5}$ (failure probability)
- $\varepsilon_q = 0.1$ (per-query privacy parameter)
- $\varepsilon_{\text{budget}} = 10.0$ (total privacy budget)

---

## References

1. Mironov, I. "Rényi Differential Privacy." CSF 2017. [arXiv:1702.07476](https://arxiv.org/abs/1702.07476)
2. Balle, B., Gaboardi, M., Zanella-Béguelin, B. "Privacy Profiles and Amplification by Subsampling." JMLR 2020.
3. Dwork, C., Roth, A. "The Algorithmic Foundations of Differential Privacy." FnTCS 2014.
