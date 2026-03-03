# Differentiation Strategy — What Makes PrivaSyn Novel

## vs. Standard RAG Pipelines (LangChain, RAGAS, LlamaIndex)

| Feature | Standard RAG | **PrivaSyn** |
|---------|-------------|-------------------|
| Generation feedback | None or simple retry | **CoT Critic → structured JSON → targeted fix** |
| Privacy check | Post-hoc N-gram overlap | **3-layer: N-gram + Red Team + DP noise** |
| Privacy guarantee | None | **Formal Rényi DP accounting (Theorem 1)** |
| Coherence check | None | **Perplexity gate** |
| Temperature strategy | Fixed or random | **Adaptive cosine annealing by failure type** |
| Privacy evaluation | Basic overlap stats | **Shadow model MIA (5-feature attack classifier)** |
| Multi-dataset | Manual setup | **Auto registry (SST-2, AG News, IMDB)** |
| Experiments | Ad hoc | **YAML configs, CLI runner, ablation suite** |
| Statistical rigor | None | **Bootstrap CI, paired t-test, Cohen's d** |

## 6 Novel Contributions for Paper

1. **Rényi DP Accounting** — Provable privacy budget with formal theorem
2. **Chain-of-Thought Critic** — LLM-based structured feedback (not just temperature tuning)
3. **Red Team Adversarial Filter** — Catches semantic leaks N-gram checks miss
4. **Perplexity-Gated Coherence** — Filters hallucinated/incoherent outputs
5. **Adaptive Temperature Scheduling** — Cosine annealing based on failure type
6. **Shadow Model MIA Evaluation** — Publication-standard privacy evaluation
