# PrivaSyn — Privacy-Preserving Synthetic Data Generation

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.9+-green.svg)
![Tests](https://img.shields.io/badge/tests-58%20passing-brightgreen)
![Stage](https://img.shields.io/badge/stage-Research%20Paper-blueviolet)

## Overview

**PrivaSyn** is a multi-agent framework for privacy-preserving synthetic data generation with formal differential privacy guarantees using retrieval-augmented self-correction. The system uses a multi-agent architecture (Generator, Critic, Red Team) with a closed-loop self-correction workflow that balances the **Privacy × Utility × Coherence** quality triangle.

### Novel Contributions

| # | Contribution | Description |
|---|-------------|-------------|
| 1 | **Rényi DP Accounting** | Formal privacy budget tracking with provable guarantees (Theorem 1) |
| 2 | **Chain-of-Thought Critic** | Structured JSON feedback for intelligent self-correction |
| 3 | **Red Team Filtering** | Adversarial privacy auditing — catches leaks N-gram checks miss |
| 4 | **Perplexity Gate** | Coherence filtering via perplexity thresholding |
| 5 | **Adaptive Temperature** | Cosine-annealed temperature scheduling based on failure type |
| 6 | **Shadow Model MIA** | 5-feature attack classifier for rigorous privacy evaluation |

---

## Quick Start

### Installation

```bash
git clone <repo-url>
cd Retrieval-Guided-Synthetic-Data-Generation
pip install -r requirements.txt
```

### Run the Pipeline

```bash
# Full pipeline (loads config from src/config.py)
python3 -m src.main

# Or use the experiment runner with YAML configs
python run_experiment.py --config experiments/configs/full_pipeline.yaml --dataset sst2
```

### Run Experiments

```bash
# Single experiment
python run_experiment.py --config experiments/configs/full_pipeline.yaml --dataset sst2 --seed 42

# Full ablation study (runs all 7 configs)
python run_experiment.py --ablation all --dataset ag_news --seed 42

# Dry run (tiny samples for quick testing)
python run_experiment.py --config experiments/configs/full_pipeline.yaml --dry-run

# Generate publication-ready results tables
python run_experiment.py --results-table --results-dir experiments/results/

# List available datasets
python run_experiment.py --list-datasets
```

### Run Tests

```bash
python3 -m pytest tests/ -v   # 58 tests
```

---

## Architecture

The pipeline uses a **6-stage agentic workflow**:

```
1. Data Loading        → Dataset registry (SST-2, AG News, IMDB)
2. Semantic Indexing   → FAISS index with calibrated DP noise on queries
3. LoRA Fine-Tuning    → Domain adaptation with PEFT
4. Agentic Generation  → Generator → Quality Gate → Critic → Red Team → Accept/Retry
5. Evaluation          → Quality + Privacy + MIA + DP Audit + Downstream
6. Results             → LaTeX/Markdown tables with bootstrap CI
```

See [Docs/architecture_flow.md](Docs/architecture_flow.md) for the full Mermaid diagram.

---

## Project Structure

```
├── run_experiment.py              # CLI experiment runner
├── requirements.txt               # Dependencies
├── experiments/
│   ├── configs/                   # YAML experiment configs
│   │   ├── full_pipeline.yaml     # All novelties ON
│   │   ├── baseline_vanilla.yaml  # All novelties OFF
│   │   ├── ablation_no_dp.yaml
│   │   ├── ablation_no_critic.yaml
│   │   ├── ablation_no_perplexity.yaml
│   │   ├── ablation_no_redteam.yaml
│   │   └── ablation_fixed_temp.yaml
│   └── results/                   # Saved experiment results (JSON)
├── src/
│   ├── main.py                    # Pipeline orchestrator
│   ├── config.py                  # PipelineConfig dataclass
│   ├── dataset_registry.py        # Multi-dataset registry
│   ├── dataloader.py              # Data loading with validation
│   ├── model_loader.py            # Shared model factory (quantization)
│   ├── logger.py                  # Centralized logging
│   ├── privacy_budget.py          # Rényi DP accountant
│   ├── utils.py                   # Seeds, device, I/O helpers
│   ├── pipeline/
│   │   ├── indexing.py            # FAISS + noisy retrieval
│   │   ├── training.py            # LoRA fine-tuning
│   │   ├── generation.py          # Agentic generation loop
│   │   ├── critic.py              # CoT Critic agent
│   │   └── prompts.py             # All prompt templates
│   └── evaluation/
│       ├── quality.py             # Semantic similarity, TTR, Self-BLEU
│       ├── privacy.py             # N-gram overlap, exact match
│       ├── downstream_task.py     # BERT classifier utility test
│       ├── red_team.py            # Adversarial privacy attacker
│       ├── membership_inference.py # Simple MIA (similarity-based)
│       ├── shadow_model_mia.py    # Full shadow model MIA
│       ├── statistical_tests.py   # Bootstrap CI, t-tests, Cohen's d
│       └── results_table.py       # LaTeX + Markdown table generator
├── tests/                         # 58 unit tests
│   ├── test_config.py
│   ├── test_dataloader.py
│   ├── test_generation.py
│   ├── test_prompts.py
│   ├── test_privacy_budget.py
│   ├── test_utils.py
│   ├── test_differentiation.py
│   └── test_novelty.py
└── Docs/
    ├── architecture_flow.md       # System diagrams (Mermaid)
    ├── formal_dp_guarantee.md     # Theorem 1 + proofs
    ├── master_guide.md            # Developer guide
    ├── Datasets.md                # Supported datasets
    ├── research/
    │   ├── paper_summaries.md     # 20 related papers
    │   └── paper_links.md         # Reference links
    └── literature-survey.md       # Full literature survey
```

---

## Configuration

### YAML Config (Experiments)

```yaml
# experiments/configs/full_pipeline.yaml
dataset: sst2
enable_dp_accounting: true
enable_perplexity_gate: true
enable_red_team: true
temp_schedule: cosine       # cosine | linear | fixed
generation_temp: 0.8
max_retries: 3
```

### Python Config (Default)

All defaults in `src/config.py` — validated `@dataclass` with `__post_init__` checks:

```python
from src.config import PipelineConfig
cfg = PipelineConfig(PRIVACY_EPSILON=0.05, TEMP_SCHEDULE="linear")
print(cfg.to_dict())  # Serializable
```

---

## Supported Datasets

| Key | Dataset | Task | Labels | Source |
|-----|---------|------|--------|--------|
| `sst2` | Stanford Sentiment Treebank v2 | Sentiment | 2 | `glue/sst2` |
| `ag_news` | AG News | Topic Classification | 4 | `fancyzhx/ag_news` |
| `imdb` | IMDB Reviews | Sentiment | 2 | `stanfordnlp/imdb` |

Auto column mapping and public corpus generation handled by `dataset_registry.py`.

---

## Evaluation Metrics

| Category | Metric | Direction | Description |
|----------|--------|-----------|-------------|
| **Quality** | Semantic Similarity | ↑ Higher | Meaning preservation |
| **Quality** | TTR | ↑ Higher | Lexical richness |
| **Quality** | Self-BLEU | ↓ Lower | Diversity (less mode collapse) |
| **Privacy** | Exact Match | ↓ Lower | Verbatim copy rate |
| **Privacy** | 5-gram Overlap | ↓ Lower | N-gram leakage |
| **Privacy** | MIA ASR | ↓ Lower | Membership inference attack success |
| **Privacy** | MIA AUC-ROC | ↓ Lower | Attacker discrimination ability |
| **Privacy** | TPR@1%FPR | ↓ Lower | True positive rate at low false positive |
| **Utility** | Downstream Accuracy | ↑ Higher | Classifier trained on synthetic data |
| **DP** | ε spent | — | Cumulative privacy budget consumed |

---

## References

1. Mironov, I. "Rényi Differential Privacy." CSF 2017
2. Shokri et al. "Membership Inference Attacks Against ML Models." IEEE S&P 2017
3. Hu et al. "LoRA: Low-Rank Adaptation of Large Language Models." ICLR 2022
4. Lewis et al. "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." NeurIPS 2020

---
