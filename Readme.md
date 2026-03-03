# PrivaSyn - A Multi-Agent Framework for Privacy-Preserving Synthetic Data Generation with Formal Differential Privacy Guarantees

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/Tests-58%20Passing-success.svg)](tests/)
[![Stage](https://img.shields.io/badge/Stage-Research%20Paper-blueviolet.svg)]()

## Abstract

**PrivaSyn** addresses the critical challenge of generating high-fidelity synthetic data while providing provable privacy guarantees. Our framework introduces a *multi-agent architecture* — comprising a Generator, Chain-of-Thought Critic, and Red Team Attacker — orchestrated through a closed-loop self-correction workflow. By integrating Renyi Differential Privacy accounting with retrieval-augmented generation, PrivaSyn balances three competing objectives: **privacy**, **utility**, and **coherence**.

> **Key Insight:** Rather than relying on simple temperature-based retry loops, PrivaSyn uses structured LLM-generated feedback to *intelligently* fix privacy leaks — achieving both stronger guarantees and higher data quality.

---

## Novel Contributions

| # | Contribution | Description |
|:-:|:-------------|:------------|
| 1 | **Renyi DP Accounting** | Formal privacy budget tracking with provable guarantees ([Theorem 1](Docs/formal_dp_guarantee.md)) |
| 2 | **Chain-of-Thought Critic** | Structured JSON feedback for targeted self-correction |
| 3 | **Red Team Filtering** | Adversarial privacy auditing that catches leaks N-gram checks miss |
| 4 | **Perplexity Gate** | Coherence filtering via perplexity thresholding |
| 5 | **Adaptive Temperature** | Cosine-annealed scheduling based on failure type |
| 6 | **Shadow Model MIA** | 5-feature attack classifier for rigorous privacy evaluation |

---

## Getting Started

### Prerequisites

- Python 3.9+
- GPU recommended (validated on NVIDIA T4)

### Installation

```bash
git clone <repo-url>
cd Retrieval-Guided-Synthetic-Data-Generation
pip install -r requirements.txt
```

### Quick Run

```bash
# Full pipeline with default config
python3 -m src.main

# Experiment runner with YAML config
python run_experiment.py --config experiments/configs/full_pipeline.yaml --dataset sst2

# Run test suite (58 tests)
python3 -m pytest tests/ -v
```

---

## Architecture

PrivaSyn implements a **6-stage agentic workflow**:

```text
Data Loading ──> Semantic Indexing ──> LoRA Fine-Tuning ──> Agentic Generation ──> Evaluation ──> Results
     |                  |                    |                      |                   |            |
  Registry         FAISS + DP           PEFT Adapter        Generator -> Gate      Quality +     LaTeX +
  (3 datasets)     Noise on queries                         -> Critic -> Red Team   Privacy +    Markdown
                                                            -> Accept/Retry         MIA + DP     Tables
```

> See the full Mermaid diagrams in [Docs/architecture_flow.md](Docs/architecture_flow.md).

---

## Experiment Framework

PrivaSyn includes a complete experiment suite with **7 YAML configurations** for reproducible evaluation:

```bash
# Full ablation study (all 7 configs)
python run_experiment.py --ablation all --dataset sst2 --seed 42

# Individual configs
python run_experiment.py --config experiments/configs/full_pipeline.yaml --dataset sst2
python run_experiment.py --config experiments/configs/baseline_vanilla.yaml --dataset sst2

# Dry run (4 samples for quick testing)
python run_experiment.py --config experiments/configs/full_pipeline.yaml --dry-run

# Generate publication-ready tables
python run_experiment.py --results-table --results-dir experiments/results/

# List available datasets
python run_experiment.py --list-datasets
```

### Ablation Configurations

| Config | DP | Critic | Perplexity | Red Team | Temp | Retries |
|:-------|:--:|:------:|:----------:|:--------:|:----:|:-------:|
| `full_pipeline` | Yes | Yes | Yes | Yes | Cosine | 3 |
| `baseline_vanilla` | -- | -- | -- | -- | Fixed | 0 |
| `ablation_no_dp` | -- | Yes | Yes | Yes | Cosine | 3 |
| `ablation_no_critic` | Yes | -- | Yes | Yes | Cosine | 0 |
| `ablation_no_perplexity` | Yes | Yes | -- | Yes | Cosine | 3 |
| `ablation_no_redteam` | Yes | Yes | Yes | -- | Cosine | 3 |
| `ablation_fixed_temp` | Yes | Yes | Yes | Yes | Fixed | 3 |

---

## Supported Datasets

| Key | Dataset | Task | Labels | Source |
|:----|:--------|:-----|:------:|:-------|
| `sst2` | Stanford Sentiment Treebank v2 | Sentiment | 2 | `glue/sst2` |
| `ag_news` | AG News | Topic Classification | 4 | `fancyzhx/ag_news` |
| `imdb` | IMDB Reviews | Sentiment | 2 | `stanfordnlp/imdb` |

Datasets are auto-downloaded with column mapping handled by `dataset_registry.py`.

---

## Evaluation Metrics

| Category | Metric | Goal | Description |
|:---------|:-------|:----:|:------------|
| **Quality** | Semantic Similarity | Higher | Meaning preservation |
| **Quality** | Type-Token Ratio | Higher | Lexical richness |
| **Quality** | Self-BLEU | Lower | Output diversity |
| **Privacy** | Exact Match Rate | Lower | Verbatim copy detection |
| **Privacy** | 5-gram Overlap | Lower | N-gram leakage |
| **Privacy** | MIA Attack Success Rate | Lower | Membership inference resistance |
| **Privacy** | MIA AUC-ROC | Lower | Attacker discrimination ability |
| **Privacy** | TPR @ 1% FPR | Lower | True positive rate at low false positive |
| **Utility** | Downstream Accuracy | Higher | Classifier trained on synthetic data |
| **DP** | Epsilon Spent | -- | Cumulative privacy budget consumed |

---

## Project Structure

```text
privasyn/
├── run_experiment.py                  # CLI experiment runner
├── requirements.txt                   # Dependencies
│
├── experiments/
│   ├── configs/                       # 7 YAML experiment configs
│   └── results/                       # Saved results (JSON)
│
├── src/
│   ├── config.py                      # Validated @dataclass config
│   ├── dataset_registry.py            # Multi-dataset registry
│   ├── dataloader.py                  # Data loading + validation
│   ├── model_loader.py                # Shared quantized model factory
│   ├── logger.py                      # Centralized logging
│   ├── privacy_budget.py              # Renyi DP accountant
│   ├── utils.py                       # Seeds, device, I/O
│   ├── main.py                        # Pipeline orchestrator
│   │
│   ├── pipeline/
│   │   ├── prompts.py                 # All prompt templates
│   │   ├── indexing.py                # FAISS + noisy retrieval
│   │   ├── training.py                # LoRA fine-tuning
│   │   ├── critic.py                  # CoT Critic agent
│   │   └── generation.py              # Agentic generation loop
│   │
│   └── evaluation/
│       ├── quality.py                 # Similarity, TTR, Self-BLEU
│       ├── privacy.py                 # N-gram overlap, exact match
│       ├── red_team.py                # Adversarial attacker
│       ├── membership_inference.py    # Simple MIA
│       ├── shadow_model_mia.py        # Shadow model MIA
│       ├── downstream_task.py         # BERT classifier
│       ├── statistical_tests.py       # Bootstrap CI, t-tests, Cohen's d
│       └── results_table.py           # LaTeX + Markdown tables
│
├── tests/                             # 58 unit tests
├── notebooks/
│   └── PrivaSyn.ipynb                 # Complete Colab notebook
│
└── Docs/
    ├── architecture_flow.md           # System diagrams
    ├── formal_dp_guarantee.md         # Theorem 1 + proofs
    ├── master_guide.md                # Developer reference
    ├── commands.md                    # All CLI commands
    ├── Datasets.md                    # Dataset documentation
    └── research/                      # Related papers
```

---

## Documentation

| Document | Description |
|:---------|:------------|
| [Architecture Flow](Docs/architecture_flow.md) | System diagrams (Mermaid) |
| [Formal DP Guarantee](Docs/formal_dp_guarantee.md) | Theorem 1 (RDP Composition) + proof sketches |
| [Master Guide](Docs/master_guide.md) | Developer reference + configuration guide |
| [Commands](Docs/commands.md) | Complete CLI command reference |
| [Datasets](Docs/Datasets.md) | Supported datasets + how to add new ones |
| [Paper Summaries](Docs/research/paper_summaries.md) | 20 related research papers |

---

## Running on Google Colab

Upload `notebooks/PrivaSyn.ipynb` to Google Colab with a T4 GPU runtime. The notebook includes all source code and experiment cells.

```text
Section 1: Setup             → Clone + install
Section 2-6: Source Code      → All 18 modules
Section 7: Tests              → 58 unit tests
Section 8: Experiments        → Ablation, multi-seed, results tables
```

---

## Citation

If you use PrivaSyn in your research, please cite:

```bibtex
@article{privasyn2026,
  title={PrivaSyn: A Multi-Agent Framework for Privacy-Preserving Synthetic Data Generation with Formal Differential Privacy Guarantees},
  year={2026}
}
```

---

## References

1. Mironov, I. *Renyi Differential Privacy.* CSF 2017
2. Shokri et al. *Membership Inference Attacks Against ML Models.* IEEE S&P 2017
3. Hu et al. *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022
4. Lewis et al. *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* NeurIPS 2020

---
