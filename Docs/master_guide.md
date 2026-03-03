# Master Guide: PrivaSyn

## 1. Overview

**PrivaSyn** is a multi-agent framework for privacy-preserving synthetic data generation with formal differential privacy guarantees using retrieval-augmented self-correction. It generates synthetic data that is semantically faithful to the original while provably preventing data leakage.

### Three Pillars

| Pillar | Agent | How It Works |
|--------|-------|-------------|
| **Intelligent Critique** | CoT Critic | Analyzes failures → produces structured JSON feedback → guides regeneration |
| **Adversarial Auditing** | Red Team | Attempts to extract PII → rejects samples the Critic missed |
| **Formal Privacy** | DP Accountant | Tracks Rényi DP budget → halts generation when budget exhausted |

---

## 2. Running the System

### Option A: Simple Pipeline

```bash
python3 -m src.main
```

Runs the full pipeline using defaults from `src/config.py`.

### Option B: Experiment Runner (Recommended)

```bash
# Single experiment with YAML config
python run_experiment.py --config experiments/configs/full_pipeline.yaml --dataset sst2

# Full ablation study
python run_experiment.py --ablation all --dataset sst2 --seed 42

# Quick testing
python run_experiment.py --config experiments/configs/full_pipeline.yaml --dry-run

# Generate LaTeX + Markdown results tables
python run_experiment.py --results-table
```

### Option C: Run Tests

```bash
python3 -m pytest tests/ -v    # 58 tests
```

---

## 3. Configuration

### YAML Configs (for experiments)

Located in `experiments/configs/`. Each file controls which novelties are enabled:

| Config | DP | Perplexity | Red Team | Adaptive Temp | Retries |
|--------|----|-----------|----------|--------------|---------|
| `full_pipeline.yaml` | ✅ | ✅ | ✅ | cosine | 3 |
| `baseline_vanilla.yaml` | ❌ | ❌ | ❌ | fixed | 0 |
| `ablation_no_dp.yaml` | ❌ | ✅ | ✅ | cosine | 3 |
| `ablation_no_critic.yaml` | ✅ | ✅ | ❌ | cosine | 0 |
| `ablation_no_perplexity.yaml` | ✅ | ❌ | ✅ | cosine | 3 |
| `ablation_no_redteam.yaml` | ✅ | ✅ | ❌ | cosine | 3 |
| `ablation_fixed_temp.yaml` | ✅ | ✅ | ✅ | fixed | 3 |

### Python Config (defaults)

`src/config.py` — a validated `@dataclass` with `__post_init__` checks:

```python
# Key parameters
PRIVACY_EPSILON = 0.1           # Per-query DP noise (lower = more private)
TOTAL_PRIVACY_BUDGET = 10.0     # Max ε before halting
ENABLE_DP_ACCOUNTING = True     # Track Rényi DP budget
ENABLE_PERPLEXITY_GATE = True   # Filter incoherent samples
ENABLE_RED_TEAM = True          # Adversarial privacy audit
MAX_RETRIES = 3                 # Self-correction attempts
TEMP_SCHEDULE = "cosine"        # cosine | linear | fixed
SEED = 42                       # Reproducibility
```

---

## 4. Module Reference

### Infrastructure

| Module | Description |
|--------|-------------|
| `config.py` | Validated `@dataclass` with `to_dict()`, computed properties |
| `model_loader.py` | Shared factory for quantized models (eliminates duplication) |
| `logger.py` | `logging` module with console + file handlers |
| `dataset_registry.py` | Multi-dataset registry with auto column mapping |
| `privacy_budget.py` | Rényi DP accountant + noise calibration |

### Pipeline

| Module | Description |
|--------|-------------|
| `indexing.py` | FAISS index with calibrated DP noise on queries |
| `training.py` | LoRA fine-tuning with PEFT |
| `generation.py` | Agentic loop: generate → gate → critique → retry |
| `critic.py` | CoT Critic agent (structured JSON feedback) |
| `prompts.py` | All prompt templates (single source of truth) |

### Evaluation

| Module | Description |
|--------|-------------|
| `quality.py` | Semantic similarity, TTR, Self-BLEU |
| `privacy.py` | N-gram overlap, exact match |
| `red_team.py` | Adversarial privacy attacker |
| `membership_inference.py` | Simple similarity-based MIA |
| `shadow_model_mia.py` | Full shadow-model MIA (5-feature classifier) |
| `statistical_tests.py` | Bootstrap CI, paired t-test, Cohen's d |
| `results_table.py` | LaTeX + Markdown table generation |
| `downstream_task.py` | BERT classifier utility evaluation |

---

## 5. What to Expect in Logs

```
STAGE 1: Loading Data
  Loading dataset: sst2 (Stanford Sentiment Treebank)
  Loaded 67,349 samples, 2 labels

STAGE 4: Enhanced Agentic Generation
  [Batch 1/84] Processing 8 samples...
  [Sample 3] Privacy violation (overlap=0.72) → Critic feedback → Retry 1
  [Sample 3] CoT Critic: "Replace hospital name 'Mount Sinai' with generic term"
  [Sample 3] Retry passed ✅ (overlap=0.18, sim=0.83, ppl=24.5)
  [Sample 5] Red Team caught semantic leak → Rejected
  DP Budget: ε_spent=2.34, remaining=7.66

STAGE 5: Comprehensive Evaluation
  Semantic Similarity: 0.8142
  Self-BLEU: 0.3215
  MIA ASR: 52.3% (✅ near random chance)
  Shadow MIA AUC: 0.5145
```

---

## 6. Adding a New Dataset

1. Add entry to `DATASET_REGISTRY` in `src/dataset_registry.py`
2. Create YAML configs in `experiments/configs/`
3. Run: `python run_experiment.py --config your_config.yaml --dataset your_key`

---

## 7. References

- [Formal DP Guarantee](formal_dp_guarantee.md) — Theorem 1 + proof sketches
- [Architecture Flow](architecture_flow.md) — System diagrams
- [Research Papers](research/paper_summaries.md) — 20 related papers
