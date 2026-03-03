# PrivaSyn — Commands Reference

## Setup

```bash
# Clone and install
git clone <repo-url>
cd Retrieval-Guided-Synthetic-Data-Generation
pip install -r requirements.txt
```

---

## Running the Pipeline

```bash
# Default pipeline (uses src/config.py defaults)
python3 -m src.main
```

---

## Experiment Runner

```bash
# List available datasets
python run_experiment.py --list-datasets

# Single experiment with YAML config
python run_experiment.py --config experiments/configs/full_pipeline.yaml --dataset sst2 --seed 42

# Dry run (4 samples — quick testing)
python run_experiment.py --config experiments/configs/full_pipeline.yaml --dry-run

# Run on different datasets
python run_experiment.py --config experiments/configs/full_pipeline.yaml --dataset ag_news
python run_experiment.py --config experiments/configs/full_pipeline.yaml --dataset imdb

# Full ablation study (runs all 7 configs)
python run_experiment.py --ablation all --dataset sst2 --seed 42

# Generate results tables (LaTeX + Markdown)
python run_experiment.py --results-table --results-dir experiments/results/

# Show all options
python run_experiment.py --help
```

---

## Experiment Configs

```bash
# Full pipeline (all novelties ON)
python run_experiment.py --config experiments/configs/full_pipeline.yaml --dataset sst2

# Vanilla baseline (all novelties OFF)
python run_experiment.py --config experiments/configs/baseline_vanilla.yaml --dataset sst2

# Ablation: No DP accounting
python run_experiment.py --config experiments/configs/ablation_no_dp.yaml --dataset sst2

# Ablation: No CoT Critic
python run_experiment.py --config experiments/configs/ablation_no_critic.yaml --dataset sst2

# Ablation: No perplexity gate
python run_experiment.py --config experiments/configs/ablation_no_perplexity.yaml --dataset sst2

# Ablation: No Red Team
python run_experiment.py --config experiments/configs/ablation_no_redteam.yaml --dataset sst2

# Ablation: Fixed temperature (no adaptive scheduling)
python run_experiment.py --config experiments/configs/ablation_fixed_temp.yaml --dataset sst2
```

---

## Tests

```bash
# Run ALL tests (58 tests)
python3 -m pytest tests/ -v

# Run specific test files
python3 -m pytest tests/test_config.py -v              # 10 tests — config validation
python3 -m pytest tests/test_privacy_budget.py -v       # 11 tests — DP accountant
python3 -m pytest tests/test_prompts.py -v              # 11 tests — prompt templates
python3 -m pytest tests/test_generation.py -v           #  8 tests — adaptive temperature
python3 -m pytest tests/test_dataloader.py -v           #  5 tests — data loading
python3 -m pytest tests/test_utils.py -v                #  5 tests — utilities
python3 -m pytest tests/test_differentiation.py -v      #  4 tests — integration
python3 -m pytest tests/test_novelty.py -v              #  1 test  — batched self-correction

# Quick summary (no verbose)
python3 -m pytest tests/ -q

# Run with coverage (if pytest-cov installed)
python3 -m pytest tests/ --cov=src --cov-report=term-missing
```

---

## Multi-Dataset Experiments

```bash
# SST-2 (sentiment, 2 labels)
python run_experiment.py --ablation all --dataset sst2 --seed 42

# AG News (topic classification, 4 labels)
python run_experiment.py --ablation all --dataset ag_news --seed 42

# IMDB (sentiment, 2 labels)
python run_experiment.py --ablation all --dataset imdb --seed 42

# Compare results across datasets
python run_experiment.py --results-table
```

---

## Multi-Seed Experiments (Reproducibility)

```bash
# Run same config with different seeds
python run_experiment.py --config experiments/configs/full_pipeline.yaml --dataset sst2 --seed 42
python run_experiment.py --config experiments/configs/full_pipeline.yaml --dataset sst2 --seed 123
python run_experiment.py --config experiments/configs/full_pipeline.yaml --dataset sst2 --seed 456

# Results saved to experiments/results/ with seed in filename
```

---

## Utility Commands

```bash
# Download NLTK data manually (auto-downloaded by pipeline)
python3 -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

# Check device (CUDA/MPS/CPU)
python3 -c "from src.utils import get_device; print(get_device())"

# Validate config
python3 -c "from src.config import PipelineConfig; cfg = PipelineConfig(); print(cfg.to_dict())"

# Check dataset registry
python3 -c "from src.dataset_registry import list_datasets, get_dataset_spec
for d in list_datasets():
    s = get_dataset_spec(d)
    print(f'{d:12s} {s.description} ({s.num_labels} labels)')
"
```
