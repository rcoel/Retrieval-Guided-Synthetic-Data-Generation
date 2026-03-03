# Walkthrough: PrivaSyn — System Verification & How to Run

## Running the System

### Full Pipeline

```bash
python3 -m src.main
```

### Experiment Runner (Recommended)

```bash
# Single experiment
python run_experiment.py --config experiments/configs/full_pipeline.yaml --dataset sst2 --seed 42

# Full ablation (7 configs)
python run_experiment.py --ablation all --dataset sst2

# Dry run (4 samples for quick testing)
python run_experiment.py --config experiments/configs/full_pipeline.yaml --dry-run

# Generate LaTeX + Markdown results tables
python run_experiment.py --results-table

# List available datasets
python run_experiment.py --list-datasets
```

### Run Tests

```bash
python3 -m pytest tests/ -v    # 58 tests
```

## Test Suite (58 Tests)

| File | Tests | Coverage |
|------|-------|----------|
| `test_config.py` | 10 | Config validation, serialization, backward compat |
| `test_privacy_budget.py` | 11 | DP accountant, noise calibration, budget exhaustion |
| `test_prompts.py` | 11 | All 3 prompt templates |
| `test_generation.py` | 8 | Adaptive temperature scheduling |
| `test_dataloader.py` | 5 | Error handling, data validation |
| `test_utils.py` | 5 | Seed reproducibility, device, I/O |
| `test_differentiation.py` | 4 | Config, critic prompt, red team |
| `test_novelty.py` | 1 | Batched self-correction (mocked) |

## Key Commands Reference

```bash
# Pipeline
python3 -m src.main                     # Run default pipeline
python run_experiment.py --help         # Show all CLI options

# Experiments
python run_experiment.py --ablation all --dataset ag_news --seed 42
python run_experiment.py --results-table --results-dir experiments/results/

# Tests
python3 -m pytest tests/ -v            # All 58 tests
python3 -m pytest tests/test_config.py  # Specific test file
```
