"""
PrivaSyn — Privacy-Preserving Synthetic Data Generation.

A research-grade pipeline for privacy-preserving synthetic data generation
using an agentic multi-model architecture with formal DP guarantees.

Modules:
    config          — Validated PipelineConfig dataclass
    dataset_registry — Multi-dataset registry (SST-2, AG News, IMDB)
    dataloader      — Data loading with validation
    model_loader    — Shared model factory (quantization, caching)
    logger          — Centralized logging (console + file)
    privacy_budget  — Rényi DP accounting and noise calibration
    utils           — Seeds, device selection, I/O helpers

Sub-packages:
    pipeline        — Indexing, training, generation, critique, prompts
    evaluation      — Quality, privacy, MIA, statistical tests, results tables
"""
