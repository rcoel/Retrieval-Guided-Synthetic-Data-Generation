# Retrieval-Guided Synthetic Data Generation

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.9+-green.svg)
![Stage](https://img.shields.io/badge/stage-Research%20Preview-orange)

## Overview

This project implements a novel **Adaptive Retrieval-Augmented Generation (RAG)** pipeline designed to generate high-fidelity synthetic data. By combining retrieval-augmented fine-tuning (RAFT) with a feedback-guided self-correction loop, the system produces synthetic samples that balance **utility** (semantic faithfulness) and **privacy** (data rewriting).

Unlike robust "open-loop" generation scripts, this architecture employs a **closed-loop agentic workflow** that actively monitors quality metrics in real-time and dynamically adjusts generation parameters to self-correct errors.

## Key Features

- **Adaptive Self-Correction**: A novel feedback loop that evaluates generated samples on-the-fly for Privacy (N-gram overlap) and Utility (Semantic Similarity). It effectively "retries" generation with adjusted temperature settings if thresholds are violated.
- **Privacy-Preserving**: Designed to rewrite sensitive private examples using public context, ensuring the synthetic output is not a direct copy.
- **High-Efficiency**: Optimized with **batched processing** for both retrieval and generation stages, maximizing GPU throughput.
- **Small Language Model (SLM) Ready**: Validated with `Qwen/Qwen2-0.5B-Instruct`, demonstrating high-quality results without requiring massive compute resources.

## Architecture

The pipeline consists of four integrated stages:

1. **Indexing (Public Knowledge Base)**:
    - Ingests a public corpus (CSV) and chunks text into passages.
    - Builds a high-performance **FAISS** vector index using `sentence-transformers` embeddings.

2. **Fine-Tuning (Domain Adaptation)**:
    - Retrieves relevant public context for each private training example.
    - Fine-tunes the LLM (using **LoRA**) to understand the data distribution and task structure.

3. **Adaptive Generation (The Core Engine)**:
    - Generates candidates in batches.
    - **Evaluation**: Checks `N-Gram Overlap < 50%` and `Semantic Similarity > 0.7`.
    - **Loop**: If a check fails, the system alters the sampling temperature (Higher for privacy violations, Lower for utility failures) and regenerates.

4. **Evaluation**:
    - **Downstream Utility**: Trains a BERT classifier on synthetic data to verify performance on real test sets.
    - **Quality Metrics**: Reports Exact Match, Self-BLEU (Diversity), and TTR (Richness).

## Setup & Installation

### Prerequisites

- Python 3.9+
- (Optional) NVIDIA GPU for acceleration

### Installation

1. **Clone the Repository**

    ```bash
    git clone <repo-url>
    cd Retrieval-Guided-Synthetic-Data-Generation
    ```

2. **Install Dependencies**

    ```bash
    pip install -r requirements.txt
    ```

    *Note: On macOS, this defaults to `faiss-cpu`. On Linux/Windows with CUDA, ensure `faiss-gpu` is installed.*

3. **Download NLTK Data**
    The system attempts to download this automatically, but you can manually ensure `punkt` is available:

    ```bash
    python -m nltk.downloader punkt
    ```

## Usage

### Quick Start

Run the entire pipeline (Indexing → Training → Generation → Evaluation) with a single command:

```bash
python main.py
```

### Configuration

All hyperparameters are centralized in `src/config.py`. Key parameters for the Adaptive RAG system include:

```python
# --- Adaptive RAG Parameters ---
MAX_RETRIES = 3           # Max attempts to fix a sample
MAX_NGRAM_OVERLAP = 0.5   # Privacy Threshold (Lower is stricter)
MIN_SEMANTIC_SIM = 0.7    # Utility Threshold (Higher is stricter)
BATCH_SIZE_GENERATION = 8 # Batch size (Increase for powerful GPUs)
```

To run a fast test drive, set `MAX_TRAIN_SAMPLES` and `MAX_GENERATION_SAMPLES` to small integers (e.g., 4 or 8) in `src/config.py`.

## Project Structure

```bash
├── data/                   # Input data storage
├── notebooks/              # Jupyter notebooks for experimentation
├── output/                 # Generated data, models, and indices
├── src/
│   ├── main.py             # Pipeline orchestrator
│   ├── config.py           # Configuration hub
│   ├── dataloader.py       # Data ingestion utilities
│   ├── pipeline/
│   │   ├── indexing.py     # FAISS indexing logic
│   │   ├── training.py     # LoRA Fine-tuning logic
│   │   └── generation.py   # Adaptive RAG Loop logic
│   └── evaluation/
│       ├── quality.py      # Similarity & diversity metrics
│       ├── privacy.py      # Privacy leakage metrics
│       └── downstream.py   # Utility validation
└── tests/                  # Unit tests for verification
```

## 🔗 Resources

- **Research Paper Resources**: [Drive Link](https://drive.google.com/drive/folders/1M2ul7W8e9H4aOqx-4ACgh5aqHPM9NXfs)

---
*Built for the Advanced Agentic Coding Project.*
