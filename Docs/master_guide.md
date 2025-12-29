# Master Guide: Adaptive Retrieval-Guided Synthetic Data Generation

 This document serves as the comprehensive guide to the **Adaptive RAG** system for synthetic data generation. It details the system's architecture, including the novel feedback-guided self-correction mechanism, and provides step-by-step instructions for usage.

## 1. Project Overview

The goal of this project is to generate high-fidelity synthetic data from private datasets (like GLUE/SST2). It uses **Retrieval-Augmented Generation (RAG)** to ensure data utility while adapting a Large Language Model (LLM) to the specific domain.

**Key Features:**

- **Privacy-Preserving**: Mitigates risk by rewriting private examples using public context.
- **Adaptive Self-Correction (Novelty)**: Automatically detects and fixes low-quality or privacy-violating samples during generation.
- **High Efficiency**: Uses batched processing for retrieval and generation to maximize GPU throughput.

## 2. System Architecture

The pipeline consists of four main stages, orchestrated by `src/main.py`.

### Stage 1: Indexing

- **Goal**: Create a searchable knowledge base from public data.
- **Process**:
    1. Loads a public corpus (CSV).
    2. Chunks text into passages.
    3. Generates embeddings using `sentence-transformers/all-MiniLM-L6-v2`.
    4. Builds a **FAISS** index for fast similarity search.

### Stage 2: Fine-Tuning

- **Goal**: Adapt the LLM to the domain of the private data.
- **Process**:
    1. Retrieves relevant public context for each private training example.
    2. Fine-tunes the base model (`Qwen/Qwen2-0.5B-Instruct`) using **LoRA (Low-Rank Adaptation)**.
    3. This teaches the model the *style* and *structure* of the desired data.

### Stage 3: Adaptive Generation (The Core Innovation)

- **Goal**: Generate synthetic samples that satisfy both Privacy and Utility constraints.
- **Mechanism**: A **Feedback-Guided Loop** runs for every batch of data.
    1. **Generate**: The model enables a candidate response.
    2. **Evaluate**:
        - **Privacy Check**: Calculates N-gram overlap with the original private text. (Target: `< 50%`)
        - **Utility Check**: Calculates Semantic Similarity with the original text. (Target: `> 0.7`)
    3. **Self-Correct**:
        - **If Privacy Fails**: The system *increases temperature* (randomness) to encourage diverse rewriting and retries.
        - **If Utility Fails**: The system *decreases temperature* to focus the model and retries.
    4. **Finalize**: The best candidate within `MAX_RETRIES` is saved.

### Stage 4: Evaluation

- **Goal**: Measure the success of the synthetic data.
- **Metrics**:
  - **Downstream Utility**: Trains a BERT classifier on synthetic data and tests on real data.
  - **Quality**: diversity (Self-BLEU) and richness (TTR).
  - **Privacy**: Exact Match Ratio and N-gram Overlap.

## 3. Setup & Installation

**Prerequisites:**

- Python 3.9+
- (Optional) NVIDIA GPU for acceleration

**Installation:**

1. **Install Dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

    *(Note: On Mac, ensures `faiss-cpu` is used. On Linux/Windows with GPU, install `faiss-gpu`)*.

2. **Download NLTK Data**:
    The script handles this, but you may need to run:

    ```bash
    python -m nltk.downloader punkt
    ```

## 4. Usage

Run the entire pipeline with a single command:

```bash
python main.py
```

### Configuration (`src/config.py`)

You can control the behavior of the Adaptive RAG system by modifying `src/config.py`:

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `MAX_RETRIES` | `3` | Maximum attempts to self-correct a sample. |
| `MAX_NGRAM_OVERLAP` | `0.5` | Maximum allowed 5-gram overlap (Privacy threshold). |
| `MIN_SEMANTIC_SIM` | `0.7` | Minimum required semantic similarity (Utility threshold). |
| `BATCH_SIZE_GENERATION` | `8` | Number of samples to process in parallel. |
| `MAX_TRAIN_SAMPLES` | `4` | Number of samples for training (Set to `None` for full run). |

## 5. Directory Structure

- `src/main.py`: Entry point.
- `src/pipeline/generation.py`: Contains the **Adaptive Logic**.
- `src/evaluation/`: Metrics for Privacy and Quality.
- `tests/test_novelty.py`: Unit test verifying the self-correction logic.
