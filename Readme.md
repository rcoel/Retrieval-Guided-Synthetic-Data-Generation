# Research

Research paper Drive Link - <https://drive.google.com/drive/folders/1M2ul7W8e9H4aOqx-4ACgh5aqHPM9NXfs>

# Synthetic Data Generation with RAG

This project implements the methodology for generating high-fidelity synthetic data while mitigating privacy risks using a Retrieval-Augmented Generation (RAG) pipeline. The entire pipeline, from fine-tuning to evaluation, is implemented using the Hugging Face ecosystem.

## Project Structure

- `main.py`: The main script to run the entire pipeline: indexing, training, generation, and evaluation.
- `requirements.txt`: A list of Python dependencies for the project.
- `src/`: Contains the source code.
- `src/config.py`: Central configuration for file paths, model names, and all hyperparameters.
- `src/data_loader.py`: Handles loading and preprocessing of datasets from Hugging Face Hub or local files.
- `src/pipeline/`: Modules related to the data generation pipeline.
  - `indexing.py`: Creates and manages the FAISS semantic index for the public corpus.
  - `training.py`: Fine-tunes the LLM using PEFT (LoRA) and the Hugging Face Trainer.
  - `generation.py`: Generates synthetic data using the fine-tuned RAG model.
- `src/evaluation/`: Modules for evaluating the generated data.
  - `downstream_task.py`: Trains and evaluates a classifier on downstream tasks.
  - `quality.py`: Calculates data quality metrics (Perplexity, Self-BLEU, TTR, Semantic Similarity).
  - `privacy.py`: Performs privacy leakage analysis (Exact Match, N-gram Overlap).
- `src/utils.py`: Utility functions used across the project.

## Setup

1.  **Create a Virtual Environment (Recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

2.  **Install Dependencies:**
    This project requires PyTorch with CUDA support for efficient model training.

    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure Paths:**
    Update the placeholder paths in `src/config.py` to point to your data and desired output locations.

4.  **Hugging Face Login (Optional but Recommended):**
    To use gated models like Qwen2, log in with your Hugging Face token.
    ```bash
    huggingface-cli login
    ```

## Usage

Run the main experiment script from the root directory. This will execute all stages of the methodology.

```bash
python main.py
```
