# Master Guide: Agentic Retrieval-Guided Synthetic Data Generation

This guide serves as the definitive manual for the **Enhanced Adaptive RAG System**. This project has been upgraded from a standard RAG pipeline to a **Closed-Loop Agentic Framework** designed to generate high-fidelity synthetic data while strictly preserving privacy.

---

## 1. Core Architecture & Uniqueness

Unlike standard tools (e.g., RAGAS, LangChain RAG) that rely on simple "generation-evaluation" steps, this system implements three novel layers of intelligence to solve the Privacy-Utility tradeoff.

### 🧠 Pillar I: Agentic Critique (The "Coach")

**Problem**: Standard systems just "retry" with higher randomness (temperature) when privacy checks fail. This is inefficient.
**Our Solution**: Failing samples are sent to a **Critic Agent**.

1. **Analyze**: The Critic reviews *why* the sample failed (e.g., "Retained specific location 'Mount Sinai Hospital'").
2. **Instruct**: It generates a natural language instruction (e.g., "Replace hospital names with generic facility types").
3. **Refine**: The Generator retries using this explicit instruction, solving the problem intelligently.

### 🛡️ Pillar II: Red Team Filtering (The "Hacker")

**Problem**: Passing an N-Gram overlap check doesn't guarantee privacy (e.g., "John Doe" -> "Doe, John" has 0% overlap but huge leakage).
**Our Solution**: A **Privacy Attacker (Red Team)** runs as a text classifier on the final output.

- It proactively attempts to extract PII or infer the original identity.
- If the Attacker succeeds, the sample is **rejected immediately**, ensuring robustness against model inversion attacks.

### 👻 Pillar III: Noisy Retrieval (DP-Inspired)

**Problem**: If you retrieve public context that is *too* similar to the private data, you risk reconstruction.
**Our Solution**: **Laplacian Noise Injection**.

- Before querying the database, we add calibrated noise (`PRIVACY_EPSILON`) to the query embeddings.
- This creates a mathematical "fuzziness" that prevents exact pattern matching, adding a layer of plausible deniability to the retrieval process.

---

## 2. Impact on Workflows

| Component | Old Logic | **New Agentic Logic** | Benefit |
| :--- | :--- | :--- | :--- |
| **Indexing** | Exact Match | **Noisy/Fuzzy Match** | Prevents Context Reconstruction |
| **Generation** | Retry Loop | **Critic -> Refiner Loop** | Higher efficacy in fixing leaks |
| **Evaluation** | Quality Metrics | **Adversarial Attack** | Robustness against inference |

---

## 3. Configuration Guide

All improvements are controlled via `src/config.py`.

### Privacy Tuning

```python
# --- Advanced Privacy Features ---

# Controls "Fuzziness" of retrieval. 
# 0.0 = Exact Match (High Utility, Low Privacy)
# 0.1 - 0.5 = Balanced (Recommended)
# > 1.0 = High Noise (High Privacy, Low Utility)
PRIVACY_EPSILON = 0.1   

# Enable/Disable the Adversarial Attacker
ENABLE_RED_TEAM = True  

# Standard thresholds
MAX_NGRAM_OVERLAP = 0.5 # Stricter = Lower overlap allowed
```

---

## 4. Running the Pipeline

### Prerequisites

- Python 3.9+
- `pip install -r requirements.txt`
- (Optional) GPU for faster Red Teaming and LoRA training.

### Execution

Run the full end-to-end pipeline:

```bash
python3 src/main.py
# OR
python3 -m src.main
```

### What to Expect in Logs

You will see the Agentic system in action:

1. **"Loading Red Team Attacker..."**: The adversarial model initializes.
2. **"High Overlap (0.85)... CRITIC STEP"**: The Agent detects a leak and generates a critique.
3. **"Privacy Leak Detected by Red Team"**: The generic check passed, but the Red Team caught a subtle leak. The sample is discarded/flagged.

---

## 5. Directory Structure

- **`src/pipeline/generation.py`**: Contains the **Critic-Refiner** loop logic.
- **`src/pipeline/indexing.py`**: Contains the **Noisy Retrieval** logic.
- **`src/evaluation/red_team.py`**: Contains the **PrivacyAttacker** class.
- **`src/evaluation/privacy.py`**: Standard N-gram and Exact Match metrics.

---
*Built for the Advanced Agentic Coding Project.*
