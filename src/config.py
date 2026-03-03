"""
Pipeline Configuration — Typed, Validated, Serializable.

All hyperparameters are centralized here as a dataclass with
validation, type hints, and serialization support.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Optional
import os


@dataclass
class PipelineConfig:
    """Typed configuration for the entire RAG pipeline."""

    # --- Sample Limits (set small for testing, None for full dataset) ---
    MAX_TRAIN_SAMPLES: int = 4
    MAX_GENERATION_SAMPLES: int = 4
    MAX_EVAL_SAMPLES: int = 4

    # --- File Paths ---
    PUBLIC_CORPUS_PATH: str = "data/corpus.csv"
    PRIVATE_DATA_NAME: str = "glue"
    PRIVATE_DATA_SUBSET: str = "sst2"
    OUTPUT_DIR: str = "output/"

    # --- Model Identifiers (HuggingFace Hub) ---
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    BASE_LLM_MODEL: str = "Qwen/Qwen2-0.5B-Instruct"
    CLASSIFIER_MODEL: str = "bert-base-uncased"

    # --- Indexing Parameters ---
    CHUNK_SIZE: int = 256
    CHUNK_OVERLAP: int = 128
    EMBEDDING_DIM: int = 384

    # --- Fine-Tuning (PEFT / LoRA) ---
    LORA_R: int = 8
    LORA_ALPHA: int = 16
    LORA_TARGET_MODULES: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    LORA_DROPOUT: float = 0.1
    LEARNING_RATE: float = 1e-4
    NUM_EPOCHS: int = 1
    BATCH_SIZE: int = 2
    WARMUP_STEPS: int = 5

    # --- Generation ---
    NUM_RETRIEVED_DOCS_K: int = 2
    GENERATION_TEMP: float = 0.8
    GENERATION_TOP_P: float = 0.9
    MAX_NEW_TOKENS: int = 64

    # --- Evaluation ---
    EVAL_BATCH_SIZE: int = 16

    # --- Adaptive RAG (Self-Correction) ---
    MAX_NGRAM_OVERLAP: float = 0.5
    MIN_SEMANTIC_SIM: float = 0.7
    MAX_RETRIES: int = 3
    BATCH_SIZE_GENERATION: int = 8

    # --- Advanced Privacy Features ---
    PRIVACY_EPSILON: float = 0.1
    ENABLE_RED_TEAM: bool = True

    # --- Rényi DP Accounting ---
    ENABLE_DP_ACCOUNTING: bool = True
    TOTAL_PRIVACY_BUDGET: float = 10.0
    RDP_ALPHA: float = 5.0
    RDP_DELTA: float = 1e-5

    # --- Perplexity-Gated Quality Control ---
    ENABLE_PERPLEXITY_GATE: bool = True
    MAX_PERPLEXITY_THRESHOLD: float = 50.0

    # --- Adaptive Temperature Scheduling ---
    TEMP_MIN: float = 0.5
    TEMP_MAX: float = 1.5
    TEMP_SCHEDULE: str = "cosine"  # "cosine" | "linear" | "fixed"

    # --- Experiment Logging ---
    ENABLE_METRICS_LOGGING: bool = True
    
    # --- Reproducibility ---
    SEED: int = 42

    def __post_init__(self):
        """Validate configuration values."""
        assert self.CHUNK_OVERLAP < self.CHUNK_SIZE, \
            f"CHUNK_OVERLAP ({self.CHUNK_OVERLAP}) must be < CHUNK_SIZE ({self.CHUNK_SIZE})"
        assert 0.0 < self.PRIVACY_EPSILON, \
            f"PRIVACY_EPSILON must be positive, got {self.PRIVACY_EPSILON}"
        assert self.TEMP_MIN <= self.GENERATION_TEMP <= self.TEMP_MAX, \
            f"GENERATION_TEMP ({self.GENERATION_TEMP}) must be within [{self.TEMP_MIN}, {self.TEMP_MAX}]"
        assert self.TEMP_SCHEDULE in ("cosine", "linear", "fixed"), \
            f"TEMP_SCHEDULE must be 'cosine', 'linear', or 'fixed', got '{self.TEMP_SCHEDULE}'"
        assert 0.0 < self.MAX_NGRAM_OVERLAP <= 1.0, \
            f"MAX_NGRAM_OVERLAP must be in (0, 1], got {self.MAX_NGRAM_OVERLAP}"
        assert 0.0 <= self.MIN_SEMANTIC_SIM <= 1.0, \
            f"MIN_SEMANTIC_SIM must be in [0, 1], got {self.MIN_SEMANTIC_SIM}"
        assert self.RDP_ALPHA > 1.0, \
            f"RDP_ALPHA must be > 1, got {self.RDP_ALPHA}"

    @property
    def FAISS_INDEX_PATH(self) -> str:
        return os.path.join(self.OUTPUT_DIR, "faiss_index.bin")

    @property
    def SYNTHETIC_DATA_PATH(self) -> str:
        return os.path.join(self.OUTPUT_DIR, "synthetic_data.jsonl")

    @property
    def MODEL_OUTPUT_DIR(self) -> str:
        return os.path.join(self.OUTPUT_DIR, "models/")

    @property
    def ADAPTER_PATH(self) -> str:
        return os.path.join(self.MODEL_OUTPUT_DIR, "final_lora_adapter")

    @property
    def METRICS_LOG_PATH(self) -> str:
        return os.path.join(self.OUTPUT_DIR, "experiment_metrics.json")

    def to_dict(self) -> dict:
        """Serialize config to dictionary (for JSON logging)."""
        d = asdict(self)
        # Add computed properties
        d["FAISS_INDEX_PATH"] = self.FAISS_INDEX_PATH
        d["SYNTHETIC_DATA_PATH"] = self.SYNTHETIC_DATA_PATH
        d["MODEL_OUTPUT_DIR"] = self.MODEL_OUTPUT_DIR
        d["ADAPTER_PATH"] = self.ADAPTER_PATH
        d["METRICS_LOG_PATH"] = self.METRICS_LOG_PATH
        return d


# ---------------------------------------------------------------------------
# Module-level singleton — backward compatible with `from src import config`
# All other modules can do `config.GENERATION_TEMP` as before.
# ---------------------------------------------------------------------------
_cfg = PipelineConfig()

# Export all attributes at module level for backward compatibility
MAX_TRAIN_SAMPLES = _cfg.MAX_TRAIN_SAMPLES
MAX_GENERATION_SAMPLES = _cfg.MAX_GENERATION_SAMPLES
MAX_EVAL_SAMPLES = _cfg.MAX_EVAL_SAMPLES
PUBLIC_CORPUS_PATH = _cfg.PUBLIC_CORPUS_PATH
PRIVATE_DATA_NAME = _cfg.PRIVATE_DATA_NAME
PRIVATE_DATA_SUBSET = _cfg.PRIVATE_DATA_SUBSET
OUTPUT_DIR = _cfg.OUTPUT_DIR
FAISS_INDEX_PATH = _cfg.FAISS_INDEX_PATH
SYNTHETIC_DATA_PATH = _cfg.SYNTHETIC_DATA_PATH
MODEL_OUTPUT_DIR = _cfg.MODEL_OUTPUT_DIR
ADAPTER_PATH = _cfg.ADAPTER_PATH
EMBEDDING_MODEL = _cfg.EMBEDDING_MODEL
BASE_LLM_MODEL = _cfg.BASE_LLM_MODEL
CLASSIFIER_MODEL = _cfg.CLASSIFIER_MODEL
CHUNK_SIZE = _cfg.CHUNK_SIZE
CHUNK_OVERLAP = _cfg.CHUNK_OVERLAP
EMBEDDING_DIM = _cfg.EMBEDDING_DIM
LORA_R = _cfg.LORA_R
LORA_ALPHA = _cfg.LORA_ALPHA
LORA_TARGET_MODULES = _cfg.LORA_TARGET_MODULES
LORA_DROPOUT = _cfg.LORA_DROPOUT
LEARNING_RATE = _cfg.LEARNING_RATE
NUM_EPOCHS = _cfg.NUM_EPOCHS
BATCH_SIZE = _cfg.BATCH_SIZE
WARMUP_STEPS = _cfg.WARMUP_STEPS
NUM_RETRIEVED_DOCS_K = _cfg.NUM_RETRIEVED_DOCS_K
GENERATION_TEMP = _cfg.GENERATION_TEMP
GENERATION_TOP_P = _cfg.GENERATION_TOP_P
MAX_NEW_TOKENS = _cfg.MAX_NEW_TOKENS
EVAL_BATCH_SIZE = _cfg.EVAL_BATCH_SIZE
MAX_NGRAM_OVERLAP = _cfg.MAX_NGRAM_OVERLAP
MIN_SEMANTIC_SIM = _cfg.MIN_SEMANTIC_SIM
MAX_RETRIES = _cfg.MAX_RETRIES
BATCH_SIZE_GENERATION = _cfg.BATCH_SIZE_GENERATION
PRIVACY_EPSILON = _cfg.PRIVACY_EPSILON
ENABLE_RED_TEAM = _cfg.ENABLE_RED_TEAM
ENABLE_DP_ACCOUNTING = _cfg.ENABLE_DP_ACCOUNTING
TOTAL_PRIVACY_BUDGET = _cfg.TOTAL_PRIVACY_BUDGET
RDP_ALPHA = _cfg.RDP_ALPHA
RDP_DELTA = _cfg.RDP_DELTA
ENABLE_PERPLEXITY_GATE = _cfg.ENABLE_PERPLEXITY_GATE
MAX_PERPLEXITY_THRESHOLD = _cfg.MAX_PERPLEXITY_THRESHOLD
TEMP_MIN = _cfg.TEMP_MIN
TEMP_MAX = _cfg.TEMP_MAX
TEMP_SCHEDULE = _cfg.TEMP_SCHEDULE
ENABLE_METRICS_LOGGING = _cfg.ENABLE_METRICS_LOGGING
METRICS_LOG_PATH = _cfg.METRICS_LOG_PATH
SEED = _cfg.SEED