# Set these to a small number for quick testing, or None to use the full dataset
MAX_TRAIN_SAMPLES = 4
MAX_GENERATION_SAMPLES = 4
MAX_EVAL_SAMPLES = 4

# --- File Paths ---
# Create a dummy CSV with a 'text' column for this to run
PUBLIC_CORPUS_PATH = "data/corpus.csv"
PRIVATE_DATA_NAME = "glue"
PRIVATE_DATA_SUBSET = "sst2"
OUTPUT_DIR = "output/"
FAISS_INDEX_PATH = OUTPUT_DIR + "faiss_index.bin"
SYNTHETIC_DATA_PATH = OUTPUT_DIR + "synthetic_data.jsonl"
MODEL_OUTPUT_DIR = OUTPUT_DIR + "models/"
ADAPTER_PATH = MODEL_OUTPUT_DIR + "final_lora_adapter"

# --- Model Identifiers (from Hugging Face Hub) ---
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BASE_LLM_MODEL = "Qwen/Qwen2-0.5B-Instruct" # Using a smaller model for accessibility
CLASSIFIER_MODEL = "bert-base-uncased"

# --- Indexing Parameters ---
CHUNK_SIZE = 256
CHUNK_OVERLAP = 128
EMBEDDING_DIM = 768 # Dimension for all-mpnet-base-v2

# --- Fine-Tuning (PEFT) Parameters ---
LORA_R = 8
LORA_ALPHA = 16
LORA_TARGET_MODULES = ["q_proj", "v_proj"]
LEARNING_RATE = 1e-4
NUM_EPOCHS = 1 # Keep low for demonstration
BATCH_SIZE = 2 # Adjust based on your GPU memory
WARMUP_STEPS = 5

# --- Generation Parameters ---
NUM_RETRIEVED_DOCS_K = 2
GENERATION_TEMP = 0.8
GENERATION_TOP_P = 0.9
MAX_NEW_TOKENS = 64 # Max length of the synthetic sample
MAX_PERPLEXITY = 50.0

# --- Evaluation Parameters ---
EVAL_BATCH_SIZE = 16

# --- Adaptive RAG (Self-Correction) Parameters ---
MAX_NGRAM_OVERLAP = 0.5 # Max allowed 5-gram overlap ratio (Privacy)
MIN_SEMANTIC_SIM = 0.7  # Min required semantic similarity (Utility)
MAX_RETRIES = 3         # Max attempts to self-correct
BATCH_SIZE_GENERATION = 8 # Batch size for generation loop