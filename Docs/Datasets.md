# PrivaSyn — Supported Datasets

## Dataset Registry

The framework supports multiple datasets via `src/dataset_registry.py`. Each dataset is auto-downloaded from HuggingFace with standardized column mapping.

| Key | Dataset | Task | Labels | HuggingFace Source |
|-----|---------|------|--------|--------------------|
| `sst2` | Stanford Sentiment Treebank v2 | Sentiment | 2 (positive/negative) | [glue/sst2](https://huggingface.co/datasets/nyu-mll/glue) |
| `ag_news` | AG News Topic Classification | Topic | 4 (World/Sports/Business/Sci-Tech) | [fancyzhx/ag_news](https://huggingface.co/datasets/fancyzhx/ag_news) |
| `imdb` | IMDB Movie Review | Sentiment | 2 (positive/negative) | [stanfordnlp/imdb](https://huggingface.co/datasets/stanfordnlp/imdb) |

## Usage

```bash
# List available datasets
python run_experiment.py --list-datasets

# Run on specific dataset
python run_experiment.py --config experiments/configs/full_pipeline.yaml --dataset ag_news
```

## Adding a New Dataset

Add an entry to `DATASET_REGISTRY` in `src/dataset_registry.py`:

```python
"your_dataset": DatasetSpec(
    source="huggingface/source",
    subset=None,
    text_column="text",        # Column containing text
    label_column="label",      # Column containing labels
    task_type="classification",
    num_labels=3,
    splits=["train", "test"],
    description="Your Dataset Description",
    public_corpus_query="domain keywords for Wikipedia corpus",
)
```

## Public Corpus

Each dataset auto-generates a domain-matched public corpus from Wikipedia if `data/corpus.csv` doesn't exist. The corpus is used for RAG retrieval during generation.
