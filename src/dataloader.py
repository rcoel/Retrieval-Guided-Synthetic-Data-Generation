import pandas as pd
from datasets import load_dataset, Dataset

def load_public_corpus(path):
    """Loads the public corpus from a CSV file."""
    print(f"Loading public corpus from {path}...")
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        print(f"Error: Public corpus file not found at {path}. Please create a dummy CSV with a 'text' column.")
        exit()


def load_private_dataset(dataset_name, subset=None):
    """Loads a private dataset from Hugging Face datasets."""
    print(f"Loading private dataset: {dataset_name} ({subset or 'default'})")
    dataset = load_dataset(dataset_name, subset)

    # For GLUE tasks, combine train and validation and rename columns for consistency
    if dataset_name == "glue":
        train_df = dataset["train"].to_pandas()
        val_df = dataset["validation"].to_pandas()
        full_df = pd.concat([train_df, val_df], ignore_index=True)
        # GLUE tasks have different column names, we standardize to 'text' and 'label'
        if "sentence" in full_df.columns:
            full_df = full_df.rename(columns={"sentence": "text"})
        elif "question" in full_df.columns:
             full_df = full_df.rename(columns={"question": "text"})
        # Add more mappings as needed for other GLUE tasks
    else:
        full_df = dataset["train"].to_pandas()

    return Dataset.from_pandas(full_df)