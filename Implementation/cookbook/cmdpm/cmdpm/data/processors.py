import pandas as pd
from sklearn.model_selection import train_test_split

def load_dataset(data_path, val_split=0.2):
    df = pd.read_csv(data_path)
    texts = df["text"].tolist()
    labels = df["label"].tolist()
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=val_split, random_state=42
    )
    return train_texts, train_labels, val_texts, val_labels