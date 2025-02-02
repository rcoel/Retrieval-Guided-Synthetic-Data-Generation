```python
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertModel, BertTokenizer
from sklearn.cluster import KMeans
import numpy as np
import torch.distributions.laplace as laplace_dist

# --- 1. Codebook Implementation ---

class Codebook:
    def __init__(self, codebook_size, embedding_dim):
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.codebook_vectors = None # Initialize as None, train later

    def train_codebook(self, embeddings):
        """Trains the codebook using k-means clustering on input embeddings."""
        kmeans = KMeans(n_clusters=self.codebook_size, random_state=42, n_init=10) # n_init to suppress warning
        kmeans.fit(embeddings)
        self.codebook_vectors = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)

    def quantize(self, embedding):
        """Quantizes an embedding by mapping it to the closest codebook vector."""
        if self.codebook_vectors is None:
            raise ValueError("Codebook not trained yet. Call train_codebook first.")

        distances = torch.cdist(embedding.unsqueeze(0), self.codebook_vectors.unsqueeze(0))[0] # Cosine distance could be used too
        closest_index = torch.argmin(distances)
        return closest_index # Or return self.codebook_vectors[closest_index] if you want the vector

    def dequantize(self, index):
        """Returns the codebook vector for a given index."""
        if self.codebook_vectors is None:
            raise ValueError("Codebook not trained yet. Call train_codebook first.")
        return self.codebook_vectors[index]

    def add_dp_noise(self, noise_scale):
        """Adds Laplace noise to the codebook vectors for differential privacy."""
        if self.codebook_vectors is None:
            raise ValueError("Codebook not trained yet. Call train_codebook first.")

        laplace = laplace_dist.Laplace(torch.tensor([0.0]), torch.tensor([noise_scale]))
        noise = laplace.sample(self.codebook_vectors.shape)
        self.codebook_vectors += noise

# --- 2. CMDP Encoder ---

class CMDPEncoder(nn.Module):
    def __init__(self, base_encoder, codebook, k, dp_noise_scale=0.0):
        super().__init__()
        self.base_encoder = base_encoder # e.g., BertModel
        self.codebook = codebook
        self.k = k # Mixing factor
        self.dp_noise_scale = dp_noise_scale

    def forward(self, input_ids, attention_mask):
        """
        Forward pass for CMDP encoder.

        Args:
            input_ids: Input token IDs.
            attention_mask: Attention mask.

        Returns:
            Mixed and DP-perturbed codebook representations.
        """
        batch_size, seq_len = input_ids.shape
        base_embeddings = self.base_encoder(input_ids, attention_mask=attention_mask).last_hidden_state # [batch_size, seq_len, embedding_dim]

        quantized_embeddings_indices = []
        for b in range(batch_size):
            seq_indices = []
            for s in range(seq_len):
                seq_indices.append(self.codebook.quantize(base_embeddings[b, s])) # Quantize each token embedding
            quantized_embeddings_indices.append(torch.stack(seq_indices)) # Stack indices for sequence
        quantized_embeddings_indices = torch.stack(quantized_embeddings_indices) # [batch_size, seq_len] - indices

        mixed_codebook_vectors = []
        for b in range(batch_size):
            seq_mixed_vectors = []
            for s in range(seq_len):
                mixed_vector = torch.zeros(self.codebook.embedding_dim) # Initialize mixed vector
                for _ in range(self.k): # Simple averaging of k codebook vectors
                    random_index = np.random.randint(batch_size) # Or select from other examples in batch, or synthetic examples
                    codebook_index = quantized_embeddings_indices[random_index, s]
                    mixed_vector += self.codebook.dequantize(codebook_index)
                mixed_vector /= self.k
                seq_mixed_vectors.append(mixed_vector)
            mixed_codebook_vectors.append(torch.stack(seq_mixed_vectors))
        mixed_codebook_vectors = torch.stack(mixed_codebook_vectors) # [batch_size, seq_len, embedding_dim]

        if self.dp_noise_scale > 0:
             laplace = laplace_dist.Laplace(torch.tensor([0.0]), torch.tensor([self.dp_noise_scale]))
             noise = laplace.sample(mixed_codebook_vectors.shape)
             mixed_codebook_vectors += noise

        return mixed_codebook_vectors


# --- 3. CMDP Model (Example for Sentence Classification) ---

class CMDPModelForClassification(nn.Module):
    def __init__(self, cmdp_encoder, num_labels):
        super().__init__()
        self.cmdp_encoder = cmdp_encoder
        self.classifier = nn.Linear(cmdp_encoder.codebook.embedding_dim, num_labels) # Simple linear classifier

    def forward(self, input_ids, attention_mask, labels=None):
        cmdp_representations = self.cmdp_encoder(input_ids, attention_mask)
        pooled_output = cmdp_representations[:, 0, :]  # Take [CLS] token representation for sentence classification
        logits = self.classifier(pooled_output)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return {"loss": loss, "logits": logits}

    @property
    def num_labels(self): # Helper property
        return self.classifier.out_features


# --- 4. Training and Inference Example ---

def train(model, train_dataloader, optimizer, num_epochs, device):
    model.to(device)
    model.train()
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, labels=labels)
            loss = outputs['loss']
            loss.backward()
            optimizer.step()
            print(f"Epoch: {epoch+1}, Loss: {loss.item():.4f}")

def inference(model, tokenizer, text, device):
    model.to(device)
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs['logits']
        predicted_class_id = torch.argmax(logits, dim=-1).item()
    return predicted_class_id


if __name__ == "__main__":
    # --- Hyperparameters ---
    codebook_size = 256
    embedding_dim = 768 # BERT base embedding dimension
    k_mixing_factor = 4
    dp_noise_scale = 0.1 # Example DP noise scale, tune this!
    num_labels = 2 # Example: Binary classification
    num_epochs = 3
    learning_rate = 2e-5
    batch_size = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 1. Prepare Data and Tokenizer ---
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Dummy training data (replace with your actual data)
    train_texts = ["This movie was great!", "The plot was terrible.", "Excellent acting.", "Predictable and boring."]
    train_labels = [1, 0, 1, 0] # 1: positive, 0: negative

    def create_dataloader(texts, labels, tokenizer, batch_size):
        encodings = tokenizer(texts, truncation=True, padding=True, return_tensors='pt')
        dataset = torch.utils.data.TensorDataset(encodings['input_ids'], encodings['attention_mask'], torch.tensor(labels))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return dataloader

    train_dataloader = create_dataloader(train_texts, train_labels, tokenizer, batch_size)


    # --- 2. Train Codebook (on a representative set of embeddings) ---
    # For demonstration, we'll use embeddings from the *training data itself*.
    # In a real scenario, you might use a larger, representative corpus embeddings.

    bert_model_for_codebook = BertModel.from_pretrained('bert-base-uncased') # Separate BERT just for codebook training
    all_train_embeddings = []
    with torch.no_grad():
        for batch in train_dataloader: # Just one epoch for example
            input_ids = batch[0]
            attention_mask = batch[1]
            batch_embeddings = bert_model_for_codebook(input_ids, attention_mask=attention_mask).last_hidden_state
            all_train_embeddings.append(batch_embeddings.view(-1, embedding_dim)) # Flatten to get token embeddings

    all_train_embeddings = torch.cat(all_train_embeddings, dim=0).numpy() # Convert to numpy for sklearn
    codebook = Codebook(codebook_size=codebook_size, embedding_dim=embedding_dim)
    codebook.train_codebook(all_train_embeddings)
    codebook.add_dp_noise(noise_scale=dp_noise_scale) # Add DP noise to the codebook vectors

    # --- 3. Initialize CMDP Encoder and Model ---
    base_bert_encoder = BertModel.from_pretrained('bert-base-uncased') # BERT for CMDP encoder
    cmdp_encoder = CMDPEncoder(base_encoder=base_bert_encoder, codebook=codebook, k=k_mixing_factor, dp_noise_scale=dp_noise_scale)
    model = CMDPModelForClassification(cmdp_encoder=cmdp_encoder, num_labels=num_labels)


    # --- 4. Training ---
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    train(model, train_dataloader, optimizer, num_epochs, device)


    # --- 5. Inference Example ---
    example_text = "This was an amazing film, I loved it!"
    predicted_class = inference(model, tokenizer, example_text, device)
    print(f"Example Text: '{example_text}'")
    print(f"Predicted Class ID: {predicted_class} (1: Positive, 0: Negative)")
```

**Explanation of the Python Implementation:**

1.  **`Codebook` Class:**
    *   `__init__`: Initializes the codebook with a specified size and embedding dimension. `codebook_vectors` is initially `None` and will be populated during training.
    *   `train_codebook(embeddings)`: Trains the codebook using K-means clustering from `sklearn.cluster.KMeans`. It takes a set of embeddings as input and clusters them into `codebook_size` clusters. The cluster centers become the `codebook_vectors`.
    *   `quantize(embedding)`:  Takes a single embedding and finds the closest vector in the `codebook_vectors` using cosine distance (you could switch to Euclidean distance if preferred). It returns the *index* of the closest codebook vector (or you could return the vector itself).
    *   `dequantize(index)`: Retrieves the codebook vector given its index.
    *   `add_dp_noise(noise_scale)`: Adds Laplace noise to the `codebook_vectors`. This is a simplified way to introduce DP conceptually. For a true DP implementation, you'd need to use a DP library and carefully manage privacy accounting.

2.  **`CMDPEncoder` Class:**
    *   `__init__`: Takes a `base_encoder` (like `BertModel`), the `codebook` object, the `k` mixing factor, and `dp_noise_scale`.
    *   `forward(input_ids, attention_mask)`:
        *   Passes the input through the `base_encoder` (BERT) to get the initial hidden states (embeddings).
        *   **Quantization:** Iterates through the batch and sequence length, quantizing each token embedding using `codebook.quantize()`. Stores the *indices* of the quantized embeddings.
        *   **Codebook-Based Mixing:** Iterates through the batch and sequence length again. For each token position:
            *   Initializes a `mixed_vector` to zero.
            *   Averages `k` codebook vectors. In this simplified example, it randomly selects `k` codebook vectors from *within the same batch* for mixing. In a more advanced implementation, you could mix with vectors from a larger dataset or synthetic vectors.
            *   Averages the selected codebook vectors to create `mixed_vector`.
        *   **DP Noise Addition:** If `dp_noise_scale > 0`, adds Laplace noise to the `mixed_codebook_vectors`.
        *   Returns the `mixed_codebook_vectors`.

3.  **`CMDPModelForClassification` Class:**
    *   A simple example model for sentence classification. It wraps the `CMDPEncoder` and adds a linear classifier on top of the `[CLS]` token representation (you can adapt this for other tasks).

4.  **`train()` and `inference()` Functions:**
    *   Standard PyTorch training and inference loops to demonstrate how to use the `CMDPModel`.

5.  **`if __name__ == "__main__":` Block (Example Usage):**
    *   **Hyperparameter Setup:** Sets example hyperparameters for codebook size, mixing, DP noise, training, etc. *These are just examples, you'll need to tune them for your specific task and dataset.*
    *   **Data and Tokenizer Preparation:** Loads the BERT tokenizer and creates dummy training data.  **Replace this with your actual dataset and data loading pipeline.**
    *   **Codebook Training:**
        *   Creates a *separate* `BertModel` (`bert_model_for_codebook`) to generate embeddings for codebook training. This is important; you train the codebook on *representative embeddings*, not the CMDP-encoded embeddings themselves.
        *   Iterates through the training dataloader (just one epoch in this example) to get embeddings.
        *   Trains the `Codebook` object using `codebook.train_codebook()`.
        *   Adds DP noise to the *trained codebook vectors* using `codebook.add_dp_noise()`.
    *   **CMDP Encoder and Model Initialization:**
        *   Loads the `BertModel` again (this will be the base encoder used *within* the `CMDPEncoder`).
        *   Initializes the `CMDPEncoder` with the base BERT model, the *trained and noisy* `codebook`, `k`, and `dp_noise_scale`.
        *   Initializes the `CMDPModelForClassification` with the `cmdp_encoder` and the number of labels for your classification task.
    *   **Training:** Sets up an AdamW optimizer and calls the `train()` function to fine-tune the `CMDPModel`.
    *   **Inference Example:** Shows how to use the `inference()` function to classify a single example text.

**To Use This Code:**

1.  **Install Libraries:** Make sure you have PyTorch, Transformers, scikit-learn, and NumPy installed:
    ```bash
    pip install torch transformers scikit-learn numpy
    ```
2.  **Replace Dummy Data:**  **Crucially, replace the dummy `train_texts` and `train_labels` with your actual training dataset and data loading pipeline.**
3.  **Tune Hyperparameters:** Experiment with different values for `codebook_size`, `k_mixing_factor`, `dp_noise_scale`, learning rate, etc., to find the best balance between privacy and performance for your specific NLP task.
4.  **Evaluate Privacy and Performance:**  Implement privacy evaluation metrics (like those used in TextFusion, TextMixer, TextHide, or more formal DP metrics) and evaluate the performance of your CMDP model on your task.
5.  **Consider Advanced DP Techniques:** For a true differentially private system, replace the simplified `codebook.add_dp_noise()` with a proper DP implementation using libraries like `opacus` or `diffprivlib`. You'll need to perform sensitivity analysis and manage privacy accounting correctly.

This Python code provides a starting point for implementing and experimenting with the CMDP methodology. Remember that this is a simplified demonstration, and further research, development, and rigorous evaluation would be needed to fully realize the potential of CMDP.