import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm
from .. import config
from ..utils import get_device
from sentence_transformers import SentenceTransformer
from ..evaluation.quality import measure_similarity_batch
from ..evaluation.privacy import calculate_single_pair_ngram_overlap
import numpy as np

def create_prompt(private_example, context_docs):
    """Creates the structured prompt for generation."""
    context_str = "\n".join([f"{i+1}. {doc}" for i, doc in enumerate(context_docs)])
    
    prompt = f"""[SYSTEM] You are an AI assistant that generates a high-quality, semantically equivalent variant of a given text example. The new variant should retain the original's intent, meaning, and key information but should not be an exact copy.

[USER]
### CONTEXT FROM PUBLIC DOCUMENTS:
{context_str}

### TASK:
Generate a synthetic variant of the following example.

### ORIGINAL PRIVATE EXAMPLE:
{private_example}

[ASSISTANT]"""
    return prompt

class SyntheticDataGenerator:
    def __init__(self, base_model_name, lora_adapter_path):
        self.device = get_device()
        
        if torch.cuda.is_available():
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            quantization_config = bnb_config
        else:
            quantization_config = None

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True
        )
        self.model = PeftModel.from_pretrained(base_model, lora_adapter_path)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"Loading embedding model {config.EMBEDDING_MODEL} for self-correction...")
        self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)

    def generate(self, private_data, public_passages, retriever):
        """Generates synthetic data for the entire private dataset using batched Adaptive RAG."""
        synthetic_samples = []
        batch_size = config.BATCH_SIZE_GENERATION
        
        print(f"Generating synthetic data in batches of {batch_size} with self-correction...")
        
        # Convert dataset to list for easier batching
        data_list = list(private_data)
        
        for i in tqdm(range(0, len(data_list), batch_size), desc="Batched Generation"):
            batch_examples = data_list[i : i + batch_size]
            
            # 1. Retrieve Context for Batch
            batch_private_texts = [ex['text'] for ex in batch_examples]
            # retrieve expects list of strings, returns list of list of indices ?? 
            # Check indexing.py: retrieve returns indices (list of lists if k>1? No)
            # indexing.py: _distances, indices = self.index.search(query_embeddings, k)
            # indices is (n_queries, k) array.
            
            retrieved_indices_batch = retriever.retrieve(batch_private_texts, k=config.NUM_RETRIEVED_DOCS_K)
            
            batch_prompts = []
            batch_context_docs = []
            
            for j, private_text in enumerate(batch_private_texts):
                indices = retrieved_indices_batch[j]
                context_docs = [public_passages[idx] for idx in indices]
                batch_context_docs.append(context_docs)
                batch_prompts.append(create_prompt(private_text, context_docs))
            
            # 2. Adaptive Generation Loop
            # Track which items in the batch are "done"
            # We initialize active_indices pointing to local batch indices 0..len(batch)-1
            active_indices = list(range(len(batch_examples)))
            final_results = [None] * len(batch_examples)
            
            # Initial params
            current_temps = [config.GENERATION_TEMP] * len(batch_examples)
            current_top_p = [config.GENERATION_TOP_P] * len(batch_examples)
            
            for attempt in range(config.MAX_RETRIES + 1):
                if not active_indices:
                    break
                
                # Prepare inputs for active items
                active_prompts = [batch_prompts[idx] for idx in active_indices]
                inputs = self.tokenizer(active_prompts, return_tensors="pt", padding=True, truncation=True).to(self.device)
                
                # Generate (handling different params is tricky in batch if they vary per item)
                # Ideally, we group by params, but for simplicity in this "Novel" architecture, 
                # we might just use the params of the first item or average? 
                # OR simpler: Use the params of the majority? 
                # Actually, HuggingFace generate doesn't support per-sample temp/top_p.
                # Compromise: We will set params based on the "average" need or just use the modified params 
                # for the WHOLE active batch if we want to be strict, but that affects others.
                # BETTER APPROACH for Batching: Use a common "retry" strategy.
                # If we increase temp, we increase it for the whole retry batch.
                
                # Check mean temp of active indices
                mean_temp = np.mean([current_temps[idx] for idx in active_indices])
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=config.MAX_NEW_TOKENS,
                        do_sample=True,
                        temperature=mean_temp, 
                        top_p=config.GENERATION_TOP_P, # Keep top_p constant or similar logic
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                decoded_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                
                # Process outputs
                for local_idx_in_active, global_idx_in_batch in enumerate(active_indices):
                    full_text = decoded_outputs[local_idx_in_active]
                    assistant_response = full_text.split("[ASSISTANT]")[-1].strip()
                    original_text = batch_private_texts[global_idx_in_batch]
                    
                    # 3. Assess Quality & Privacy
                    ngram_overlap = calculate_single_pair_ngram_overlap(original_text, assistant_response)
                    # For semantic similarity, we need a batch check, but here we do one by one or 
                    # we can batch the check after generation. 
                    # Let's do single check here to keep logic simple inside the loop since active set changes.
                    # Actually measure_similarity_batch is better.
                    
                    # Store candidate temporarily
                    candidate = {
                        "original_text": original_text,
                        "synthetic_text": assistant_response,
                        "label": batch_examples[global_idx_in_batch]['label'],
                        "ngram_overlap": ngram_overlap
                    }
                    
                    # Similarity check (we'll do it individually here for code simplicity, 
                    # though less efficient than massive batch, but ok for batch=8)
                    sim_score = measure_similarity_batch([original_text], [assistant_response], self.embedding_model)[0]
                    candidate["semantic_sim"] = sim_score
                    
                    # 4. Feedback Logic
                    passed = True
                    feedback_msg = "Success"
                    
                    if ngram_overlap > config.MAX_NGRAM_OVERLAP:
                        passed = False
                        feedback_msg = f"High Overlap ({ngram_overlap:.2f})"
                        # Increase temp to encourage diversity
                        current_temps[global_idx_in_batch] = min(1.5, current_temps[global_idx_in_batch] + 0.2)
                        
                    elif sim_score < config.MIN_SEMANTIC_SIM:
                        passed = False
                        feedback_msg = f"Low Similarity ({sim_score:.2f})"
                        # Decrease temp to encourage focus
                        current_temps[global_idx_in_batch] = max(0.1, current_temps[global_idx_in_batch] - 0.2)
                    
                    # If passed or last retry, accept it
                    if passed or attempt == config.MAX_RETRIES:
                        final_results[global_idx_in_batch] = candidate
                    else:
                        # Keep in active indices for next round
                        pass
                
                # Update active indices
                active_indices = [idx for idx in active_indices if final_results[idx] is None]
            
            synthetic_samples.extend(final_results)
            
        return synthetic_samples