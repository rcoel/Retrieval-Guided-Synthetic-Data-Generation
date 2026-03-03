"""
Enhanced Adaptive RAG Generation Pipeline.

Novel features:
1. Chain-of-Thought (CoT) Critic prompting with structured JSON output
2. Perplexity-gated quality control (coherence filtering)
3. Adaptive temperature scheduling (cosine annealing within retry loop)
4. Rényi DP budget integration
5. Multi-agent separation of concerns (Generator, Critic, Attacker)

Structural improvements:
- Uses shared model_loader (no duplicate BitsAndBytesConfig)
- Prompt logic imported from prompts.py (no circular imports)
- Critic extracted to critic.py (separation of concerns)
- Proper logging via logger module
- Full type hints
"""

from __future__ import annotations
import torch
import math
from typing import List, Dict, Any, Optional
from peft import PeftModel
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import numpy as np

from .. import config
from ..utils import get_device
from ..logger import get_logger
from ..model_loader import load_causal_model, load_tokenizer
from ..evaluation.quality import measure_similarity_batch
from ..evaluation.privacy import calculate_single_pair_ngram_overlap
from ..evaluation.red_team import PrivacyAttacker
from .prompts import create_prompt
from .critic import CriticAgent

logger = get_logger("rag_pipeline.generation")

# Conditionally import privacy budget manager
if config.ENABLE_DP_ACCOUNTING:
    from ..privacy_budget import RenyiDPAccountant, PrivacyBudgetExhaustedError


def compute_adaptive_temperature(
    attempt: int,
    max_retries: int,
    failure_type: str,
    schedule: str = "cosine",
) -> float:
    """
    Adaptive temperature scheduling within the retry loop.
    
    Uses cosine annealing to smoothly adjust temperature based on
    the retry attempt and the type of failure encountered.
    
    Privacy failures  → increase temperature (more randomness)
    Utility failures  → decrease temperature (more adherence)
    Coherence failures → slight decrease (more fluency)
    
    Args:
        attempt: Current retry attempt (0-indexed).
        max_retries: Maximum number of retries.
        failure_type: "privacy", "utility", or "coherence".
        schedule: "cosine", "linear", or "fixed".
        
    Returns:
        Temperature value for this attempt.
    """
    if schedule == "fixed":
        return config.GENERATION_TEMP

    progress = attempt / max(max_retries, 1)

    if schedule == "cosine":
        cos_factor = 0.5 * (1 + math.cos(math.pi * progress))
    else:  # linear
        cos_factor = 1.0 - progress

    temp_map = {
        "privacy": config.GENERATION_TEMP + (config.TEMP_MAX - config.GENERATION_TEMP) * (1 - cos_factor),
        "utility": config.GENERATION_TEMP - (config.GENERATION_TEMP - config.TEMP_MIN) * (1 - cos_factor),
        "coherence": config.GENERATION_TEMP - 0.5 * (config.GENERATION_TEMP - config.TEMP_MIN) * (1 - cos_factor),
    }
    temp = temp_map.get(failure_type, config.GENERATION_TEMP)
    return max(config.TEMP_MIN, min(config.TEMP_MAX, temp))


def compute_perplexity(model, tokenizer, text: str, device: str) -> float:
    """
    Compute perplexity of generated text as a coherence measure.
    
    Args:
        model: The language model.
        tokenizer: The tokenizer.
        text: The generated text to evaluate.
        device: Compute device.
        
    Returns:
        Perplexity score (lower = more coherent).
    """
    if not text.strip():
        return float('inf')
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
    
    return math.exp(loss.item()) if loss.item() < 100 else float('inf')


class SyntheticDataGenerator:
    """
    Generates privacy-preserving synthetic data using an Adaptive RAG loop
    with multi-agent feedback (Generator, Critic, Red Team Attacker).
    """

    def __init__(self, base_model_name: str, lora_adapter_path: str):
        self.device = get_device()
        
        # Load model via shared loader (eliminates duplication)
        base_model = load_causal_model(base_model_name, quantize=True)
        self.model = PeftModel.from_pretrained(base_model, lora_adapter_path)
        self.tokenizer = load_tokenizer(base_model_name)
        
        logger.info(f"Loading embedding model: {config.EMBEDDING_MODEL}")
        self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
        
        # Initialize Critic Agent (uses same model instance)
        self.critic = CriticAgent(self.model, self.tokenizer, self.device)
        
        # Initialize Red Team Attacker
        self.red_team: Optional[PrivacyAttacker] = None
        if config.ENABLE_RED_TEAM:
            self.red_team = PrivacyAttacker(base_model_name)

        # Initialize DP accountant
        self.dp_accountant: Optional[RenyiDPAccountant] = None
        if config.ENABLE_DP_ACCOUNTING:
            self.dp_accountant = RenyiDPAccountant(
                total_budget=config.TOTAL_PRIVACY_BUDGET,
                delta=config.RDP_DELTA,
                alpha=config.RDP_ALPHA,
            )
            logger.info(
                f"DP Accountant initialized: budget={config.TOTAL_PRIVACY_BUDGET}, "
                f"α={config.RDP_ALPHA}"
            )

        # Metrics tracking
        self.generation_metrics: Dict[str, Any] = {
            "total_samples": 0,
            "total_retries": 0,
            "privacy_failures": 0,
            "utility_failures": 0,
            "coherence_failures": 0,
            "red_team_catches": 0,
            "temperature_trajectory": [],
            "dp_budget_used": 0.0,
        }

    def generate(
        self,
        private_data,
        public_passages: List[str],
        retriever,
    ) -> List[Dict[str, Any]]:
        """
        Generates synthetic data using the enhanced Adaptive RAG pipeline.
        
        Pipeline: Retrieve → Generate → Quality Gate → Critic → Red Team → Accept/Retry
        
        Args:
            private_data: Iterable of private examples with 'text' and 'label' keys.
            public_passages: List of public text passages for context.
            retriever: SemanticIndexer instance for retrieval.
            
        Returns:
            List of synthetic sample dictionaries.
        """
        synthetic_samples: List[Dict[str, Any]] = []
        batch_size = config.BATCH_SIZE_GENERATION
        data_list = list(private_data)
        
        logger.info(f"Starting generation: {len(data_list)} samples, batch_size={batch_size}")
        
        for i in tqdm(range(0, len(data_list), batch_size), desc="Batched Generation"):
            # Check DP budget before each batch
            if self.dp_accountant and not self.dp_accountant.can_query(config.PRIVACY_EPSILON):
                logger.warning(f"DP budget exhausted after {i} samples. Stopping.")
                break

            batch_examples = data_list[i : i + batch_size]
            batch_private_texts = [ex['text'] for ex in batch_examples]
            
            # Record DP consumption
            if self.dp_accountant:
                try:
                    self.dp_accountant.consume(config.PRIVACY_EPSILON, f"retrieval_batch_{i}")
                except PrivacyBudgetExhaustedError as e:
                    logger.warning(str(e))
                    break
            
            # 1. Retrieve Context (with DP noise)
            retrieved_indices = retriever.retrieve(
                batch_private_texts,
                k=config.NUM_RETRIEVED_DOCS_K,
                privacy_epsilon=config.PRIVACY_EPSILON,
            )
            
            batch_context_docs = []
            for j, _text in enumerate(batch_private_texts):
                indices = retrieved_indices[j]
                batch_context_docs.append([public_passages[idx] for idx in indices])
            
            # 2. Adaptive Agentic Loop
            batch_results = self._run_agentic_loop(
                batch_examples, batch_private_texts, batch_context_docs
            )
            
            self.generation_metrics["total_samples"] += len(batch_examples)
            synthetic_samples.extend(batch_results)

        # Finalize metrics
        if self.dp_accountant:
            self.generation_metrics["dp_budget_used"] = self.dp_accountant.epsilon_spent
            self.generation_metrics["dp_budget_remaining"] = self.dp_accountant.budget_remaining

        self._log_summary()
        return synthetic_samples

    def _run_agentic_loop(
        self,
        batch_examples: List[dict],
        batch_texts: List[str],
        batch_context: List[List[str]],
    ) -> List[Dict[str, Any]]:
        """
        Run the self-correction loop for a single batch.
        
        Args:
            batch_examples: Raw examples with 'text' and 'label'.
            batch_texts: Private texts for this batch.
            batch_context: Retrieved context docs for each sample.
            
        Returns:
            List of result dictionaries.
        """
        n = len(batch_examples)
        active_indices = list(range(n))
        final_results: List[Optional[Dict]] = [None] * n
        feedbacks: List[Optional[str]] = [None] * n
        failure_types: List[str] = ["privacy"] * n

        for attempt in range(config.MAX_RETRIES + 1):
            if not active_indices:
                break

            # Compute adaptive temperatures
            temps = [
                compute_adaptive_temperature(
                    attempt, config.MAX_RETRIES, failure_types[idx], config.TEMP_SCHEDULE
                )
                for idx in active_indices
            ]
            batch_temp = sum(temps) / len(temps)
            
            # Track temperature trajectory
            for idx, temp in zip(active_indices, temps):
                self.generation_metrics["temperature_trajectory"].append({
                    "sample": idx, "attempt": attempt,
                    "temp": temp, "failure_type": failure_types[idx]
                })

            # Build prompts with feedback
            active_prompts = [
                create_prompt(batch_texts[idx], batch_context[idx], feedbacks[idx])
                for idx in active_indices
            ]

            # Generate
            inputs = self.tokenizer(
                active_prompts, return_tensors="pt",
                padding=True, truncation=True,
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=config.MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=batch_temp,
                    top_p=config.GENERATION_TOP_P,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            
            decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

            # Evaluate each output
            still_active = []
            for local_idx, global_idx in enumerate(active_indices):
                response = decoded[local_idx].split("[ASSISTANT]")[-1].strip()
                original = batch_texts[global_idx]
                
                result = self._evaluate_sample(
                    original, response, batch_examples[global_idx],
                    global_idx, attempt, batch_temp
                )
                
                if result is not None:
                    final_results[global_idx] = result
                else:
                    still_active.append(global_idx)

                # Force accept on last retry
                if result is None and attempt == config.MAX_RETRIES:
                    final_results[global_idx] = {
                        "original_text": original,
                        "synthetic_text": response,
                        "label": batch_examples[global_idx]['label'],
                        "num_retries": attempt,
                        "generation_temp": batch_temp,
                        "forced_accept": True,
                    }
            
            active_indices = [idx for idx in still_active if final_results[idx] is None]

        return [r for r in final_results if r is not None]

    def _evaluate_sample(
        self,
        original: str,
        response: str,
        example: dict,
        idx: int,
        attempt: int,
        temp: float,
    ) -> Optional[Dict[str, Any]]:
        """
        Evaluate a single generated sample against the quality triangle.
        
        Returns the sample dict if it passes all checks, None if it needs retry.
        """
        # Privacy: N-gram overlap
        ngram_overlap = calculate_single_pair_ngram_overlap(original, response)
        
        # Utility: Semantic similarity
        sim_score = measure_similarity_batch([original], [response], self.embedding_model)[0]
        
        # Coherence: Perplexity gate
        perplexity = float('inf')
        if config.ENABLE_PERPLEXITY_GATE:
            perplexity = compute_perplexity(self.model, self.tokenizer, response, self.device)

        candidate = {
            "original_text": original,
            "synthetic_text": response,
            "label": example['label'],
            "ngram_overlap": ngram_overlap,
            "semantic_sim": sim_score,
            "perplexity": perplexity,
            "generation_temp": temp,
            "num_retries": attempt,
        }

        # Check 1: Privacy
        if ngram_overlap > config.MAX_NGRAM_OVERLAP:
            self.generation_metrics["privacy_failures"] += 1
            self._request_critique(idx, original, response, "High Text Overlap/Privacy Violation", "privacy")
            return None

        # Check 2: Utility
        if sim_score < config.MIN_SEMANTIC_SIM:
            self.generation_metrics["utility_failures"] += 1
            self._request_critique(idx, original, response, "Low Semantic Consistency/Meaning Lost", "utility")
            return None

        # Check 3: Coherence
        if config.ENABLE_PERPLEXITY_GATE and perplexity > config.MAX_PERPLEXITY_THRESHOLD:
            self.generation_metrics["coherence_failures"] += 1
            # No model-based critique needed — template feedback
            return None

        # Check 4: Red Team
        if self.red_team:
            compromised, reasoning = self.red_team.attack(response)
            if compromised:
                self.generation_metrics["red_team_catches"] += 1
                return None

        return candidate

    def _request_critique(
        self, idx: int, original: str, response: str,
        issue_type: str, failure_type: str,
    ) -> None:
        """Request structured critique from the Critic Agent."""
        self.critic.critique(original, response, issue_type)

    def _log_summary(self) -> None:
        """Log a summary of the generation process."""
        m = self.generation_metrics
        logger.info("=" * 60)
        logger.info("  GENERATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"  Total samples:      {m['total_samples']}")
        logger.info(f"  Total retries:      {m['total_retries']}")
        logger.info(f"  Privacy failures:   {m['privacy_failures']}")
        logger.info(f"  Utility failures:   {m['utility_failures']}")
        logger.info(f"  Coherence failures: {m['coherence_failures']}")
        logger.info(f"  Red team catches:   {m['red_team_catches']}")
        if self.dp_accountant:
            logger.info(f"  DP budget used:     {m.get('dp_budget_used', 0):.4f}")
            logger.info(f"  DP budget left:     {m.get('dp_budget_remaining', 0):.4f}")

    def get_metrics(self) -> Dict[str, Any]:
        """Return generation metrics for logging."""
        return self.generation_metrics

    def get_dp_report(self) -> Dict[str, Any]:
        """Return the DP accountant's audit report."""
        if self.dp_accountant:
            return self.dp_accountant.get_report()
        return {"status": "DP accounting disabled"}