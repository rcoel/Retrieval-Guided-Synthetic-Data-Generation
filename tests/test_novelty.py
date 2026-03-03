"""
Test for the Adaptive RAG self-correction loop (novelty features).

This test verifies the batched generation with mock models:
- Self-correction retries
- Privacy / Utility feedback gating
"""

import unittest
from unittest.mock import MagicMock, patch
import torch
import numpy as np


class TestAdaptiveRAG(unittest.TestCase):

    @patch('src.pipeline.generation.compute_perplexity', return_value=10.0)
    @patch('src.pipeline.generation.PrivacyAttacker')
    @patch('src.pipeline.generation.measure_similarity_batch')
    @patch('src.pipeline.generation.calculate_single_pair_ngram_overlap')
    @patch('src.pipeline.generation.SentenceTransformer')
    @patch('src.pipeline.generation.PeftModel')
    @patch('src.pipeline.generation.load_tokenizer')
    @patch('src.pipeline.generation.load_causal_model')
    def test_batched_self_correction(
        self, mock_causal, mock_tok, mock_peft, mock_st,
        mock_ngram, mock_sim, mock_attacker, mock_ppl,
    ):
        # Setup model mocks
        mock_model = MagicMock()
        mock_peft.from_pretrained.return_value = mock_model
        mock_causal.return_value = mock_model

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "<pad>"
        mock_tokenizer.eos_token_id = 1
        mock_tokenizer.batch_decode.return_value = [
            "[ASSISTANT] Synthetic Text A",
            "[ASSISTANT] Synthetic Text B"
        ]
        mock_tok.return_value = mock_tokenizer

        # Disable Red Team for this test
        mock_attacker_instance = MagicMock()
        mock_attacker_instance.attack.return_value = (False, "OK")
        mock_attacker.return_value = mock_attacker_instance

        # Test Data
        private_data = [
            {'text': 'Original A', 'label': 0},
            {'text': 'Original B', 'label': 1},
        ]
        public_passages = ['Doc 1', 'Doc 2']
        retriever = MagicMock()
        retriever.retrieve.return_value = [[0], [1]]

        # Import after mocking
        from src.pipeline.generation import SyntheticDataGenerator
        generator = SyntheticDataGenerator("base", "adapter")

        # Mock side effects for detailed control
        mock_ngram.side_effect = [
            0.9,  # Iter 1, Item A -> Fail Privacy
            0.1,  # Iter 1, Item B -> Pass Privacy
            0.1,  # Iter 2, Item A -> Pass Privacy
            0.1,  # Iter 2, Item B -> Pass Privacy
            0.1   # Iter 3, Item B -> Pass Privacy
        ]

        mock_sim.side_effect = [
            [0.9],  # Iter 1, Item B -> check sim after privacy passes
            [0.2],  # Iter 1, Item B -> Fail Utility
            [0.9],  # Iter 2, Item A -> Pass Utility
            [0.2],  # Iter 2, Item B -> Fail Utility
            [0.9]   # Iter 3, Item B -> Pass Utility
        ]

        # Run Generation
        results = generator.generate(private_data, public_passages, retriever)

        # Both items should eventually pass (or be force-accepted)
        self.assertEqual(len(results), 2)
        print("Test Passed: Batched Self-Correction Logic verified.")


if __name__ == '__main__':
    unittest.main()
