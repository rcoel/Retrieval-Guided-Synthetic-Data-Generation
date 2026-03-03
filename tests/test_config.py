"""
Tests for the Pipeline Configuration (PipelineConfig dataclass).

Covers:
- Default values
- Validation constraints
- Computed properties
- Serialization
- Backward compatibility
"""

import unittest
from src.config import PipelineConfig
from src import config


class TestPipelineConfig(unittest.TestCase):
    """Tests for PipelineConfig dataclass."""

    def test_default_values(self):
        """Verify sensible defaults are set."""
        cfg = PipelineConfig()
        self.assertEqual(cfg.GENERATION_TEMP, 0.8)
        self.assertEqual(cfg.SEED, 42)
        self.assertTrue(cfg.ENABLE_DP_ACCOUNTING)
        self.assertTrue(cfg.ENABLE_RED_TEAM)
        self.assertEqual(cfg.TEMP_SCHEDULE, "cosine")

    def test_validation_chunk_overlap(self):
        """CHUNK_OVERLAP must be less than CHUNK_SIZE."""
        with self.assertRaises(AssertionError):
            PipelineConfig(CHUNK_SIZE=100, CHUNK_OVERLAP=200)

    def test_validation_epsilon_positive(self):
        """PRIVACY_EPSILON must be positive."""
        with self.assertRaises(AssertionError):
            PipelineConfig(PRIVACY_EPSILON=-0.1)

    def test_validation_temp_range(self):
        """GENERATION_TEMP must be within [TEMP_MIN, TEMP_MAX]."""
        with self.assertRaises(AssertionError):
            PipelineConfig(GENERATION_TEMP=3.0)

    def test_validation_schedule(self):
        """TEMP_SCHEDULE must be one of cosine/linear/fixed."""
        with self.assertRaises(AssertionError):
            PipelineConfig(TEMP_SCHEDULE="exponential")

    def test_validation_rdp_alpha(self):
        """RDP_ALPHA must be greater than 1."""
        with self.assertRaises(AssertionError):
            PipelineConfig(RDP_ALPHA=0.5)

    def test_validation_ngram_overlap(self):
        """MAX_NGRAM_OVERLAP must be in (0, 1]."""
        with self.assertRaises(AssertionError):
            PipelineConfig(MAX_NGRAM_OVERLAP=1.5)

    def test_computed_properties(self):
        """Verify computed properties derive correctly."""
        cfg = PipelineConfig(OUTPUT_DIR="test_output/")
        self.assertIn("test_output", cfg.FAISS_INDEX_PATH)
        self.assertIn("test_output", cfg.SYNTHETIC_DATA_PATH)
        self.assertIn("test_output", cfg.MODEL_OUTPUT_DIR)
        self.assertIn("test_output", cfg.ADAPTER_PATH)
        self.assertIn("test_output", cfg.METRICS_LOG_PATH)

    def test_to_dict(self):
        """Verify serialization includes all fields and properties."""
        cfg = PipelineConfig()
        d = cfg.to_dict()
        self.assertIn("SEED", d)
        self.assertIn("FAISS_INDEX_PATH", d)  # Computed property
        self.assertIn("LORA_TARGET_MODULES", d)
        self.assertIsInstance(d["LORA_TARGET_MODULES"], list)

    def test_backward_compatibility(self):
        """Module-level globals must match dataclass defaults."""
        self.assertEqual(config.GENERATION_TEMP, 0.8)
        self.assertEqual(config.SEED, 42)
        self.assertEqual(config.PRIVACY_EPSILON, 0.1)
        self.assertTrue(config.ENABLE_DP_ACCOUNTING)
        self.assertEqual(config.TOTAL_PRIVACY_BUDGET, 10.0)
        self.assertEqual(config.RDP_ALPHA, 5.0)


if __name__ == "__main__":
    unittest.main()
