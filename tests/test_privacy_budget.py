"""
Tests for the Privacy Budget Manager (Rényi DP Accounting).

Covers:
- Initialization and basic properties
- Budget consumption and tracking
- Budget exhaustion behavior
- Calibrated noise generation
- Report generation
- Edge cases
"""

import unittest
import numpy as np
from src.privacy_budget import (
    RenyiDPAccountant,
    PrivacyBudgetExhaustedError,
    calibrate_noise_scale,
    add_calibrated_noise,
)


class TestRenyiDPAccountant(unittest.TestCase):
    """Tests for the RenyiDPAccountant class."""

    def setUp(self):
        self.accountant = RenyiDPAccountant(
            total_budget=5.0, delta=1e-5, alpha=5.0
        )

    def test_initial_state(self):
        """Verify accountant starts with zero consumption."""
        self.assertEqual(self.accountant.rdp_consumed, 0.0)
        self.assertTrue(self.accountant.can_query())
        self.assertEqual(len(self.accountant.ledger), 0)

    def test_consume_increases_budget(self):
        """Verify consuming budget increases rdp_consumed."""
        initial_rdp = self.accountant.rdp_consumed
        self.accountant.consume(1.0, "test_query")  # Larger ε for clear RDP signal
        self.assertGreater(self.accountant.rdp_consumed, initial_rdp)

    def test_consume_records_ledger(self):
        """Verify ledger records each consumption event."""
        self.accountant.consume(0.1, "query_1")
        self.accountant.consume(0.2, "query_2")
        self.assertEqual(len(self.accountant.ledger), 2)
        self.assertEqual(self.accountant.ledger[0].step_name, "query_1")
        self.assertEqual(self.accountant.ledger[1].step_name, "query_2")

    def test_budget_remaining_decreases(self):
        """Verify rdp_consumed increases with each consume call."""
        initial_rdp = self.accountant.rdp_consumed
        self.accountant.consume(0.5, "test")  # Use larger epsilon for clearer signal
        self.assertGreater(self.accountant.rdp_consumed, initial_rdp)

    def test_budget_exhaustion_raises_error(self):
        """Verify PrivacyBudgetExhaustedError when budget is exceeded."""
        small_budget = RenyiDPAccountant(total_budget=0.5, delta=1e-5, alpha=5.0)
        with self.assertRaises(PrivacyBudgetExhaustedError):
            for i in range(1000):
                small_budget.consume(0.3, f"query_{i}")

    def test_can_query_with_projection(self):
        """Verify can_query projects future consumption correctly."""
        large_step = RenyiDPAccountant(total_budget=0.1, delta=1e-5, alpha=5.0)
        # With a tiny budget, a large step should be rejected
        self.assertFalse(large_step.can_query(epsilon_step=100.0))

    def test_report_generation(self):
        """Verify get_report returns correct structure."""
        self.accountant.consume(0.1, "test")
        report = self.accountant.get_report()
        
        self.assertIn("total_budget", report)
        self.assertIn("epsilon_spent", report)
        self.assertIn("budget_remaining", report)
        self.assertIn("num_queries", report)
        self.assertIn("is_exhausted", report)
        self.assertIn("ledger_summary", report)
        self.assertEqual(report["total_budget"], 5.0)
        self.assertEqual(report["num_queries"], 1)

    def test_reset(self):
        """Verify reset clears all state."""
        self.accountant.consume(0.1, "test")
        self.accountant.reset()
        self.assertEqual(self.accountant.rdp_consumed, 0.0)
        self.assertEqual(len(self.accountant.ledger), 0)


class TestCalibratedNoise(unittest.TestCase):
    """Tests for noise calibration utilities."""

    def test_calibrate_noise_scale(self):
        """Verify noise scale = sensitivity / epsilon."""
        scale = calibrate_noise_scale(epsilon=0.1, sensitivity=2.0)
        self.assertAlmostEqual(scale, 20.0)

    def test_calibrate_noise_scale_small_epsilon(self):
        """Smaller epsilon → larger noise scale (more privacy)."""
        scale_small = calibrate_noise_scale(0.01)
        scale_large = calibrate_noise_scale(1.0)
        self.assertGreater(scale_small, scale_large)

    def test_calibrate_noise_scale_invalid(self):
        """Zero or negative epsilon should raise ValueError."""
        with self.assertRaises(ValueError):
            calibrate_noise_scale(0.0)
        with self.assertRaises(ValueError):
            calibrate_noise_scale(-1.0)

    def test_add_calibrated_noise_shape(self):
        """Output shape should match input shape."""
        emb = np.random.randn(4, 384).astype(np.float32)
        noisy = add_calibrated_noise(emb, epsilon=0.1)
        self.assertEqual(noisy.shape, emb.shape)

    def test_add_calibrated_noise_normalization(self):
        """Output should be approximately unit-normalized."""
        emb = np.random.randn(8, 384).astype(np.float32)
        noisy = add_calibrated_noise(emb, epsilon=0.5)
        norms = np.linalg.norm(noisy, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=0.01)

    def test_add_calibrated_noise_differs(self):
        """Noisy output should differ from original."""
        emb = np.ones((2, 10), dtype=np.float32)
        noisy = add_calibrated_noise(emb, epsilon=0.5)
        self.assertFalse(np.allclose(emb, noisy))


if __name__ == "__main__":
    unittest.main()
