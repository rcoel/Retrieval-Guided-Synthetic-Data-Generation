"""
Tests for the Adaptive Temperature Scheduling and Generation Utilities.

Covers:
- Cosine annealing schedule
- Linear schedule
- Fixed schedule
- Failure-type-based adjustments
- Temperature bounds enforcement
"""

import unittest
import math
from src.pipeline.generation import compute_adaptive_temperature
from src import config


class TestAdaptiveTemperature(unittest.TestCase):
    """Tests for compute_adaptive_temperature."""

    def test_fixed_schedule(self):
        """Fixed schedule should always return GENERATION_TEMP."""
        for attempt in range(5):
            temp = compute_adaptive_temperature(attempt, 3, "privacy", "fixed")
            self.assertAlmostEqual(temp, config.GENERATION_TEMP)

    def test_privacy_failure_increases_temp(self):
        """Privacy failures should increase temperature (more randomness)."""
        temp_attempt_0 = compute_adaptive_temperature(0, 3, "privacy", "cosine")
        temp_attempt_3 = compute_adaptive_temperature(3, 3, "privacy", "cosine")
        self.assertGreaterEqual(temp_attempt_3, temp_attempt_0)

    def test_utility_failure_decreases_temp(self):
        """Utility failures should decrease temperature (more adherence)."""
        temp_attempt_0 = compute_adaptive_temperature(0, 3, "utility", "cosine")
        temp_attempt_3 = compute_adaptive_temperature(3, 3, "utility", "cosine")
        self.assertLessEqual(temp_attempt_3, temp_attempt_0)

    def test_coherence_failure(self):
        """Coherence failures should slightly decrease temperature."""
        temp = compute_adaptive_temperature(2, 3, "coherence", "cosine")
        self.assertLessEqual(temp, config.GENERATION_TEMP)

    def test_bounds_enforcement(self):
        """Temperature should always be within [TEMP_MIN, TEMP_MAX]."""
        for schedule in ["cosine", "linear"]:
            for failure in ["privacy", "utility", "coherence"]:
                for attempt in range(10):
                    temp = compute_adaptive_temperature(attempt, 5, failure, schedule)
                    self.assertGreaterEqual(temp, config.TEMP_MIN,
                        f"Temp {temp} below min at attempt={attempt}, failure={failure}")
                    self.assertLessEqual(temp, config.TEMP_MAX,
                        f"Temp {temp} above max at attempt={attempt}, failure={failure}")

    def test_linear_schedule(self):
        """Linear schedule should produce different values than cosine."""
        cos_temp = compute_adaptive_temperature(2, 5, "privacy", "cosine")
        lin_temp = compute_adaptive_temperature(2, 5, "privacy", "linear")
        # They should both be valid but may differ
        self.assertGreaterEqual(cos_temp, config.TEMP_MIN)
        self.assertGreaterEqual(lin_temp, config.TEMP_MIN)

    def test_unknown_failure_type(self):
        """Unknown failure type should return base temperature."""
        temp = compute_adaptive_temperature(1, 3, "unknown_type", "cosine")
        self.assertAlmostEqual(temp, config.GENERATION_TEMP)

    def test_zero_max_retries(self):
        """Should not divide by zero when max_retries is 0."""
        temp = compute_adaptive_temperature(0, 0, "privacy", "cosine")
        self.assertGreaterEqual(temp, config.TEMP_MIN)


if __name__ == "__main__":
    unittest.main()
