"""
Tests for utility functions.

Covers:
- Device selection
- Seed management
- Data I/O (save/load synthetic data)
"""

import unittest
import tempfile
import os
import json
from src.utils import get_device, set_seed, save_synthetic_data, load_synthetic_data


class TestGetDevice(unittest.TestCase):
    """Tests for get_device."""

    def test_returns_valid_device(self):
        """Should return 'cuda', 'mps', or 'cpu'."""
        device = get_device()
        self.assertIn(device, ("cuda", "mps", "cpu"))


class TestSetSeed(unittest.TestCase):
    """Tests for set_seed."""

    def test_reproducibility(self):
        """Same seed should produce same random numbers."""
        import numpy as np
        set_seed(123)
        a = np.random.rand(5)
        set_seed(123)
        b = np.random.rand(5)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds(self):
        """Different seeds should produce different random numbers."""
        import numpy as np
        set_seed(1)
        a = np.random.rand(5)
        set_seed(2)
        b = np.random.rand(5)
        self.assertFalse((a == b).all())


class TestSyntheticDataIO(unittest.TestCase):
    """Tests for save/load synthetic data."""

    def test_roundtrip(self):
        """Save and load should preserve data."""
        data = [
            {"text": "hello", "label": 0},
            {"text": "world", "label": 1},
        ]
        with tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False) as f:
            path = f.name
        try:
            save_synthetic_data(data, path)
            loaded = load_synthetic_data(path)
            self.assertEqual(len(loaded), 2)
            self.assertEqual(loaded[0]["text"], "hello")
            self.assertEqual(loaded[1]["label"], 1)
        finally:
            os.unlink(path)

    def test_empty_data(self):
        """Should handle empty data list."""
        with tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False) as f:
            path = f.name
        try:
            save_synthetic_data([], path)
            loaded = load_synthetic_data(path)
            self.assertEqual(len(loaded), 0)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
