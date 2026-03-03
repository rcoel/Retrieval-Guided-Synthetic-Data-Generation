"""
Tests for the Data Loader module.

Covers:
- Proper exception handling (DataLoadError instead of exit())
- Data validation
- Column standardization
"""

import unittest
import tempfile
import os
import pandas as pd
from src.dataloader import DataLoadError, load_public_corpus


class TestLoadPublicCorpus(unittest.TestCase):
    """Tests for load_public_corpus."""

    def test_valid_csv(self):
        """Should load a valid CSV with 'text' column."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("text\nhello world\nfoo bar\n")
            f.flush()
            try:
                df = load_public_corpus(f.name)
                self.assertEqual(len(df), 2)
                self.assertIn("text", df.columns)
            finally:
                os.unlink(f.name)

    def test_missing_file(self):
        """Should raise DataLoadError when file is missing."""
        with self.assertRaises(DataLoadError) as ctx:
            load_public_corpus("/nonexistent/path.csv")
        self.assertIn("not found", str(ctx.exception).lower())

    def test_missing_text_column(self):
        """Should raise DataLoadError when 'text' column is missing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("wrong_column\nhello\n")
            f.flush()
            try:
                with self.assertRaises(DataLoadError) as ctx:
                    load_public_corpus(f.name)
                self.assertIn("text", str(ctx.exception).lower())
            finally:
                os.unlink(f.name)

    def test_drops_nan_rows(self):
        """Should drop rows with NaN text values."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("text\nhello\n\nworld\n")
            f.flush()
            try:
                df = load_public_corpus(f.name)
                self.assertLessEqual(len(df), 2)
            finally:
                os.unlink(f.name)

    def test_drops_empty_string_rows(self):
        """Should drop rows with empty string text."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write('text\nhello\n""\nworld\n')
            f.flush()
            try:
                df = load_public_corpus(f.name)
                for _, row in df.iterrows():
                    self.assertTrue(len(row['text'].strip()) > 0)
            finally:
                os.unlink(f.name)


if __name__ == "__main__":
    unittest.main()
