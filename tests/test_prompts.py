"""
Tests for the Prompt Templates (prompts.py).

Covers:
- Basic prompt structure
- Feedback injection
- CoT Critic prompt structure
- Red Team prompt structure
- Edge cases
"""

import unittest
from src.pipeline.prompts import (
    create_prompt,
    create_critic_prompt,
    create_red_team_prompt,
)


class TestCreatePrompt(unittest.TestCase):
    """Tests for the generation prompt template."""

    def test_basic_structure(self):
        """Prompt should contain all required sections."""
        prompt = create_prompt("Hello world", ["doc1", "doc2"])
        self.assertIn("[SYSTEM]", prompt)
        self.assertIn("[USER]", prompt)
        self.assertIn("[ASSISTANT]", prompt)
        self.assertIn("Hello world", prompt)
        self.assertIn("doc1", prompt)
        self.assertIn("doc2", prompt)

    def test_context_numbering(self):
        """Context docs should be numbered."""
        prompt = create_prompt("text", ["alpha", "beta", "gamma"])
        self.assertIn("1. alpha", prompt)
        self.assertIn("2. beta", prompt)
        self.assertIn("3. gamma", prompt)

    def test_feedback_injection(self):
        """Feedback should be injected into the prompt."""
        prompt = create_prompt("text", ["ctx"], feedback="Remove names")
        self.assertIn("CRITIQUE FROM PREVIOUS ATTEMPT", prompt)
        self.assertIn("Remove names", prompt)
        self.assertIn("FIX INSTRUCTION", prompt)

    def test_no_feedback(self):
        """Without feedback, critique section should not appear."""
        prompt = create_prompt("text", ["ctx"])
        self.assertNotIn("CRITIQUE FROM PREVIOUS ATTEMPT", prompt)

    def test_empty_context(self):
        """Should work with empty context list."""
        prompt = create_prompt("text", [])
        self.assertIn("[ASSISTANT]", prompt)


class TestCriticPrompt(unittest.TestCase):
    """Tests for the CoT Critic prompt template."""

    def test_structure(self):
        """Critic prompt should contain structured analysis instructions."""
        prompt = create_critic_prompt("original", "generated", "High Overlap")
        self.assertIn("Data Privacy and Quality Assurance Critic", prompt)
        self.assertIn("step-by-step", prompt)

    def test_json_format(self):
        """Critic prompt should request JSON format output."""
        prompt = create_critic_prompt("a", "b", "test")
        self.assertIn("problematic_spans", prompt)
        self.assertIn("fix_instruction", prompt)
        self.assertIn("severity", prompt)

    def test_includes_texts(self):
        """Prompt should include both original and generated texts."""
        prompt = create_critic_prompt("alpha original", "beta generated", "issue")
        self.assertIn("alpha original", prompt)
        self.assertIn("beta generated", prompt)

    def test_includes_issue_type(self):
        """Issue type should appear in the prompt."""
        prompt = create_critic_prompt("a", "b", "Privacy Violation")
        self.assertIn("Privacy Violation", prompt)


class TestRedTeamPrompt(unittest.TestCase):
    """Tests for the Red Team audit prompt template."""

    def test_structure(self):
        """Red Team prompt should set up a privacy audit scenario."""
        prompt = create_red_team_prompt("some synthetic text")
        self.assertIn("privacy auditor", prompt)
        self.assertIn("some synthetic text", prompt)
        self.assertIn("YES", prompt)
        self.assertIn("NO", prompt)

    def test_assistant_tag(self):
        """Prompt should end with assistant tag for generation."""
        prompt = create_red_team_prompt("text")
        self.assertIn("[ASSISTANT]", prompt)


if __name__ == "__main__":
    unittest.main()
