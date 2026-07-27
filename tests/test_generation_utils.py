"""Regression tests for model stopping and finish-reason diagnostics."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from generation_utils import (
    infer_finish_reason,
    resolve_eos_token_id,
    resolve_generation_counts,
)


class GenerationUtilsTest(unittest.TestCase):
    def test_resolve_generation_counts(self) -> None:
        prompts = [
            {"prompt_id": "a", "generation_count": 3},
            {"prompt_id": "b", "generation_count": 6},
        ]
        self.assertEqual(
            resolve_generation_counts(prompts, 3, "generation_count"),
            [3, 6],
        )
        self.assertEqual(resolve_generation_counts(prompts, 4, None), [4, 4])
        with self.assertRaises(ValueError):
            resolve_generation_counts(
                [{"prompt_id": "bad", "generation_count": 0}],
                3,
                "generation_count",
            )

    def test_prefers_full_model_generation_stop_list(self) -> None:
        model = SimpleNamespace(
            generation_config=SimpleNamespace(eos_token_id=[1, 107]),
            config=SimpleNamespace(eos_token_id=1),
        )
        tokenizer = SimpleNamespace(eos_token_id=1)

        self.assertEqual(resolve_eos_token_id(model, tokenizer), [1, 107])

    def test_falls_back_to_model_then_tokenizer(self) -> None:
        model = SimpleNamespace(
            generation_config=SimpleNamespace(eos_token_id=None),
            config=SimpleNamespace(eos_token_id=[1, 107]),
        )
        tokenizer = SimpleNamespace(eos_token_id=1)
        self.assertEqual(resolve_eos_token_id(model, tokenizer), [1, 107])

        model.config.eos_token_id = None
        self.assertEqual(resolve_eos_token_id(model, tokenizer), 1)

    def test_finish_reason_distinguishes_eos_and_length(self) -> None:
        eos_ids = [1, 107]
        self.assertEqual(infer_finish_reason(125, 600, 107, eos_ids), "eos")
        self.assertEqual(infer_finish_reason(600, 600, 107, eos_ids), "eos")
        self.assertEqual(infer_finish_reason(600, 600, 42, eos_ids), "length")
        self.assertEqual(infer_finish_reason(125, 600, 42, eos_ids), "other")


if __name__ == "__main__":
    unittest.main()
