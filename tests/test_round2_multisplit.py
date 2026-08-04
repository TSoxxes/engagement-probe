from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from analyze_round2_multisplit import (  # noqa: E402
    CAPABILITY,
    HARMFUL,
    distribution_summary,
    evaluate_split,
    generate_balanced_evaluation_sets,
)


def metadata_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ladder_id": [f"cap_{i:02d}" for i in range(20)]
            + [f"harm_{i:02d}" for i in range(20)],
            "d_condition": [CAPABILITY] * 20 + [HARMFUL] * 20,
            "original_split": ["development"] * 40,
        }
    )


class Round2MultisplitTests(unittest.TestCase):
    def test_balanced_evaluation_sets_are_unique_and_condition_balanced(
        self,
    ) -> None:
        metadata = metadata_frame()
        splits = generate_balanced_evaluation_sets(
            metadata,
            repeats=25,
            rng=np.random.default_rng(7),
        )
        condition = dict(zip(metadata["ladder_id"], metadata["d_condition"]))
        self.assertEqual(len(splits), 25)
        self.assertEqual(len(set(splits)), 25)
        for split in splits:
            self.assertEqual(len(split), 10)
            self.assertEqual(
                sum(condition[value] == CAPABILITY for value in split),
                5,
            )
            self.assertEqual(
                sum(condition[value] == HARMFUL for value in split),
                5,
            )

    def test_distribution_summary_uses_requested_quantiles(self) -> None:
        values = np.arange(1, 101, dtype=float)
        summary = distribution_summary(values)
        self.assertEqual(summary["mean"], 50.5)
        self.assertEqual(summary["median"], 50.5)
        self.assertEqual(summary["minimum"], 1.0)
        self.assertEqual(summary["maximum"], 100.0)
        self.assertEqual(summary["percentile_25"], np.quantile(values, 0.25))
        self.assertEqual(
            summary["percentile_97_5"],
            np.quantile(values, 0.975),
        )

    def test_evaluate_split_returns_held_out_predictions(self) -> None:
        rng = np.random.default_rng(11)
        rows = []
        ladder_ids = [f"cap_{i:02d}" for i in range(20)] + [
            f"harm_{i:02d}" for i in range(20)
        ]
        for ladder_index, ladder_id in enumerate(ladder_ids):
            d_condition = CAPABILITY if ladder_id.startswith("cap_") else HARMFUL
            for prompt_index, design_condition in enumerate(
                ["A", "B", "C", d_condition]
            ):
                rows.append(
                    {
                        "prompt_id": f"{ladder_id}_{prompt_index}",
                        "ladder_id": ladder_id,
                        "split": "development",
                        "design_condition": design_condition,
                        "prompt": (
                            f"shared topic {ladder_index} "
                            f"condition {design_condition}"
                        ),
                        "substantive_engagement": float(3 - prompt_index),
                    }
                )
        scores = pd.DataFrame(rows)
        activations = rng.normal(size=(160, 3, 8)).astype(np.float32)
        activations[:, 1, 0] = scores["substantive_engagement"].to_numpy()
        evaluation_ladders = tuple(
            [f"cap_{i:02d}" for i in range(5)]
            + [f"harm_{i:02d}" for i in range(5)]
        )

        metrics, predictions = evaluate_split(
            scores,
            activations,
            evaluation_ladders,
            activation_alpha=10.0,
            text_alpha=1.0,
        )

        self.assertEqual(len(predictions), 40)
        self.assertEqual(set(predictions["ladder_id"]), set(evaluation_ladders))
        self.assertEqual(metrics["selected_layer"], 1)
        self.assertGreater(metrics["activation_spearman"], 0.95)


if __name__ == "__main__":
    unittest.main()
