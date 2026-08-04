from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from analyze_round2_length import (  # noqa: E402
    development_quantile,
    ladder_bootstrap_association,
)


class DevelopmentQuantileTests(unittest.TestCase):
    def test_quantile_scale_is_monotone_and_bounded(self) -> None:
        development = np.asarray([10.0, 20.0, 30.0, 40.0])
        values = np.asarray([5.0, 15.0, 25.0, 35.0, 45.0])
        quantiles = development_quantile(development, values)
        self.assertTrue(np.all(np.diff(quantiles) > 0))
        self.assertGreaterEqual(quantiles.min(), 0.0)
        self.assertLessEqual(quantiles.max(), 1.0)

    def test_ties_use_the_midpoint_convention(self) -> None:
        development = np.asarray([1.0, 2.0, 2.0, 3.0])
        quantiles = development_quantile(development, np.asarray([2.0]))
        self.assertAlmostEqual(float(quantiles[0]), 0.5)

    def test_held_out_values_do_not_define_the_scale(self) -> None:
        """Confirmation lengths must not shift the transform."""
        development = np.asarray([10.0, 20.0, 30.0])
        first = development_quantile(development, np.asarray([25.0]))
        second = development_quantile(
            development,
            np.asarray([25.0, 1000.0, -50.0]),
        )
        self.assertAlmostEqual(float(first[0]), float(second[0]))


class LadderBootstrapTests(unittest.TestCase):
    def test_observed_association_matches_full_sample(self) -> None:
        rng = np.random.default_rng(3)
        first = np.arange(20, dtype=float)
        second = first + rng.normal(scale=0.5, size=20)
        ladders = np.repeat([f"l{i}" for i in range(10)], 2)
        result = ladder_bootstrap_association(first, second, ladders, 200, rng)
        expected = np.corrcoef(first, second)[0, 1]
        self.assertAlmostEqual(result["pearson"], expected, places=10)
        self.assertEqual(result["independent_ladders"], 10)

    def test_interval_brackets_a_strong_association(self) -> None:
        rng = np.random.default_rng(5)
        first = np.arange(40, dtype=float)
        second = first * 2.0
        ladders = np.repeat([f"l{i}" for i in range(10)], 4)
        result = ladder_bootstrap_association(first, second, ladders, 500, rng)
        low, high = result["spearman_95_percent_interval"]
        self.assertLessEqual(low, result["spearman"])
        self.assertLessEqual(result["spearman"], high)
        self.assertGreater(low, 0.9)

    def test_degenerate_resamples_are_dropped_not_scored_as_zero(self) -> None:
        """A constant resample carries no rank information.

        Scoring it as zero would pull the small fixed-engagement subgroup
        intervals toward the null.
        """
        rng = np.random.default_rng(7)
        first = np.asarray([1.0, 2.0, 3.0])
        second = np.asarray([2.0, 4.0, 6.0])
        ladders = np.asarray(["a", "b", "c"])
        result = ladder_bootstrap_association(first, second, ladders, 400, rng)
        self.assertGreater(result["degenerate_bootstrap_samples"], 0)
        self.assertEqual(
            result["usable_bootstrap_samples"]
            + result["degenerate_bootstrap_samples"],
            400,
        )
        self.assertGreater(result["spearman_95_percent_interval"][0], 0.9)

    def test_no_usable_samples_yields_null_intervals(self) -> None:
        rng = np.random.default_rng(9)
        first = np.asarray([1.0, 2.0])
        second = np.asarray([5.0, 7.0])
        ladders = np.asarray(["a", "b"])
        result = ladder_bootstrap_association(first, second, ladders, 1, rng)
        self.assertIn("spearman_95_percent_interval", result)


if __name__ == "__main__":
    unittest.main()
