"""Post-hoc reliability analysis for the Round 2 variability subset.

The protocol collected three additional generations for a prespecified
32-prompt subset in order to estimate how much the model varies when given the
same prompt. That estimate was never computed. This script computes it and
reports the ceiling it implies on any predictor's agreement with the
prompt-level target.

Not part of the prespecified analysis. Descriptive only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

KEY = ["prompt_id", "generation_id"]
TARGET = "substantive_engagement"


def combine_judges(judge_files: list[Path], manual_scores: Path | None) -> pd.DataFrame:
    """Replicates src/analyze_round2.combine_judges for the target column."""
    frames = []
    for judge, path in enumerate(judge_files, start=1):
        frame = pd.read_json(path, lines=True)[KEY + [TARGET]].copy()
        frame["judge"] = judge
        frames.append(frame)
    long = pd.concat(frames, ignore_index=True)
    means = long.groupby(KEY, sort=False)[[TARGET]].mean().reset_index()
    if manual_scores is not None:
        manual = pd.read_json(manual_scores, lines=True)
        indexed = means.set_index(KEY)
        for _, row in manual.iterrows():
            key = (row["prompt_id"], int(row["generation_id"]))
            if TARGET in manual.columns and pd.notna(row.get(TARGET)):
                indexed.loc[key, TARGET] = float(row[TARGET])
        means = indexed.reset_index()
    return means, long


def variance_components(frame: pd.DataFrame) -> dict:
    """One-way random-effects decomposition, unbalanced-safe."""
    groups = frame.groupby("prompt_id")[TARGET]
    counts = groups.size().to_numpy(dtype=float)
    group_means = groups.mean().to_numpy()
    k, total = len(counts), counts.sum()
    grand = float((counts * group_means).sum() / total)

    ss_between = float((counts * (group_means - grand) ** 2).sum())
    ss_within = float(((frame[TARGET] - frame["prompt_id"].map(groups.mean())) ** 2).sum())
    ms_between = ss_between / (k - 1)
    ms_within = ss_within / (total - k)

    # Effective group size (equals n for a balanced design).
    n0 = (total - (counts**2).sum() / total) / (k - 1)
    var_within = ms_within
    var_between = max(0.0, (ms_between - ms_within) / n0)

    total_var = var_between + var_within
    return {
        "prompts": int(k),
        "observations": int(total),
        "generations_per_prompt": (
            int(counts[0]) if len(set(counts)) == 1 else f"{counts.min():.0f}-{counts.max():.0f}"
        ),
        "var_between": var_between,
        "var_within": var_within,
        "between_sd": float(np.sqrt(var_between)),
        "within_sd": float(np.sqrt(var_within)),
        # Undefined when a condition is fully saturated: if every response
        # scores identically there is no variance of either kind to apportion.
        "within_share": float(var_within / total_var) if total_var > 0 else None,
        "icc_single_generation": float(var_between / total_var) if total_var > 0 else None,
        "saturated": bool(total_var == 0),
    }


def reliability(components: dict, m: int) -> float:
    """Reliability of a mean of m generations."""
    vb, vw = components["var_between"], components["var_within"]
    return vb / (vb + vw / m)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path,
                        default=Path("results/round2_main/run_round2"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/round2_main/run_round2/analysis/reliability_summary.json"))
    args = parser.parse_args()

    agg = args.run_dir / "judging_web" / "aggregated"
    scores, judge_long = combine_judges(
        [agg / f"judge_{i}.jsonl" for i in (1, 2, 3)],
        agg / "manual_scores.jsonl",
    )
    prompts = pd.read_csv(args.run_dir / "analysis" / "prompt_level_scores.csv")
    scores = scores.merge(
        prompts[["prompt_id", "variability_subset", "design_condition", "split"]],
        on="prompt_id", how="left", validate="many_to_one",
    )

    subset = scores[scores["variability_subset"]].copy()
    full = scores.copy()

    result = {
        "note": "Post-hoc descriptive analysis. Not part of the prespecified plan.",
        "target": TARGET,
        "variability_subset": variance_components(subset),
        "all_prompts": variance_components(full),
    }
    for name, comp in (("variability_subset", result["variability_subset"]),
                       ("all_prompts", result["all_prompts"])):
        comp["reliability_of_mean"] = {
            f"{m}_generations": reliability(comp, m) for m in (1, 3, 6)
        }
        comp["max_attainable_correlation"] = {
            f"{m}_generations": float(np.sqrt(reliability(comp, m))) for m in (1, 3, 6)
        }

    # Reliability of the condition-residual target used in the exploratory
    # follow-up: engagement after removing design-condition means. This is the
    # target section 08 of the report analyses.
    for name, frame in (("variability_subset", subset), ("all_prompts", full)):
        residual = frame.copy()
        residual[TARGET] = residual[TARGET] - residual.groupby("design_condition")[TARGET].transform("mean")
        comp = variance_components(residual)
        comp["reliability_of_mean"] = {
            f"{m}_generations": reliability(comp, m) for m in (1, 3, 6)
        }
        comp["max_attainable_correlation"] = {
            f"{m}_generations": float(np.sqrt(reliability(comp, m))) for m in (1, 3, 6)
        }
        result[f"condition_residual_{name}"] = comp

    # Per-condition estimates use all prompts, not the variability subset.
    # The subset holds only 3-5 prompts in each D condition, too few for a
    # stable between-prompt variance: it returned an ICC of 0.00 for harmful-
    # request prompts, an artifact of the estimator hitting its floor, where
    # the full 20 prompts give 0.94.
    result["by_condition_all_prompts"] = {
        str(cond): variance_components(part)
        for cond, part in full.groupby("design_condition")
    }

    # How much of the response-level score is judge disagreement, pre-adjudication.
    spread = judge_long.groupby(KEY)[TARGET].agg(["std", "max", "min"])
    result["judge_disagreement_pre_adjudication"] = {
        "mean_sd_across_three_judges": float(spread["std"].mean()),
        "share_of_responses_with_unanimous_score": float((spread["max"] == spread["min"]).mean()),
        "share_with_range_above_one_point": float(((spread["max"] - spread["min"]) > 1).mean()),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
