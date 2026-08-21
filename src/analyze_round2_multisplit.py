"""Post-hoc repeated-split sensitivity analysis for the Round 2 probe.

The procedure is frozen in docs/round2_repeated_split_sensitivity_spec.md.
This analysis is descriptive and does not replace the preregistered result.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from analyze import fit_predict_ridge, lolo_predictions, summarize
from text_features import ridge_predict as text_ridge_predict
from text_features import tfidf_features


TARGET = "substantive_engagement"
CAPABILITY = "D_capability"
HARMFUL = "D_harmful_request"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--activations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--splits", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--activation-alpha", type=float, default=10.0)
    parser.add_argument("--text-alpha", type=float, default=1.0)
    return parser.parse_args()


def ladder_metadata(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ladder_id, frame in scores.groupby("ladder_id", sort=True):
        d_rows = frame.loc[frame["design_condition"].astype(str).str.startswith("D_")]
        if len(frame) != 4 or len(d_rows) != 1:
            raise ValueError(
                f"{ladder_id} must contain four prompts and exactly one D prompt"
            )
        rows.append(
            {
                "ladder_id": str(ladder_id),
                "d_condition": str(d_rows["design_condition"].iloc[0]),
                "original_split": str(frame["split"].iloc[0]),
            }
        )
    metadata = pd.DataFrame(rows)
    if set(metadata["d_condition"]) != {CAPABILITY, HARMFUL}:
        raise ValueError("Unexpected D-prompt conditions")
    if metadata["ladder_id"].nunique() != 40:
        raise ValueError("Expected exactly 40 semantic ladders")
    return metadata


def generate_balanced_evaluation_sets(
    metadata: pd.DataFrame,
    repeats: int,
    rng: np.random.Generator,
) -> list[tuple[str, ...]]:
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    capability = metadata.loc[
        metadata["d_condition"].eq(CAPABILITY), "ladder_id"
    ].to_numpy(str)
    harmful = metadata.loc[
        metadata["d_condition"].eq(HARMFUL), "ladder_id"
    ].to_numpy(str)
    if len(capability) < 5 or len(harmful) < 5:
        raise ValueError("Need at least five ladders for each D condition")

    evaluation_sets: list[tuple[str, ...]] = []
    seen: set[tuple[str, ...]] = set()
    attempts = 0
    maximum_attempts = repeats * 100
    while len(evaluation_sets) < repeats and attempts < maximum_attempts:
        attempts += 1
        sampled = tuple(
            sorted(
                [*rng.choice(capability, 5, replace=False),
                 *rng.choice(harmful, 5, replace=False)]
            )
        )
        if sampled not in seen:
            seen.add(sampled)
            evaluation_sets.append(sampled)
    if len(evaluation_sets) != repeats:
        raise RuntimeError("Could not generate the requested unique splits")
    return evaluation_sets


def select_activation_layer(
    development_activations: np.ndarray,
    development_target: np.ndarray,
    development_ladders: np.ndarray,
    alpha: float,
) -> tuple[int, float]:
    rng = np.random.default_rng(0)
    spearman_by_layer = []
    for layer in range(development_activations.shape[1]):
        predictions = lolo_predictions(
            development_activations[:, layer, :],
            development_target,
            development_ladders,
            alpha,
            False,
            rng,
        )
        spearman_by_layer.append(
            summarize(development_target, predictions)["spearman"]
        )
    values = np.asarray(spearman_by_layer, dtype=float)
    best = float(np.max(values))
    selected = int(np.flatnonzero(values == best)[0])
    return selected, best


def metric_columns(
    truth: np.ndarray,
    activation_prediction: np.ndarray,
    text_prediction: np.ndarray,
) -> dict[str, float]:
    activation = summarize(truth, activation_prediction)
    text = summarize(truth, text_prediction)
    return {
        "activation_spearman": activation["spearman"],
        "text_spearman": text["spearman"],
        "spearman_delta": activation["spearman"] - text["spearman"],
        "activation_pearson": activation["pearson"],
        "text_pearson": text["pearson"],
        "pearson_delta": activation["pearson"] - text["pearson"],
        "activation_mae": activation["mae"],
        "text_mae": text["mae"],
        "mae_improvement": text["mae"] - activation["mae"],
    }


def evaluate_split(
    scores: pd.DataFrame,
    activations: np.ndarray,
    evaluation_ladders: tuple[str, ...],
    activation_alpha: float,
    text_alpha: float,
) -> tuple[dict[str, float | int | str], pd.DataFrame]:
    ladders = scores["ladder_id"].astype(str).to_numpy()
    evaluation = np.isin(ladders, np.asarray(evaluation_ladders, dtype=str))
    development = ~evaluation
    if int(evaluation.sum()) != 40 or int(development.sum()) != 120:
        raise ValueError("Every split must contain 120 development and 40 evaluation prompts")

    target = scores[TARGET].to_numpy(float)
    selected_layer, development_spearman = select_activation_layer(
        activations[development],
        target[development],
        ladders[development],
        activation_alpha,
    )
    activation_prediction = fit_predict_ridge(
        activations[development, selected_layer, :],
        target[development],
        activations[evaluation, selected_layer, :],
        activation_alpha,
    )
    prompt_text = scores["prompt"].astype(str).to_numpy()
    text_train, text_evaluation = tfidf_features(
        prompt_text[development].tolist(),
        prompt_text[evaluation].tolist(),
    )
    text_prediction = text_ridge_predict(
        text_train,
        target[development],
        text_evaluation,
        text_alpha,
    )
    metrics: dict[str, float | int | str] = {
        "selected_layer": selected_layer,
        "development_selected_layer_spearman": development_spearman,
        "evaluation_ladders": json.dumps(list(evaluation_ladders)),
        **metric_columns(
            target[evaluation],
            activation_prediction,
            text_prediction,
        ),
    }
    predictions = pd.DataFrame(
        {
            "prompt_id": scores.loc[evaluation, "prompt_id"].astype(str).to_numpy(),
            "ladder_id": ladders[evaluation],
            "true_score": target[evaluation],
            "activation_prediction": activation_prediction,
            "text_prediction": text_prediction,
            "selected_layer": selected_layer,
        }
    )
    return metrics, predictions


def distribution_summary(values: np.ndarray) -> dict[str, float]:
    quantiles = np.quantile(values, [0.025, 0.25, 0.5, 0.75, 0.975])
    return {
        "mean": float(np.mean(values)),
        "standard_deviation": float(np.std(values, ddof=1)),
        "minimum": float(np.min(values)),
        "percentile_2_5": float(quantiles[0]),
        "percentile_25": float(quantiles[1]),
        "median": float(quantiles[2]),
        "percentile_75": float(quantiles[3]),
        "percentile_97_5": float(quantiles[4]),
        "maximum": float(np.max(values)),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scores = pd.read_csv(args.scores)
    required = {
        "prompt_id",
        "ladder_id",
        "split",
        "design_condition",
        "prompt",
        TARGET,
    }
    missing = required - set(scores.columns)
    if missing:
        raise ValueError(f"Prompt-level scores lack columns: {sorted(missing)}")
    if scores["prompt_id"].duplicated().any() or len(scores) != 160:
        raise ValueError("Expected 160 unique prompt-level rows")

    archive = np.load(args.activations)
    activation_ids = archive["prompt_ids"].astype(str)
    activation_index = {value: index for index, value in enumerate(activation_ids)}
    missing_activations = set(scores["prompt_id"].astype(str)) - set(activation_index)
    if missing_activations:
        raise ValueError(f"Missing activations for {len(missing_activations)} prompts")
    activations = archive["activations"][
        np.asarray([activation_index[value] for value in scores["prompt_id"].astype(str)])
    ].astype(np.float32)

    metadata = ladder_metadata(scores)
    rng = np.random.default_rng(args.seed)
    evaluation_sets = generate_balanced_evaluation_sets(metadata, args.splits, rng)

    original_evaluation = tuple(
        sorted(metadata.loc[
            metadata["original_split"].eq("confirmation"), "ladder_id"
        ].astype(str))
    )
    original_metrics, original_predictions = evaluate_split(
        scores,
        activations,
        original_evaluation,
        args.activation_alpha,
        args.text_alpha,
    )
    original_metrics["split_id"] = "original"
    original_predictions.insert(0, "split_id", "original")

    metric_rows = []
    prediction_frames = []
    for index, evaluation_ladders in enumerate(evaluation_sets, start=1):
        metrics, predictions = evaluate_split(
            scores,
            activations,
            evaluation_ladders,
            args.activation_alpha,
            args.text_alpha,
        )
        split_id = f"repeat_{index:03d}"
        metrics["split_id"] = split_id
        predictions.insert(0, "split_id", split_id)
        metric_rows.append(metrics)
        prediction_frames.append(predictions)
        if index == 1 or index % 10 == 0 or index == args.splits:
            print(f"Completed {index}/{args.splits} repeated splits", flush=True)

    split_metrics = pd.DataFrame(metric_rows)
    heldout_predictions = pd.concat(prediction_frames, ignore_index=True)
    crossfitted = (
        heldout_predictions.groupby(["prompt_id", "ladder_id"], as_index=False)
        .agg(
            true_score=("true_score", "first"),
            activation_prediction=("activation_prediction", "mean"),
            text_prediction=("text_prediction", "mean"),
            evaluation_count=("split_id", "size"),
        )
    )
    if len(crossfitted) != 160 or crossfitted["evaluation_count"].min() < 1:
        raise RuntimeError("Every prompt must receive at least one held-out prediction")
    crossfitted_metrics = metric_columns(
        crossfitted["true_score"].to_numpy(float),
        crossfitted["activation_prediction"].to_numpy(float),
        crossfitted["text_prediction"].to_numpy(float),
    )

    layer_counts = Counter(split_metrics["selected_layer"].astype(int))
    layer_frame = pd.DataFrame(
        [
            {
                "selected_layer": layer,
                "count": count,
                "fraction": count / args.splits,
            }
            for layer, count in sorted(layer_counts.items())
        ]
    )
    delta = split_metrics["spearman_delta"].to_numpy(float)
    pearson_delta = split_metrics["pearson_delta"].to_numpy(float)
    mae_improvement = split_metrics["mae_improvement"].to_numpy(float)
    original_delta = float(original_metrics["spearman_delta"])
    summary = {
        "analysis_status": "post_hoc_sensitivity",
        "specification": "docs/round2_repeated_split_sensitivity_spec.md",
        "seed": args.seed,
        "repeated_splits": args.splits,
        "development_ladders_per_split": 30,
        "evaluation_ladders_per_split": 10,
        "activation_alpha": args.activation_alpha,
        "text_alpha": args.text_alpha,
        "original_split": original_metrics,
        "repeated_split_spearman_delta": {
            **distribution_summary(delta),
            "fraction_above_zero": float(np.mean(delta > 0)),
            "fraction_at_or_below_original": float(np.mean(delta <= original_delta)),
        },
        "repeated_split_pearson_delta": {
            **distribution_summary(pearson_delta),
            "fraction_above_zero": float(np.mean(pearson_delta > 0)),
        },
        "repeated_split_mae_improvement": {
            **distribution_summary(mae_improvement),
            "fraction_above_zero": float(np.mean(mae_improvement > 0)),
        },
        "crossfitted_aggregate_metrics": crossfitted_metrics,
        "crossfitted_evaluation_count": {
            "minimum": int(crossfitted["evaluation_count"].min()),
            "mean": float(crossfitted["evaluation_count"].mean()),
            "maximum": int(crossfitted["evaluation_count"].max()),
        },
        "layer_selection_counts": {
            str(layer): count for layer, count in sorted(layer_counts.items())
        },
        "interpretation_constraint": (
            "Overlapping repeated splits are a descriptive stability diagnostic, "
            "not a confidence interval, p-value, or confirmatory test."
        ),
    }

    split_metrics.sort_values("split_id").to_csv(
        args.output_dir / "split_metrics.csv", index=False
    )
    heldout_predictions.to_csv(
        args.output_dir / "heldout_predictions.csv", index=False
    )
    crossfitted.to_csv(
        args.output_dir / "prompt_crossfitted_predictions.csv", index=False
    )
    layer_frame.to_csv(
        args.output_dir / "layer_selection_counts.csv", index=False
    )
    original_predictions.to_csv(
        args.output_dir / "original_split_predictions.csv", index=False
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
