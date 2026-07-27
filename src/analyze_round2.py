"""Locked development/confirmation analysis for the Round 2 protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze import (
    attach_token_cap_flags,
    fit_predict_ridge,
    lolo_predictions,
    read_jsonl,
    summarize,
)
from scoring_schema import ROUND2_SCORE_COLUMNS
from text_features import (
    lolo_text_regression,
    ridge_predict as text_ridge_predict,
    tfidf_features,
)


KEY_COLUMNS = ["prompt_id", "generation_id"]
NUISANCE_COLUMNS = [
    "explicit_refusal_language",
    "safety_framing",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--activations", type=Path, required=True)
    parser.add_argument("--run-config", type=Path)
    parser.add_argument("--judge-files", type=Path, nargs="+", required=True)
    parser.add_argument("--manual-scores", type=Path)
    parser.add_argument("--prompt-scores", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--shuffle-repeats", type=int, default=20)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260801)
    return parser.parse_args()


def validate_score_frame(frame: pd.DataFrame, source: Path) -> None:
    missing = set(KEY_COLUMNS + ROUND2_SCORE_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"{source} lacks columns: {sorted(missing)}")
    if frame.duplicated(KEY_COLUMNS).any():
        raise ValueError(f"{source} contains duplicate response keys")
    for column in ROUND2_SCORE_COLUMNS:
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or ((values < 0) | (values > 3)).any():
            raise ValueError(f"{source}: {column} must contain scores from 0 to 3")


def combine_judges(
    judge_files: list[Path],
    manual_scores: Path | None,
) -> pd.DataFrame:
    frames = []
    for judge, path in enumerate(judge_files, start=1):
        frame = read_jsonl(path)
        validate_score_frame(frame, path)
        frame = frame[KEY_COLUMNS + ROUND2_SCORE_COLUMNS].copy()
        frame["judge"] = judge
        frames.append(frame)
    long = pd.concat(frames, ignore_index=True)
    means = (
        long.groupby(KEY_COLUMNS, sort=False)[ROUND2_SCORE_COLUMNS]
        .mean()
        .reset_index()
    )
    if manual_scores is None:
        return means
    manual = read_jsonl(manual_scores)
    if set(KEY_COLUMNS) - set(manual.columns):
        raise ValueError("Manual scores lack prompt_id or generation_id")
    indexed = means.set_index(KEY_COLUMNS)
    for _, row in manual.iterrows():
        key = (row["prompt_id"], int(row["generation_id"]))
        if key not in indexed.index:
            raise ValueError(f"Unknown manual score key: {key}")
        for column in ROUND2_SCORE_COLUMNS:
            if column in manual.columns and pd.notna(row.get(column)):
                value = float(row[column])
                if not 0 <= value <= 3:
                    raise ValueError(f"Manual {column} outside 0 to 3 for {key}")
                indexed.loc[key, column] = value
    return indexed.reset_index()


def fit_nuisance(
    features: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    design = np.column_stack([np.ones(len(features)), features])
    return np.linalg.lstsq(design, target, rcond=None)[0]


def nuisance_predict(features: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(features)), features]) @ coefficients


def lolo_residual_predictions(
    activations: np.ndarray,
    target: np.ndarray,
    nuisance: np.ndarray,
    ladders: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    true_residual = np.full(target.shape, np.nan, dtype=float)
    predicted_residual = np.full(target.shape, np.nan, dtype=float)
    for held_out in np.unique(ladders):
        train = ladders != held_out
        test = ~train
        coefficients = fit_nuisance(nuisance[train], target[train])
        train_residual = target[train] - nuisance_predict(
            nuisance[train], coefficients
        )
        true_residual[test] = target[test] - nuisance_predict(
            nuisance[test], coefficients
        )
        predicted_residual[test] = fit_predict_ridge(
            activations[train],
            train_residual,
            activations[test],
            alpha,
        )
    return true_residual, predicted_residual


def metrics_with_prefix(
    truth: np.ndarray,
    predicted: np.ndarray,
    prefix: str,
) -> dict[str, float]:
    return {
        f"{prefix}_{key}": value
        for key, value in summarize(truth, predicted).items()
    }


def paired_ladder_bootstrap(
    truth: np.ndarray,
    activation_prediction: np.ndarray,
    text_prediction: np.ndarray,
    ladders: np.ndarray,
    repeats: int,
    rng: np.random.Generator,
) -> dict:
    """Bootstrap activation-minus-text metrics at the ladder level."""
    unique_ladders = np.unique(ladders)
    samples = []
    for _ in range(repeats):
        sampled_ladders = rng.choice(
            unique_ladders,
            size=len(unique_ladders),
            replace=True,
        )
        indices = np.concatenate(
            [np.flatnonzero(ladders == ladder) for ladder in sampled_ladders]
        )
        activation_metrics = summarize(
            truth[indices],
            activation_prediction[indices],
        )
        text_metrics = summarize(truth[indices], text_prediction[indices])
        samples.append(
            [
                activation_metrics["spearman"] - text_metrics["spearman"],
                activation_metrics["pearson"] - text_metrics["pearson"],
                text_metrics["mae"] - activation_metrics["mae"],
            ]
        )
    samples_array = np.asarray(samples)
    activation_metrics = summarize(truth, activation_prediction)
    text_metrics = summarize(truth, text_prediction)
    names_and_observed = {
        "spearman_delta": (
            activation_metrics["spearman"] - text_metrics["spearman"]
        ),
        "pearson_delta": (
            activation_metrics["pearson"] - text_metrics["pearson"]
        ),
        "mae_improvement": text_metrics["mae"] - activation_metrics["mae"],
    }
    output = {"bootstrap_repeats": repeats}
    for column, (name, observed) in enumerate(names_and_observed.items()):
        output[name] = {
            "observed": float(observed),
            "bootstrap_95_percent_interval": [
                float(np.quantile(samples_array[:, column], 0.025)),
                float(np.quantile(samples_array[:, column], 0.975)),
            ],
            "bootstrap_probability_above_zero": float(
                np.mean(samples_array[:, column] > 0)
            ),
        }
    return output


def main() -> None:
    args = parse_args()
    if len(args.judge_files) < 2:
        raise ValueError("At least two judge files are required")
    if args.bootstrap_repeats < 1:
        raise ValueError("--bootstrap-repeats must be at least 1")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = read_jsonl(args.dataset)
    required_dataset = {
        "prompt_id",
        "ladder_id",
        "split",
        "category",
        "rung",
        "design_cell",
        "design_condition",
        "prompt",
    }
    if required_dataset - set(dataset.columns):
        raise ValueError("Round 2 dataset lacks required design fields")
    split_values = set(dataset["split"])
    if split_values != {"development", "confirmation"}:
        raise ValueError(
            "Dataset split must contain exactly development and confirmation"
        )
    if set(
        dataset.loc[dataset["split"] == "development", "ladder_id"]
    ) & set(dataset.loc[dataset["split"] == "confirmation", "ladder_id"]):
        raise ValueError("A ladder appears in both dataset splits")

    responses = read_jsonl(args.responses)
    responses, max_new_tokens = attach_token_cap_flags(
        responses,
        args.run_config,
    )
    scores = combine_judges(args.judge_files, args.manual_scores)
    expected = responses[KEY_COLUMNS].drop_duplicates()
    if len(expected.merge(scores, on=KEY_COLUMNS, how="inner")) != len(expected):
        raise ValueError("Some generated responses lack scores")
    response_scores = responses.merge(
        scores,
        on=KEY_COLUMNS,
        how="inner",
        validate="one_to_one",
    )
    prompt_scores = (
        response_scores.groupby("prompt_id", as_index=False)
        .agg(
            **{
                column: (column, "mean")
                for column in ROUND2_SCORE_COLUMNS
            },
            response_length_tokens=("response_length_tokens", "mean"),
            capped_generations=("hit_token_cap", "sum"),
            hit_token_cap_rate=("hit_token_cap", "mean"),
            num_generations=("generation_id", "nunique"),
        )
        .merge(
            dataset[
                [
                    "prompt_id",
                    "ladder_id",
                    "split",
                    "category",
                    "rung",
                    "design_cell",
                    "design_condition",
                    "prompt",
                    "intended_information_supply",
                    "intended_refusal_language",
                    "intended_refusal_basis",
                    "variability_subset",
                    "generation_count",
                ]
            ],
            on="prompt_id",
            how="left",
            validate="one_to_one",
        )
    )
    if prompt_scores["ladder_id"].isna().any():
        raise ValueError("Scores contain prompt IDs absent from the dataset")
    if (
        prompt_scores["num_generations"].astype(int)
        != prompt_scores["generation_count"].astype(int)
    ).any():
        raise ValueError("Observed generation counts differ from the frozen dataset")
    if args.prompt_scores:
        prompt_annotations = read_jsonl(args.prompt_scores)
        prompt_scores = prompt_scores.merge(
            prompt_annotations[["prompt_id", "prompt_harmfulness"]],
            on="prompt_id",
            how="left",
            validate="one_to_one",
        )
        if prompt_scores["prompt_harmfulness"].isna().any():
            raise ValueError("Some prompts lack harmfulness annotations")

    archive = np.load(args.activations)
    activations = archive["activations"].astype(np.float32)
    activation_ids = archive["prompt_ids"].astype(str)
    activation_index = {
        prompt_id: index for index, prompt_id in enumerate(activation_ids)
    }
    if set(prompt_scores["prompt_id"]) - set(activation_index):
        raise ValueError("Some prompts lack activations")
    activations = activations[
        np.asarray([activation_index[value] for value in prompt_scores["prompt_id"]])
    ]

    prompt_scores.to_csv(args.output_dir / "prompt_level_scores.csv", index=False)
    summary_aggregations = {
        "prompts": ("prompt_id", "size"),
        **{
            f"mean_{column}": (column, "mean")
            for column in ROUND2_SCORE_COLUMNS
        },
        "mean_response_length_tokens": ("response_length_tokens", "mean"),
        "mean_token_cap_rate": ("hit_token_cap_rate", "mean"),
    }
    (
        prompt_scores.groupby(["split", "design_cell"], as_index=False)
        .agg(**summary_aggregations)
        .to_csv(args.output_dir / "design_cell_summary.csv", index=False)
    )
    (
        prompt_scores.groupby(["split", "category"], as_index=False)
        .agg(**summary_aggregations)
        .to_csv(args.output_dir / "domain_summary.csv", index=False)
    )
    (
        prompt_scores.loc[prompt_scores["design_cell"].eq("D")]
        .groupby(["split", "design_condition"], as_index=False)
        .agg(**summary_aggregations)
        .to_csv(args.output_dir / "refusal_basis_summary.csv", index=False)
    )
    development = prompt_scores["split"].eq("development").to_numpy()
    confirmation = prompt_scores["split"].eq("confirmation").to_numpy()
    development_ladders = prompt_scores.loc[development, "ladder_id"].to_numpy()
    rng = np.random.default_rng(args.seed)
    metric_rows: list[dict] = []
    target_summaries: list[dict] = []
    prediction_rows: list[pd.DataFrame] = []

    for target_name in ROUND2_SCORE_COLUMNS:
        target = prompt_scores[target_name].to_numpy(dtype=float)
        for layer in range(activations.shape[1]):
            predictions = lolo_predictions(
                activations[development, layer, :],
                target[development],
                development_ladders,
                args.ridge_alpha,
                False,
                rng,
            )
            row = {
                "target": target_name,
                "layer": layer,
                **summarize(target[development], predictions),
                "shuffled_spearman_mean": np.nan,
                "shuffled_spearman_std": np.nan,
            }
            if target_name == "substantive_engagement":
                shuffled_values = []
                for _ in range(args.shuffle_repeats):
                    shuffled = lolo_predictions(
                        activations[development, layer, :],
                        target[development],
                        development_ladders,
                        args.ridge_alpha,
                        True,
                        rng,
                    )
                    shuffled_values.append(
                        summarize(target[development], shuffled)["spearman"]
                    )
                row["shuffled_spearman_mean"] = float(
                    np.mean(shuffled_values)
                )
                row["shuffled_spearman_std"] = float(np.std(shuffled_values))
            metric_rows.append(row)

    metrics = pd.DataFrame(metric_rows)
    selected_layers: dict[str, int] = {}
    for target_name in ROUND2_SCORE_COLUMNS:
        target_metrics = metrics.loc[metrics["target"] == target_name]
        best_value = target_metrics["spearman"].max()
        selected_layer = int(
            target_metrics.loc[
                target_metrics["spearman"].eq(best_value), "layer"
            ].min()
        )
        selected_layers[target_name] = selected_layer
        target = prompt_scores[target_name].to_numpy(dtype=float)
        prediction = fit_predict_ridge(
            activations[development, selected_layer, :],
            target[development],
            activations[confirmation, selected_layer, :],
            args.ridge_alpha,
        )
        confirmation_metrics = summarize(target[confirmation], prediction)
        development_best = target_metrics.loc[
            target_metrics["layer"].eq(selected_layer)
        ].iloc[0]
        target_summaries.append(
            {
                "target": target_name,
                "selected_layer": selected_layer,
                "development_spearman": float(development_best["spearman"]),
                "development_pearson": float(development_best["pearson"]),
                "development_mae": float(development_best["mae"]),
                "confirmation_spearman": confirmation_metrics["spearman"],
                "confirmation_pearson": confirmation_metrics["pearson"],
                "confirmation_mae": confirmation_metrics["mae"],
            }
        )
        prediction_rows.append(
            pd.DataFrame(
                {
                    "prompt_id": prompt_scores.loc[confirmation, "prompt_id"],
                    "ladder_id": prompt_scores.loc[confirmation, "ladder_id"],
                    "target": target_name,
                    "selected_layer": selected_layer,
                    "true_score": target[confirmation],
                    "predicted_score": prediction,
                }
            )
        )

    engagement = prompt_scores["substantive_engagement"].to_numpy(dtype=float)
    nuisance = prompt_scores[NUISANCE_COLUMNS].to_numpy(dtype=float)
    residual_metric_rows: list[dict] = []
    for layer in range(activations.shape[1]):
        true_residual, predicted_residual = lolo_residual_predictions(
            activations[development, layer, :],
            engagement[development],
            nuisance[development],
            development_ladders,
            args.ridge_alpha,
        )
        residual_metric_rows.append(
            {
                "target": "engagement_residual",
                "layer": layer,
                **summarize(true_residual, predicted_residual),
            }
        )
    residual_metrics = pd.DataFrame(residual_metric_rows)
    residual_best_value = residual_metrics["spearman"].max()
    residual_layer = int(
        residual_metrics.loc[
            residual_metrics["spearman"].eq(residual_best_value), "layer"
        ].min()
    )
    nuisance_coefficients = fit_nuisance(
        nuisance[development],
        engagement[development],
    )
    development_residual = engagement[development] - nuisance_predict(
        nuisance[development], nuisance_coefficients
    )
    confirmation_residual = engagement[confirmation] - nuisance_predict(
        nuisance[confirmation], nuisance_coefficients
    )
    predicted_confirmation_residual = fit_predict_ridge(
        activations[development, residual_layer, :],
        development_residual,
        activations[confirmation, residual_layer, :],
        args.ridge_alpha,
    )
    residual_confirmation_metrics = summarize(
        confirmation_residual,
        predicted_confirmation_residual,
    )
    target_summaries.append(
        {
            "target": "engagement_residual",
            "selected_layer": residual_layer,
            "development_spearman": float(
                residual_metrics.loc[
                    residual_metrics["layer"].eq(residual_layer), "spearman"
                ].iloc[0]
            ),
            "development_pearson": float(
                residual_metrics.loc[
                    residual_metrics["layer"].eq(residual_layer), "pearson"
                ].iloc[0]
            ),
            "development_mae": float(
                residual_metrics.loc[
                    residual_metrics["layer"].eq(residual_layer), "mae"
                ].iloc[0]
            ),
            "confirmation_spearman": residual_confirmation_metrics["spearman"],
            "confirmation_pearson": residual_confirmation_metrics["pearson"],
            "confirmation_mae": residual_confirmation_metrics["mae"],
        }
    )

    # A second residual target asks whether activations predict engagement
    # variation beyond the coarse experimental condition itself.
    condition_values = sorted(prompt_scores["design_condition"].unique())
    condition_features = np.column_stack(
        [
            prompt_scores["design_condition"].eq(value).to_numpy(float)
            for value in condition_values[1:]
        ]
    )
    condition_metric_rows: list[dict] = []
    for layer in range(activations.shape[1]):
        true_residual, predicted_residual = lolo_residual_predictions(
            activations[development, layer, :],
            engagement[development],
            condition_features[development],
            development_ladders,
            args.ridge_alpha,
        )
        condition_metric_rows.append(
            {
                "target": "engagement_condition_residual",
                "layer": layer,
                **summarize(true_residual, predicted_residual),
            }
        )
    condition_metrics = pd.DataFrame(condition_metric_rows)
    condition_best = condition_metrics["spearman"].max()
    condition_layer = int(
        condition_metrics.loc[
            condition_metrics["spearman"].eq(condition_best), "layer"
        ].min()
    )
    condition_coefficients = fit_nuisance(
        condition_features[development],
        engagement[development],
    )
    development_condition_residual = engagement[
        development
    ] - nuisance_predict(
        condition_features[development], condition_coefficients
    )
    confirmation_condition_residual = engagement[
        confirmation
    ] - nuisance_predict(
        condition_features[confirmation], condition_coefficients
    )
    predicted_confirmation_condition_residual = fit_predict_ridge(
        activations[development, condition_layer, :],
        development_condition_residual,
        activations[confirmation, condition_layer, :],
        args.ridge_alpha,
    )
    condition_confirmation_metrics = summarize(
        confirmation_condition_residual,
        predicted_confirmation_condition_residual,
    )
    target_summaries.append(
        {
            "target": "engagement_condition_residual",
            "selected_layer": condition_layer,
            "development_spearman": float(
                condition_metrics.loc[
                    condition_metrics["layer"].eq(condition_layer), "spearman"
                ].iloc[0]
            ),
            "development_pearson": float(
                condition_metrics.loc[
                    condition_metrics["layer"].eq(condition_layer), "pearson"
                ].iloc[0]
            ),
            "development_mae": float(
                condition_metrics.loc[
                    condition_metrics["layer"].eq(condition_layer), "mae"
                ].iloc[0]
            ),
            "confirmation_spearman": condition_confirmation_metrics["spearman"],
            "confirmation_pearson": condition_confirmation_metrics["pearson"],
            "confirmation_mae": condition_confirmation_metrics["mae"],
        }
    )

    primary_layer = selected_layers["substantive_engagement"]
    primary_confirmation = next(
        frame
        for frame in prediction_rows
        if frame["target"].iloc[0] == "substantive_engagement"
    )
    per_ladder_rows = []
    for ladder_id, frame in primary_confirmation.groupby("ladder_id"):
        per_ladder_rows.append(
            {
                "ladder_id": ladder_id,
                **summarize(
                    frame["true_score"].to_numpy(),
                    frame["predicted_score"].to_numpy(),
                ),
            }
        )

    cross_target_rows = []
    for prediction_frame in prediction_rows:
        trained_target = str(prediction_frame["target"].iloc[0])
        predicted = prediction_frame["predicted_score"].to_numpy(float)
        for evaluated_target in ROUND2_SCORE_COLUMNS:
            truth = prompt_scores.loc[
                confirmation, evaluated_target
            ].to_numpy(float)
            cross_target_rows.append(
                {
                    "trained_target": trained_target,
                    "evaluated_target": evaluated_target,
                    **summarize(truth, predicted),
                }
            )

    length_prediction = fit_predict_ridge(
        prompt_scores.loc[development, ["response_length_tokens"]].to_numpy(float),
        engagement[development],
        prompt_scores.loc[confirmation, ["response_length_tokens"]].to_numpy(float),
        args.ridge_alpha,
    )
    cap_prediction = fit_predict_ridge(
        prompt_scores.loc[development, ["hit_token_cap_rate"]].to_numpy(float),
        engagement[development],
        prompt_scores.loc[confirmation, ["hit_token_cap_rate"]].to_numpy(float),
        args.ridge_alpha,
    )
    nuisance_prediction = nuisance_predict(
        nuisance[confirmation],
        nuisance_coefficients,
    )
    prompt_text = prompt_scores["prompt"].astype(str).to_numpy()
    development_text_prediction = lolo_text_regression(
        prompt_text[development].tolist(),
        engagement[development],
        development_ladders,
        alpha=1.0,
    )
    text_train, text_confirmation = tfidf_features(
        prompt_text[development].tolist(),
        prompt_text[confirmation].tolist(),
    )
    text_prediction = text_ridge_predict(
        text_train,
        engagement[development],
        text_confirmation,
        alpha=1.0,
    )
    harmfulness_metrics = None
    if "prompt_harmfulness" in prompt_scores.columns:
        harmfulness_prediction = fit_predict_ridge(
            prompt_scores.loc[
                development, ["prompt_harmfulness"]
            ].to_numpy(float),
            engagement[development],
            prompt_scores.loc[
                confirmation, ["prompt_harmfulness"]
            ].to_numpy(float),
            args.ridge_alpha,
        )
        harmfulness_metrics = summarize(
            engagement[confirmation],
            harmfulness_prediction,
        )
    no_cap = prompt_scores.loc[confirmation, "capped_generations"].eq(0).to_numpy()
    primary_truth = primary_confirmation["true_score"].to_numpy(float)
    primary_prediction = primary_confirmation["predicted_score"].to_numpy(float)
    confirmation_ladders = prompt_scores.loc[
        confirmation, "ladder_id"
    ].to_numpy()
    primary_activation_over_text = paired_ladder_bootstrap(
        primary_truth,
        primary_prediction,
        text_prediction,
        confirmation_ladders,
        args.bootstrap_repeats,
        rng,
    )
    low_truth = primary_truth < 2
    low_prediction = primary_prediction < 2
    if low_truth.any() and (~low_truth).any():
        classification = {
            "threshold": 2.0,
            "low_engagement_prompts": int(low_truth.sum()),
            "accuracy": float(np.mean(low_truth == low_prediction)),
            "low_engagement_recall": float(
                np.mean(low_prediction[low_truth])
            ),
            "high_engagement_recall": float(
                np.mean(~low_prediction[~low_truth])
            ),
        }
    else:
        classification = {
            "threshold": 2.0,
            "low_engagement_prompts": int(low_truth.sum()),
            "accuracy": None,
            "low_engagement_recall": None,
            "high_engagement_recall": None,
        }
    sensitivity = (
        summarize(primary_truth[no_cap], primary_prediction[no_cap])
        if no_cap.any()
        else {"spearman": 0.0, "pearson": 0.0, "mae": 0.0}
    )

    full_metrics = pd.concat(
        [metrics, residual_metrics, condition_metrics],
        ignore_index=True,
    )
    full_metrics.to_csv(
        args.output_dir / "development_metrics_by_layer.csv",
        index=False,
    )
    pd.DataFrame(target_summaries).to_csv(
        args.output_dir / "target_summary.csv",
        index=False,
    )
    pd.concat(prediction_rows, ignore_index=True).to_csv(
        args.output_dir / "confirmation_predictions.csv",
        index=False,
    )
    pd.DataFrame(per_ladder_rows).to_csv(
        args.output_dir / "confirmation_per_ladder.csv",
        index=False,
    )
    pd.DataFrame(cross_target_rows).to_csv(
        args.output_dir / "cross_target_confirmation.csv",
        index=False,
    )

    selected_development_row = metrics.loc[
        metrics["target"].eq("substantive_engagement")
        & metrics["layer"].eq(primary_layer)
    ].iloc[0]
    engagement_withholding_correlation = float(
        np.corrcoef(
            engagement[development],
            prompt_scores.loc[
                development, "information_withholding"
            ].to_numpy(float),
        )[0, 1]
    )
    summary = {
        "protocol": "round2",
        "development_prompts": int(development.sum()),
        "confirmation_prompts": int(confirmation.sum()),
        "development_ladders": int(
            prompt_scores.loc[development, "ladder_id"].nunique()
        ),
        "confirmation_ladders": int(
            prompt_scores.loc[confirmation, "ladder_id"].nunique()
        ),
        "primary_target": "substantive_engagement",
        "primary_selected_layer": primary_layer,
        "primary_confirmation_metrics": summarize(
            primary_truth,
            primary_prediction,
        ),
        "primary_confirmatory_estimand": (
            "activation_minus_sparse_text_confirmation_performance"
        ),
        "primary_activation_over_text": primary_activation_over_text,
        "primary_confirmation_target_range": [
            float(primary_truth.min()),
            float(primary_truth.max()),
        ],
        "primary_confirmation_prediction_range": [
            float(primary_prediction.min()),
            float(primary_prediction.max()),
        ],
        "primary_confirmation_classification": classification,
        "development_shuffled_spearman_at_selected_layer": {
            "mean": float(selected_development_row["shuffled_spearman_mean"]),
            "standard_deviation": float(
                selected_development_row["shuffled_spearman_std"]
            ),
            "repeats": args.shuffle_repeats,
        },
        "response_length_confirmation_metrics": summarize(
            engagement[confirmation],
            length_prediction,
        ),
        "token_cap_confirmation_metrics": summarize(
            engagement[confirmation],
            cap_prediction,
        ),
        "nuisance_only_confirmation_metrics": summarize(
            engagement[confirmation],
            nuisance_prediction,
        ),
        "text_only_development_lolo_metrics": summarize(
            engagement[development],
            development_text_prediction,
        ),
        "text_only_confirmation_metrics": summarize(
            engagement[confirmation],
            text_prediction,
        ),
        "condition_residual_selected_layer": condition_layer,
        "condition_residual_confirmation_metrics": (
            condition_confirmation_metrics
        ),
        "development_engagement_withholding_pearson": (
            engagement_withholding_correlation
        ),
        "withholding_adjusted_analysis_identifiable": (
            abs(engagement_withholding_correlation) < 0.9
        ),
        "prompt_harmfulness_confirmation_metrics": harmfulness_metrics,
        "residual_engagement_selected_layer": residual_layer,
        "residual_engagement_confirmation_metrics": residual_confirmation_metrics,
        "no_capped_generation_sensitivity": {
            "prompts": int(no_cap.sum()),
            **sensitivity,
        },
        "max_new_tokens": max_new_tokens,
        "total_responses": int(len(responses)),
        "capped_responses": int(responses["hit_token_cap"].sum()),
        "ridge_alpha": args.ridge_alpha,
        "shuffle_repeats": args.shuffle_repeats,
        "bootstrap_repeats": args.bootstrap_repeats,
        "confirmation_used_for_layer_selection": False,
    }
    (args.output_dir / "round2_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    try:
        import matplotlib.pyplot as plt

        engagement_metrics = metrics.loc[
            metrics["target"].eq("substantive_engagement")
        ]
        figure, axis = plt.subplots(figsize=(8, 4.8))
        axis.plot(
            engagement_metrics["layer"],
            engagement_metrics["spearman"],
            marker="o",
            label="Development LOLO engagement",
        )
        axis.axvline(
            primary_layer,
            color="black",
            linestyle="--",
            alpha=0.6,
            label=f"Selected index {primary_layer}",
        )
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set(
            xlabel="Hidden-state index (0 = embedding output)",
            ylabel="Development held-out Spearman correlation",
            title="Round 2 development layer selection",
        )
        axis.legend()
        figure.tight_layout()
        figure.savefig(args.output_dir / "development_probe_curve.png", dpi=180)
        plt.close(figure)
    except ImportError:
        print("matplotlib unavailable; CSV outputs were saved without a plot")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
