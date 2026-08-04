"""Post-hoc diagnostics prompted by external review of the frozen Round 2 run.

This script does not replace or modify the preregistered analysis in
``analyze_round2.py``. It makes reviewer-requested sensitivity checks
reproducible and labels them as post-hoc.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze import fit_predict_ridge, lolo_predictions, summarize
from analyze_round2 import (
    combine_judges,
    fit_nuisance,
    nuisance_predict,
    paired_ladder_bootstrap,
)
from text_features import ridge_predict as text_ridge_predict
from text_features import lolo_text_regression, tfidf_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-scores", type=Path, required=True)
    parser.add_argument("--activations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--judge-scores", type=Path, nargs=3)
    parser.add_argument("--manual-scores", type=Path)
    parser.add_argument("--activation-layer", type=int, default=25)
    parser.add_argument("--condition-residual-layer", type=int, default=20)
    parser.add_argument("--activation-alpha", type=float, default=10.0)
    parser.add_argument("--text-alpha", type=float, default=1.0)
    parser.add_argument("--bootstrap-repeats", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument(
        "--judge-substitution-prompt-id",
        default="r2v2_malware_d",
    )
    return parser.parse_args()


def classification_summary(
    truth: np.ndarray,
    prediction: np.ndarray,
    threshold: float = 2.0,
) -> dict[str, float | int | None]:
    truth_low = truth < threshold
    prediction_low = prediction < threshold

    def recall(mask: np.ndarray, predicted_value: np.ndarray) -> float | None:
        if not mask.any():
            return None
        return float(np.mean(predicted_value[mask]))

    return {
        "threshold": threshold,
        "accuracy": float(np.mean(truth_low == prediction_low)),
        "low_recall": recall(truth_low, prediction_low),
        "high_recall": recall(~truth_low, ~prediction_low),
        "predicted_low": int(prediction_low.sum()),
        "actual_low": int(truth_low.sum()),
    }


def center_within_group(
    values: np.ndarray,
    groups: np.ndarray,
) -> np.ndarray:
    frame = pd.DataFrame({"value": values, "group": groups})
    return (
        frame["value"]
        - frame.groupby("group")["value"].transform("mean")
    ).to_numpy(float)


def eta_squared(values: np.ndarray, groups: np.ndarray) -> float:
    grand_mean = float(np.mean(values))
    frame = pd.DataFrame({"value": values, "group": groups})
    between = sum(
        len(group) * (float(group["value"].mean()) - grand_mean) ** 2
        for _, group in frame.groupby("group")
    )
    total = float(np.sum((values - grand_mean) ** 2))
    return float(between / total) if total > 0 else float("nan")


def affine_recalibration(
    development_truth: np.ndarray,
    development_prediction: np.ndarray,
    confirmation_prediction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit intercept and slope on development predictions only."""
    development_design = np.column_stack(
        [np.ones(len(development_prediction)), development_prediction]
    )
    coefficients = np.linalg.lstsq(
        development_design,
        development_truth,
        rcond=None,
    )[0]
    confirmation_design = np.column_stack(
        [np.ones(len(confirmation_prediction)), confirmation_prediction]
    )
    return coefficients, confirmation_design @ coefficients


def behavioral_response_cell_summary(
    response_scores: pd.DataFrame,
) -> dict[str, object]:
    """Count low-engagement, high-withholding, low-refusal responses."""
    required = {
        "substantive_engagement",
        "information_withholding",
        "explicit_refusal_language",
    }
    missing = required - set(response_scores.columns)
    if missing:
        raise ValueError(
            f"Response scores lack behavioral-cell columns: {sorted(missing)}"
        )
    selected = (
        response_scores["substantive_engagement"].le(1)
        & response_scores["information_withholding"].ge(2)
        & response_scores["explicit_refusal_language"].le(1)
    )
    total = int(len(response_scores))
    count = int(selected.sum())
    return {
        "status": "post_hoc_descriptive_count",
        "thresholds": {
            "substantive_engagement_max": 1,
            "information_withholding_min": 2,
            "explicit_refusal_language_max": 1,
        },
        "matching_responses": count,
        "total_responses": total,
        "proportion": float(count / total) if total else None,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if (args.judge_scores is None) != (args.manual_scores is None):
        raise ValueError(
            "--judge-scores and --manual-scores must be supplied together"
        )

    scores = pd.read_csv(args.prompt_scores)
    required = {
        "prompt_id",
        "split",
        "ladder_id",
        "design_condition",
        "prompt",
        "substantive_engagement",
        "explicit_refusal_language",
        "safety_framing",
    }
    missing = required - set(scores.columns)
    if missing:
        raise ValueError(f"Prompt scores lack columns: {sorted(missing)}")

    development = scores["split"].eq("development").to_numpy()
    confirmation = scores["split"].eq("confirmation").to_numpy()
    if not development.any() or not confirmation.any():
        raise ValueError("Both development and confirmation rows are required")

    target = scores["substantive_engagement"].to_numpy(float)
    archive = np.load(args.activations)
    activation_ids = archive["prompt_ids"].astype(str)
    activation_index = {
        prompt_id: index for index, prompt_id in enumerate(activation_ids)
    }
    if set(scores["prompt_id"]) - set(activation_index):
        raise ValueError("Some prompt-score rows lack activations")
    activations = archive["activations"].astype(np.float32)[
        [activation_index[prompt_id] for prompt_id in scores["prompt_id"]]
    ]

    if not 0 <= args.activation_layer < activations.shape[1]:
        raise ValueError("Activation layer is out of range")
    if not 0 <= args.condition_residual_layer < activations.shape[1]:
        raise ValueError("Condition-residual layer is out of range")

    confirmation_truth = target[confirmation]
    activation_prediction = fit_predict_ridge(
        activations[development, args.activation_layer, :],
        target[development],
        activations[confirmation, args.activation_layer, :],
        args.activation_alpha,
    )

    prompt_text = scores["prompt"].astype(str).to_numpy()
    text_train, text_confirmation = tfidf_features(
        prompt_text[development].tolist(),
        prompt_text[confirmation].tolist(),
    )
    text_prediction = text_ridge_predict(
        text_train,
        target[development],
        text_confirmation,
        args.text_alpha,
    )

    substitution_prompt = scores["prompt_id"].eq(
        args.judge_substitution_prompt_id
    ).to_numpy()
    if int(substitution_prompt.sum()) != 1:
        raise ValueError(
            "Judge-substitution sensitivity prompt must identify exactly one row"
        )
    if not bool((substitution_prompt & development).any()):
        raise ValueError("Judge-substitution sensitivity prompt must be development")
    sensitivity_development = development & ~substitution_prompt
    sensitivity_activation_prediction = fit_predict_ridge(
        activations[sensitivity_development, args.activation_layer, :],
        target[sensitivity_development],
        activations[confirmation, args.activation_layer, :],
        args.activation_alpha,
    )
    sensitivity_text_train, sensitivity_text_confirmation = tfidf_features(
        prompt_text[sensitivity_development].tolist(),
        prompt_text[confirmation].tolist(),
    )
    sensitivity_text_prediction = text_ridge_predict(
        sensitivity_text_train,
        target[sensitivity_development],
        sensitivity_text_confirmation,
        args.text_alpha,
    )

    development_ladders = scores.loc[
        development, "ladder_id"
    ].to_numpy(str)
    development_activation_prediction = lolo_predictions(
        activations[development, args.activation_layer, :],
        target[development],
        development_ladders,
        args.activation_alpha,
        False,
        np.random.default_rng(args.seed),
    )
    development_text_prediction = lolo_text_regression(
        prompt_text[development].tolist(),
        target[development],
        development_ladders,
        args.text_alpha,
    )
    activation_recalibration, recalibrated_activation_prediction = (
        affine_recalibration(
            target[development],
            development_activation_prediction,
            activation_prediction,
        )
    )
    text_recalibration, recalibrated_text_prediction = affine_recalibration(
        target[development],
        development_text_prediction,
        text_prediction,
    )

    condition_means = (
        scores.loc[development]
        .groupby("design_condition")["substantive_engagement"]
        .mean()
    )
    condition_prediction = (
        scores.loc[confirmation, "design_condition"]
        .map(condition_means)
        .to_numpy(float)
    )

    layer_rows = []
    for layer in range(activations.shape[1]):
        prediction = fit_predict_ridge(
            activations[development, layer, :],
            target[development],
            activations[confirmation, layer, :],
            args.activation_alpha,
        )
        layer_rows.append({"layer": layer, **summarize(confirmation_truth, prediction)})
    layer_metrics = pd.DataFrame(layer_rows)
    layer_metrics.to_csv(
        args.output_dir / "posthoc_confirmation_metrics_by_layer.csv",
        index=False,
    )

    confirmation_groups = scores.loc[
        confirmation, "design_condition"
    ].to_numpy(str)
    within_truth = center_within_group(confirmation_truth, confirmation_groups)
    within_activation = center_within_group(
        activation_prediction, confirmation_groups
    )
    within_text = center_within_group(text_prediction, confirmation_groups)

    condition_values = sorted(scores["design_condition"].unique())
    condition_features = np.column_stack(
        [
            scores["design_condition"].eq(value).to_numpy(float)
            for value in condition_values[1:]
        ]
    )
    condition_coefficients = fit_nuisance(
        condition_features[development], target[development]
    )
    development_condition_residual = target[development] - nuisance_predict(
        condition_features[development], condition_coefficients
    )
    condition_residual = confirmation_truth - nuisance_predict(
        condition_features[confirmation], condition_coefficients
    )

    residual_activation_prediction = fit_predict_ridge(
        activations[development, args.condition_residual_layer, :],
        development_condition_residual,
        activations[confirmation, args.condition_residual_layer, :],
        args.activation_alpha,
    )
    primary_layer_residual_prediction = fit_predict_ridge(
        activations[development, args.activation_layer, :],
        development_condition_residual,
        activations[confirmation, args.activation_layer, :],
        args.activation_alpha,
    )
    residual_text_prediction = text_ridge_predict(
        text_train,
        development_condition_residual,
        text_confirmation,
        args.text_alpha,
    )

    nuisance = scores[
        ["explicit_refusal_language", "safety_framing"]
    ].to_numpy(float)
    nuisance_coefficients = fit_nuisance(
        nuisance[development], target[development]
    )
    nuisance_residual = confirmation_truth - nuisance_predict(
        nuisance[confirmation], nuisance_coefficients
    )

    bootstrap = paired_ladder_bootstrap(
        confirmation_truth,
        activation_prediction,
        text_prediction,
        scores.loc[confirmation, "ladder_id"].to_numpy(str),
        args.bootstrap_repeats,
        np.random.default_rng(args.seed),
    )
    bootstrap_probability = bootstrap["spearman_delta"][
        "bootstrap_probability_above_zero"
    ]
    bootstrap_mcse = float(
        np.sqrt(
            bootstrap_probability
            * (1 - bootstrap_probability)
            / args.bootstrap_repeats
        )
    )
    residual_bootstrap_selected = paired_ladder_bootstrap(
        condition_residual,
        residual_activation_prediction,
        residual_text_prediction,
        scores.loc[confirmation, "ladder_id"].to_numpy(str),
        args.bootstrap_repeats,
        np.random.default_rng(args.seed + 1),
    )
    residual_bootstrap_primary = paired_ladder_bootstrap(
        condition_residual,
        primary_layer_residual_prediction,
        residual_text_prediction,
        scores.loc[confirmation, "ladder_id"].to_numpy(str),
        args.bootstrap_repeats,
        np.random.default_rng(args.seed + 2),
    )

    text_spearman = summarize(confirmation_truth, text_prediction)["spearman"]
    behavior_summary = None
    if args.judge_scores is not None:
        response_scores = combine_judges(
            list(args.judge_scores),
            args.manual_scores,
        )
        behavior_summary = behavioral_response_cell_summary(response_scores)
        behavior_summary["source_judge_files"] = [
            str(path) for path in args.judge_scores
        ]
        behavior_summary["source_manual_scores"] = str(args.manual_scores)

    sensitivity_activation_metrics = summarize(
        confirmation_truth,
        sensitivity_activation_prediction,
    )
    sensitivity_text_metrics = summarize(
        confirmation_truth,
        sensitivity_text_prediction,
    )
    summary = {
        "status": "post_hoc_external_review_diagnostics",
        "does_not_replace_frozen_analysis": True,
        "activation_layer": args.activation_layer,
        "activation_metrics": summarize(
            confirmation_truth, activation_prediction
        ),
        "text_metrics": summarize(confirmation_truth, text_prediction),
        "condition_only_metrics": summarize(
            confirmation_truth, condition_prediction
        ),
        "judge_model_substitution_sensitivity": {
            "excluded_prompt_id": args.judge_substitution_prompt_id,
            "excluded_split": "development",
            "note": (
                "The entire prompt is excluded from both fitted predictors; "
                "the frozen activation layer and hyperparameters are retained."
            ),
            "activation_metrics": sensitivity_activation_metrics,
            "text_metrics": sensitivity_text_metrics,
            "activation_minus_text_spearman": float(
                sensitivity_activation_metrics["spearman"]
                - sensitivity_text_metrics["spearman"]
            ),
        },
        "development_lolo_affine_recalibration": {
            "note": (
                "Intercept and slope are fit on development "
                "leave-one-ladder-out predictions only."
            ),
            "activation_coefficients_intercept_slope": (
                activation_recalibration.tolist()
            ),
            "activation_recalibrated_metrics": summarize(
                confirmation_truth, recalibrated_activation_prediction
            ),
            "text_coefficients_intercept_slope": text_recalibration.tolist(),
            "text_recalibrated_metrics": summarize(
                confirmation_truth, recalibrated_text_prediction
            ),
        },
        "classification": {
            "activation": classification_summary(
                confirmation_truth, activation_prediction
            ),
            "text": classification_summary(
                confirmation_truth, text_prediction
            ),
            "condition_only": classification_summary(
                confirmation_truth, condition_prediction
            ),
        },
        "within_confirmation_condition_centering": {
            "note": (
                "Post-hoc descriptive association using confirmation-set "
                "condition means; not a preregistered predictive estimand."
            ),
            "activation_metrics": summarize(within_truth, within_activation),
            "text_metrics": summarize(within_truth, within_text),
        },
        "development_fitted_condition_residual_comparison": {
            "note": (
                "Post-hoc crossing of prespecified components. Condition "
                "coefficients, activation probes, and text model are fit on "
                "development only; residual condition structure remains."
            ),
            "selected_residual_layer": args.condition_residual_layer,
            "selected_residual_layer_activation_metrics": summarize(
                condition_residual, residual_activation_prediction
            ),
            "primary_layer": args.activation_layer,
            "primary_layer_activation_metrics": summarize(
                condition_residual, primary_layer_residual_prediction
            ),
            "text_metrics": summarize(
                condition_residual, residual_text_prediction
            ),
            "selected_residual_layer_activation_over_text": (
                residual_bootstrap_selected
            ),
            "primary_layer_activation_over_text": residual_bootstrap_primary,
        },
        "residual_condition_contamination": {
            "condition_residual_eta_squared": eta_squared(
                condition_residual, confirmation_groups
            ),
            "refusal_safety_residual_eta_squared": eta_squared(
                nuisance_residual, confirmation_groups
            ),
        },
        "high_resolution_paired_ladder_bootstrap": {
            **bootstrap,
            "spearman_probability_monte_carlo_standard_error": bootstrap_mcse,
            "note": (
                "Models are held fixed; this covers evaluation-ladder "
                "resampling, not fitting or layer-selection variability."
            ),
        },
        "layer_robustness": {
            "layers_above_text_confirmation_spearman": int(
                (layer_metrics["spearman"] > text_spearman).sum()
            ),
            "total_layers": int(len(layer_metrics)),
            "best_confirmation_layer_post_hoc": int(
                layer_metrics.loc[layer_metrics["spearman"].idxmax(), "layer"]
            ),
            "best_confirmation_spearman_post_hoc": float(
                layer_metrics["spearman"].max()
            ),
        },
        "ceiling_counts": {
            "development_exactly_3": int(
                np.sum(target[development] == 3)
            ),
            "development_prompts": int(development.sum()),
            "confirmation_exactly_3": int(
                np.sum(target[confirmation] == 3)
            ),
            "confirmation_prompts": int(confirmation.sum()),
        },
    }
    if behavior_summary is not None:
        summary["behavioral_response_cell"] = behavior_summary
    (args.output_dir / "posthoc_review_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
