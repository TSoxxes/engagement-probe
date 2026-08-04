"""Post-hoc response-length diagnostics for the frozen Round 2 run.

This script does not replace or modify the preregistered analysis in
``analyze_round2.py``. It asks a question the frozen protocol did not
prespecify: how much of the decoded engagement signal could instead be a
signal about how long an answer the model was preparing to produce.

Every quantity here is post-hoc. Response length is measured after
generation, so it is not an operational pre-answer competitor to the probe.
More importantly, length has two incompatible readings that this analysis
cannot separate:

* nuisance -- activations encode expected verbosity, and verbosity inflates
  judged engagement;
* mediator -- a genuine intention to engage produces both more substantive
  content and a longer answer.

Residualizing observed length may remove confounding or may remove part of
the construct itself. The residual numbers below are therefore a sensitivity
analysis that brackets interpretations, not a "length-corrected" estimate of
engagement decodability.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze import fit_predict_ridge, lolo_predictions, summarize
from analyze_round2 import fit_nuisance, nuisance_predict
from analyze_round2_review import center_within_group, eta_squared


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-scores", type=Path, required=True)
    parser.add_argument("--activations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--confirmation-predictions",
        type=Path,
        help=(
            "Frozen confirmation_predictions.csv. When supplied, the "
            "recomputed primary-layer predictions are checked against it."
        ),
    )
    parser.add_argument("--activation-layer", type=int, default=25)
    parser.add_argument("--activation-alpha", type=float, default=10.0)
    parser.add_argument("--bootstrap-repeats", type=int, default=10_000)
    parser.add_argument("--prediction-tolerance", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=20260801)
    return parser.parse_args()


def ladder_bootstrap_association(
    first: np.ndarray,
    second: np.ndarray,
    ladders: np.ndarray,
    repeats: int,
    rng: np.random.Generator,
) -> dict:
    """Ladder-resampled intervals for the association between two vectors.

    Ladders, not prompts, are the independent unit. Resamples in which either
    vector becomes constant carry no rank information; they are dropped and
    counted rather than scored as zero, which would pull intervals toward the
    null in the small fixed-engagement subgroups.
    """
    unique_ladders = np.unique(ladders)
    spearman_samples: list[float] = []
    pearson_samples: list[float] = []
    degenerate = 0
    for _ in range(repeats):
        sampled = rng.choice(
            unique_ladders,
            size=len(unique_ladders),
            replace=True,
        )
        indices = np.concatenate(
            [np.flatnonzero(ladders == ladder) for ladder in sampled]
        )
        resampled_first = first[indices]
        resampled_second = second[indices]
        if np.std(resampled_first) == 0 or np.std(resampled_second) == 0:
            degenerate += 1
            continue
        metrics = summarize(resampled_first, resampled_second)
        spearman_samples.append(metrics["spearman"])
        pearson_samples.append(metrics["pearson"])

    observed = summarize(first, second)
    output: dict[str, object] = {
        "spearman": observed["spearman"],
        "pearson": observed["pearson"],
        "bootstrap_repeats": repeats,
        "usable_bootstrap_samples": len(spearman_samples),
        "degenerate_bootstrap_samples": degenerate,
        "independent_ladders": int(len(unique_ladders)),
    }
    if spearman_samples:
        output["spearman_95_percent_interval"] = [
            float(np.quantile(spearman_samples, 0.025)),
            float(np.quantile(spearman_samples, 0.975)),
        ]
        output["pearson_95_percent_interval"] = [
            float(np.quantile(pearson_samples, 0.025)),
            float(np.quantile(pearson_samples, 0.975)),
        ]
    else:
        output["spearman_95_percent_interval"] = None
        output["pearson_95_percent_interval"] = None
    return output


def development_quantile(
    development_values: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    """Map values onto the development distribution's quantile scale.

    The rank-based length adjustment needs a monotone length transform that
    can be applied to held-out prompts without using held-out lengths to
    define it, so the development empirical distribution supplies the scale.
    """
    ordered = np.sort(development_values)
    left = np.searchsorted(ordered, values, side="left")
    right = np.searchsorted(ordered, values, side="right")
    return (left + right) / (2.0 * len(ordered))


def residual_analysis(
    activations: np.ndarray,
    engagement: np.ndarray,
    nuisance_feature: np.ndarray,
    development: np.ndarray,
    confirmation: np.ndarray,
    development_ladders: np.ndarray,
    confirmation_ladders: np.ndarray,
    primary_layer: int,
    alpha: float,
    repeats: int,
    rng: np.random.Generator,
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Predict engagement after removing a development-fitted length effect.

    Returns the summary, the confirmation residual truth, and the
    primary-layer residual prediction so per-prompt values can be saved.
    """
    feature = nuisance_feature.reshape(-1, 1)
    coefficients = fit_nuisance(feature[development], engagement[development])
    development_residual = engagement[development] - nuisance_predict(
        feature[development], coefficients
    )
    confirmation_residual = engagement[confirmation] - nuisance_predict(
        feature[confirmation], coefficients
    )

    layer_rows: list[dict] = []
    for layer in range(activations.shape[1]):
        predictions = lolo_predictions(
            activations[development, layer, :],
            development_residual,
            development_ladders,
            alpha,
            False,
            rng,
        )
        layer_rows.append(
            {
                "layer": layer,
                **summarize(development_residual, predictions),
            }
        )
    layer_metrics = pd.DataFrame(layer_rows)
    best = layer_metrics["spearman"].max()
    selected_layer = int(
        layer_metrics.loc[layer_metrics["spearman"].eq(best), "layer"].min()
    )

    primary_prediction = fit_predict_ridge(
        activations[development, primary_layer, :],
        development_residual,
        activations[confirmation, primary_layer, :],
        alpha,
    )
    selected_prediction = fit_predict_ridge(
        activations[development, selected_layer, :],
        development_residual,
        activations[confirmation, selected_layer, :],
        alpha,
    )
    summary = {
        "nuisance_coefficients_intercept_slope": coefficients.tolist(),
        "primary_layer": primary_layer,
        "primary_layer_metrics": summarize(
            confirmation_residual, primary_prediction
        ),
        "primary_layer_association": ladder_bootstrap_association(
            confirmation_residual,
            primary_prediction,
            confirmation_ladders,
            repeats,
            rng,
        ),
        "development_selected_residual_layer": selected_layer,
        "development_selected_layer_metrics": summarize(
            confirmation_residual, selected_prediction
        ),
        "development_selected_layer_association": ladder_bootstrap_association(
            confirmation_residual,
            selected_prediction,
            confirmation_ladders,
            repeats,
            rng,
        ),
    }
    return summary, confirmation_residual, primary_prediction


def main() -> None:
    args = parse_args()
    if args.bootstrap_repeats < 1:
        raise ValueError("--bootstrap-repeats must be at least 1")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    scores = pd.read_csv(args.prompt_scores)
    required = {
        "prompt_id",
        "ladder_id",
        "split",
        "design_condition",
        "substantive_engagement",
        "response_length_tokens",
    }
    missing = required - set(scores.columns)
    if missing:
        raise ValueError(f"Prompt scores lack columns: {sorted(missing)}")

    development = scores["split"].eq("development").to_numpy()
    confirmation = scores["split"].eq("confirmation").to_numpy()
    if not development.any() or not confirmation.any():
        raise ValueError("Both development and confirmation rows are required")

    engagement = scores["substantive_engagement"].to_numpy(float)
    length = scores["response_length_tokens"].to_numpy(float)
    if not np.isfinite(length).all() or (length <= 0).any():
        raise ValueError("Response lengths must be positive and finite")
    ladders = scores["ladder_id"].to_numpy(str)
    conditions = scores["design_condition"].to_numpy(str)
    development_ladders = ladders[development]
    confirmation_ladders = ladders[confirmation]

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

    rng = np.random.default_rng(args.seed)

    # 1. Observed length-engagement association, the reason for this analysis.
    observed_association = {
        "note": (
            "Response length is measured after generation. This is an "
            "alternative explanation of what the activations encode, not an "
            "operational pre-answer predictor."
        ),
        "development": summarize(length[development], engagement[development]),
        "confirmation": ladder_bootstrap_association(
            length[confirmation],
            engagement[confirmation],
            confirmation_ladders,
            args.bootstrap_repeats,
            rng,
        ),
    }

    # 2. Direct activation-to-length decoding, reported as collinearity.
    length_layer_rows: list[dict] = []
    for layer in range(activations.shape[1]):
        predictions = lolo_predictions(
            activations[development, layer, :],
            length[development],
            development_ladders,
            args.activation_alpha,
            False,
            rng,
        )
        confirmation_prediction = fit_predict_ridge(
            activations[development, layer, :],
            length[development],
            activations[confirmation, layer, :],
            args.activation_alpha,
        )
        length_layer_rows.append(
            {
                "layer": layer,
                **{
                    f"development_{key}": value
                    for key, value in summarize(
                        length[development], predictions
                    ).items()
                },
                **{
                    f"confirmation_{key}": value
                    for key, value in summarize(
                        length[confirmation], confirmation_prediction
                    ).items()
                },
            }
        )
    length_layer_metrics = pd.DataFrame(length_layer_rows)
    length_layer_metrics.to_csv(
        args.output_dir / "length_decoding_by_layer.csv",
        index=False,
    )
    best_development = length_layer_metrics["development_spearman"].max()
    length_selected_layer = int(
        length_layer_metrics.loc[
            length_layer_metrics["development_spearman"].eq(best_development),
            "layer",
        ].min()
    )
    primary_layer_length_prediction = fit_predict_ridge(
        activations[development, args.activation_layer, :],
        length[development],
        activations[confirmation, args.activation_layer, :],
        args.activation_alpha,
    )
    length_selected_prediction = fit_predict_ridge(
        activations[development, length_selected_layer, :],
        length[development],
        activations[confirmation, length_selected_layer, :],
        args.activation_alpha,
    )
    length_decoding = {
        "note": (
            "Activations also predict observed response length, as expected "
            "given the strong length-engagement association. This does not "
            "establish a distinct planned-length representation; when two "
            "targets are this collinear, decoding one largely implies "
            "decoding the other."
        ),
        "development_selected_layer": length_selected_layer,
        "development_selected_layer_confirmation": (
            ladder_bootstrap_association(
                length[confirmation],
                length_selected_prediction,
                confirmation_ladders,
                args.bootstrap_repeats,
                rng,
            )
        ),
        "primary_engagement_layer": args.activation_layer,
        "primary_engagement_layer_confirmation": ladder_bootstrap_association(
            length[confirmation],
            primary_layer_length_prediction,
            confirmation_ladders,
            args.bootstrap_repeats,
            rng,
        ),
    }

    # 3. The frozen engagement probe's own predictions against length.
    engagement_prediction = fit_predict_ridge(
        activations[development, args.activation_layer, :],
        engagement[development],
        activations[confirmation, args.activation_layer, :],
        args.activation_alpha,
    )
    prediction_check: dict[str, object] = {"frozen_file_supplied": False}
    if args.confirmation_predictions is not None:
        frozen = pd.read_csv(args.confirmation_predictions)
        frozen = frozen.loc[
            frozen["target"].eq("substantive_engagement")
        ].set_index("prompt_id")
        confirmation_ids = scores.loc[confirmation, "prompt_id"]
        if set(confirmation_ids) - set(frozen.index):
            raise ValueError(
                "Frozen predictions lack some confirmation prompt IDs"
            )
        frozen_values = frozen.loc[
            confirmation_ids, "predicted_score"
        ].to_numpy(float)
        deviation = float(
            np.max(np.abs(frozen_values - engagement_prediction))
        )
        if deviation > args.prediction_tolerance:
            raise ValueError(
                "Recomputed engagement predictions differ from the frozen "
                f"file by {deviation:.3e}; the frozen layer or alpha may be "
                "misspecified"
            )
        prediction_check = {
            "frozen_file_supplied": True,
            "max_absolute_deviation": deviation,
        }
    probe_prediction_vs_length = ladder_bootstrap_association(
        engagement_prediction,
        length[confirmation],
        confirmation_ladders,
        args.bootstrap_repeats,
        rng,
    )

    # 4. Fixed-engagement subgroups. Where the rubric pins engagement but
    #    length still varies, a global length detector should still track
    #    length; the probe's predictions are the test.
    subgroup_rows: list[dict] = []
    confirmation_conditions = conditions[confirmation]
    confirmation_engagement = engagement[confirmation]
    confirmation_length = length[confirmation]
    for condition in sorted(set(confirmation_conditions)):
        mask = confirmation_conditions == condition
        subgroup_engagement = confirmation_engagement[mask]
        association = ladder_bootstrap_association(
            engagement_prediction[mask],
            confirmation_length[mask],
            confirmation_ladders[mask],
            args.bootstrap_repeats,
            rng,
        )
        subgroup_rows.append(
            {
                "design_condition": condition,
                "prompts": int(mask.sum()),
                "engagement_mean": float(np.mean(subgroup_engagement)),
                "engagement_sd": float(np.std(subgroup_engagement, ddof=1))
                if mask.sum() > 1
                else float("nan"),
                "engagement_min": float(np.min(subgroup_engagement)),
                "engagement_max": float(np.max(subgroup_engagement)),
                "engagement_fixed": bool(
                    np.std(subgroup_engagement) == 0
                ),
                "length_min": float(np.min(confirmation_length[mask])),
                "length_max": float(np.max(confirmation_length[mask])),
                "prediction_vs_length_spearman": association["spearman"],
                "prediction_vs_length_pearson": association["pearson"],
                "prediction_vs_length_spearman_low": (
                    association["spearman_95_percent_interval"][0]
                    if association["spearman_95_percent_interval"]
                    else float("nan")
                ),
                "prediction_vs_length_spearman_high": (
                    association["spearman_95_percent_interval"][1]
                    if association["spearman_95_percent_interval"]
                    else float("nan")
                ),
                "degenerate_bootstrap_samples": association[
                    "degenerate_bootstrap_samples"
                ],
            }
        )
    subgroup_frame = pd.DataFrame(subgroup_rows)
    subgroup_frame.to_csv(
        args.output_dir / "length_subgroup_diagnostics.csv",
        index=False,
    )

    # 5. Length-adjusted engagement decoding. Raw length is the principal
    #    adjustment because it has the stronger linear association; the
    #    development-quantile transform is a rank-based robustness check.
    raw_summary, raw_residual_truth, raw_residual_prediction = (
        residual_analysis(
            activations,
            engagement,
            length,
            development,
            confirmation,
            development_ladders,
            confirmation_ladders,
            args.activation_layer,
            args.activation_alpha,
            args.bootstrap_repeats,
            rng,
        )
    )
    raw_summary["confirmation_residual_condition_eta_squared"] = eta_squared(
        raw_residual_truth, confirmation_conditions
    )
    quantile_feature = development_quantile(length[development], length)
    rank_summary, rank_residual_truth, _ = residual_analysis(
        activations,
        engagement,
        quantile_feature,
        development,
        confirmation,
        development_ladders,
        confirmation_ladders,
        args.activation_layer,
        args.activation_alpha,
        args.bootstrap_repeats,
        rng,
    )
    rank_summary["confirmation_residual_condition_eta_squared"] = eta_squared(
        rank_residual_truth, confirmation_conditions
    )
    length_adjustment = {
        "note": (
            "These are sensitivity analyses, not length-corrected estimates "
            "of engagement decodability. If length is a nuisance the "
            "adjustment removes confounding; if length is a mediator of a "
            "genuine intention to engage, the same adjustment removes part "
            "of the construct. Observed length cannot distinguish the two."
        ),
        "linear_fit_quality": {
            "development_raw_pearson": float(
                summarize(length[development], engagement[development])[
                    "pearson"
                ]
            ),
            "development_log_pearson": float(
                summarize(np.log(length[development]), engagement[development])[
                    "pearson"
                ]
            ),
            "confirmation_raw_pearson": float(
                summarize(length[confirmation], engagement[confirmation])[
                    "pearson"
                ]
            ),
            "confirmation_log_pearson": float(
                summarize(
                    np.log(length[confirmation]), engagement[confirmation]
                )["pearson"]
            ),
            "note": (
                "Raw length is the better-fitting simple form here, so the "
                "raw adjustment is not a weak correction."
            ),
        },
        "raw_length_adjustment": raw_summary,
        "development_quantile_rank_adjustment": rank_summary,
    }

    # 6. Length is not merely a proxy for design condition.
    condition_values = sorted(set(conditions))
    condition_features = np.column_stack(
        [
            (conditions == value).astype(float)
            for value in condition_values[1:]
        ]
    )
    engagement_coefficients = fit_nuisance(
        condition_features[development], engagement[development]
    )
    length_coefficients = fit_nuisance(
        condition_features[development], length[development]
    )
    engagement_condition_residual = engagement[confirmation] - nuisance_predict(
        condition_features[confirmation], engagement_coefficients
    )
    length_condition_residual = length[confirmation] - nuisance_predict(
        condition_features[confirmation], length_coefficients
    )
    within_condition = {
        "note": (
            "Length carries information about engagement beyond the coarse "
            "design condition, so it cannot be dismissed as a condition "
            "proxy. This is the counterweight to the fixed-engagement "
            "subgroup result."
        ),
        "confirmation_internal_centering": {
            "note": (
                "Descriptive within-confirmation statistic using "
                "confirmation-set condition means."
            ),
            **ladder_bootstrap_association(
                center_within_group(
                    length[confirmation], confirmation_conditions
                ),
                center_within_group(
                    engagement[confirmation], confirmation_conditions
                ),
                confirmation_ladders,
                args.bootstrap_repeats,
                rng,
            ),
        },
        "development_estimated_condition_effects": {
            "note": (
                "Condition effects for both variables are fit on development "
                "only, preserving the held-out logic."
            ),
            **ladder_bootstrap_association(
                length_condition_residual,
                engagement_condition_residual,
                confirmation_ladders,
                args.bootstrap_repeats,
                rng,
            ),
        },
        "confirmation_condition_eta_squared_length": eta_squared(
            length[confirmation], confirmation_conditions
        ),
        "confirmation_condition_eta_squared_engagement": eta_squared(
            engagement[confirmation], confirmation_conditions
        ),
    }

    per_prompt = pd.DataFrame(
        {
            "prompt_id": scores.loc[confirmation, "prompt_id"].to_numpy(),
            "ladder_id": confirmation_ladders,
            "design_condition": confirmation_conditions,
            "substantive_engagement": confirmation_engagement,
            "response_length_tokens": confirmation_length,
            "engagement_prediction": engagement_prediction,
            "length_prediction_primary_layer": primary_layer_length_prediction,
            "raw_length_residual_truth": raw_residual_truth,
            "raw_length_residual_prediction": raw_residual_prediction,
        }
    )
    per_prompt.to_csv(
        args.output_dir / "length_confirmation_values.csv",
        index=False,
    )

    summary = {
        "status": "post_hoc_response_length_diagnostics",
        "does_not_replace_frozen_analysis": True,
        "frozen_prediction_reproduction_check": prediction_check,
        "activation_layer": args.activation_layer,
        "activation_alpha": args.activation_alpha,
        "bootstrap_repeats": args.bootstrap_repeats,
        "confirmation_prompts": int(confirmation.sum()),
        "confirmation_ladders": int(len(np.unique(confirmation_ladders))),
        "interval_caveat": (
            "Ladder bootstrap covers evaluation-ladder resampling only, with "
            "ten confirmation ladders. It does not cover model fitting or "
            "layer selection, and the subgroup intervals are wide."
        ),
        "observed_length_engagement_association": observed_association,
        "activation_length_decoding": length_decoding,
        "probe_prediction_versus_length": probe_prediction_vs_length,
        "fixed_engagement_subgroups": {
            "note": (
                "A falsification check, not a matched-length experiment: "
                "engagement is held constant by the rubric while length "
                "varies, rather than length being held constant. It cannot "
                "rule out nonlinear or condition-specific length dependence."
            ),
            "subgroups": subgroup_frame.to_dict("records"),
        },
        "length_adjusted_engagement_decoding": length_adjustment,
        "within_condition_length_engagement": within_condition,
    }
    (args.output_dir / "round2_length_diagnostics.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
