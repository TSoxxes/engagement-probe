"""Per-layer ridge probes with leave-one-ladder-out evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


SCORE_COLUMNS = [
    "substantive_engagement",
    "direct_refusal",
    "safety_caution",
    "epistemic_uncertainty",
    "professional_redirection",
    "prompt_harmfulness",
]
KEY_COLUMNS = ["prompt_id", "generation_id"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--activations", type=Path, required=True)
    parser.add_argument(
        "--run-config",
        type=Path,
        help="Generation run_config.json, used to recover token-cap flags from older runs.",
    )
    parser.add_argument("--judge-files", type=Path, nargs="+", required=True)
    parser.add_argument("--manual-scores", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--shuffle-repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260724)
    return parser.parse_args()


def read_jsonl(path: Path) -> pd.DataFrame:
    return pd.read_json(path, lines=True)


def attach_token_cap_flags(
    responses: pd.DataFrame,
    run_config: Path | None,
) -> tuple[pd.DataFrame, int]:
    """Validate or reconstruct the per-response token-cap indicator."""
    if "response_length_tokens" not in responses.columns:
        raise ValueError("Responses lack response_length_tokens")

    max_new_tokens: int | None = None
    if run_config is not None:
        config = json.loads(run_config.read_text(encoding="utf-8"))
        configured_max = config.get("max_new_tokens")
        if not isinstance(configured_max, int) or configured_max < 1:
            raise ValueError(f"{run_config} lacks a valid max_new_tokens value")
        max_new_tokens = configured_max

    responses = responses.copy()
    if "hit_token_cap" in responses.columns:
        values = responses["hit_token_cap"]
        if values.isna().any() or not values.isin([True, False, 0, 1]).all():
            raise ValueError("hit_token_cap must contain only boolean values")
        responses["hit_token_cap"] = values.astype(bool)
        if max_new_tokens is not None:
            lengths = responses["response_length_tokens"]
            impossible = (
                (responses["hit_token_cap"] & (lengths < max_new_tokens))
                | (lengths > max_new_tokens)
            )
            if impossible.any():
                raise ValueError(
                    "hit_token_cap values disagree with response lengths and run config"
                )
    elif max_new_tokens is not None:
        responses["hit_token_cap"] = (
            responses["response_length_tokens"] >= max_new_tokens
        )
    else:
        raise ValueError(
            "Responses lack hit_token_cap; supply --run-config to reconstruct it"
        )

    if max_new_tokens is None:
        capped_lengths = responses.loc[
            responses["hit_token_cap"], "response_length_tokens"
        ]
        max_new_tokens = (
            int(capped_lengths.min()) if not capped_lengths.empty else 0
        )
    return responses, max_new_tokens


def validate_scores(frame: pd.DataFrame, source: Path) -> None:
    missing = set(KEY_COLUMNS + SCORE_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"{source} lacks columns: {sorted(missing)}")
    for column in SCORE_COLUMNS:
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if numeric.isna().any() or ((numeric < 0) | (numeric > 3)).any():
            raise ValueError(f"{source}: {column} must contain only scores from 0 to 3")
    if frame.duplicated(KEY_COLUMNS).any():
        raise ValueError(f"{source} contains duplicate prompt/generation keys")


def combine_judges(
    judge_files: list[Path],
    manual_scores: Path | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    for judge_number, path in enumerate(judge_files, start=1):
        frame = read_jsonl(path)
        validate_scores(frame, path)
        frame = frame[KEY_COLUMNS + SCORE_COLUMNS].copy()
        frame["judge"] = judge_number
        frames.append(frame)
    long = pd.concat(frames, ignore_index=True)

    grouped = long.groupby(KEY_COLUMNS, sort=False)
    means = grouped[SCORE_COLUMNS].mean().reset_index()
    spreads = grouped[SCORE_COLUMNS].agg(lambda values: values.max() - values.min())
    disagreement_mask = (spreads > 1).any(axis=1)
    disagreement_keys = spreads.loc[disagreement_mask].reset_index()[KEY_COLUMNS]
    disagreements = long.merge(disagreement_keys, on=KEY_COLUMNS, how="inner")

    if manual_scores is not None:
        manual = read_jsonl(manual_scores)
        missing_keys = set(KEY_COLUMNS) - set(manual.columns)
        if missing_keys:
            raise ValueError(f"{manual_scores} lacks columns: {sorted(missing_keys)}")
        available_scores = [column for column in SCORE_COLUMNS if column in manual.columns]
        if not available_scores:
            raise ValueError(f"{manual_scores} contains no score columns")
        indexed = means.set_index(KEY_COLUMNS)
        for _, row in manual.iterrows():
            key = (row["prompt_id"], row["generation_id"])
            if key not in indexed.index:
                raise ValueError(f"Manual score key does not exist in judge files: {key}")
            for column in available_scores:
                if pd.notna(row[column]):
                    value = float(row[column])
                    if not 0 <= value <= 3:
                        raise ValueError(f"Manual {column} score outside 0–3 for {key}")
                    indexed.loc[key, column] = value
        means = indexed.reset_index()
    return means, disagreements


def pearson_correlation(truth: np.ndarray, predicted: np.ndarray) -> float:
    if np.std(truth) == 0 or np.std(predicted) == 0:
        return 0.0
    value = np.corrcoef(truth, predicted)[0, 1]
    return float(0.0 if np.isnan(value) else value)


def spearman_correlation(truth: np.ndarray, predicted: np.ndarray) -> float:
    truth_ranks = pd.Series(truth).rank(method="average").to_numpy()
    predicted_ranks = pd.Series(predicted).rank(method="average").to_numpy()
    return pearson_correlation(truth_ranks, predicted_ranks)


def summarize(truth: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "spearman": spearman_correlation(truth, predicted),
        "pearson": pearson_correlation(truth, predicted),
        "mae": float(np.mean(np.abs(truth - predicted))),
    }


def fit_predict_ridge(
    train_features: np.ndarray,
    train_target: np.ndarray,
    test_features: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Standardized ridge using the small-sample dual solution."""
    feature_mean = train_features.mean(axis=0)
    feature_scale = train_features.std(axis=0)
    feature_scale[feature_scale == 0] = 1.0
    train_standard = (train_features - feature_mean) / feature_scale
    test_standard = (test_features - feature_mean) / feature_scale
    target_mean = train_target.mean()
    centered_target = train_target - target_mean

    # For this pilot there are far fewer prompts than activation dimensions.
    # The dual form solves an n_train × n_train system instead of a
    # hidden_size × hidden_size system.
    kernel = train_standard @ train_standard.T
    dual_weights = np.linalg.solve(
        kernel + alpha * np.eye(kernel.shape[0]),
        centered_target,
    )
    primal_weights = train_standard.T @ dual_weights
    return test_standard @ primal_weights + target_mean


def lolo_predictions(
    features: np.ndarray,
    target: np.ndarray,
    ladders: np.ndarray,
    alpha: float,
    shuffled: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    predictions = np.full(target.shape, np.nan, dtype=float)
    for held_out in np.unique(ladders):
        train = ladders != held_out
        test = ~train
        train_target = target[train].copy()
        if shuffled:
            train_target = rng.permutation(train_target)
        predictions[test] = fit_predict_ridge(
            features[train],
            train_target,
            features[test],
            alpha,
        )
    if np.isnan(predictions).any():
        raise RuntimeError("Some prompts did not receive held-out predictions")
    return predictions


def main() -> None:
    args = parse_args()
    if args.shuffle_repeats < 1:
        raise ValueError("--shuffle-repeats must be at least 1")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = read_jsonl(args.dataset)
    responses = read_jsonl(args.responses)
    responses, max_new_tokens = attach_token_cap_flags(responses, args.run_config)
    scores, disagreements = combine_judges(args.judge_files, args.manual_scores)
    disagreements.to_csv(args.output_dir / "judge_disagreements.csv", index=False)

    expected = responses[KEY_COLUMNS].drop_duplicates()
    scored = expected.merge(scores[KEY_COLUMNS], on=KEY_COLUMNS, how="left", indicator=True)
    missing_scores = scored.loc[scored["_merge"] == "left_only", KEY_COLUMNS]
    if not missing_scores.empty:
        examples = missing_scores.head(5).to_dict("records")
        raise ValueError(f"{len(missing_scores)} response(s) lack judge scores; examples: {examples}")

    response_scores = responses.merge(scores, on=KEY_COLUMNS, how="inner", validate="one_to_one")
    prompt_scores = (
        response_scores.groupby("prompt_id", as_index=False)
        .agg(
            **{column: (column, "mean") for column in SCORE_COLUMNS},
            response_length_tokens=("response_length_tokens", "mean"),
            capped_generations=("hit_token_cap", "sum"),
            hit_token_cap_rate=("hit_token_cap", "mean"),
            num_generations=("generation_id", "nunique"),
        )
        .merge(
            dataset[["prompt_id", "ladder_id", "category", "rung"]],
            on="prompt_id",
            how="left",
            validate="one_to_one",
        )
    )
    if prompt_scores["ladder_id"].isna().any():
        raise ValueError("Some response prompt IDs are missing from the dataset")

    archive = np.load(args.activations)
    activations = archive["activations"].astype(np.float32)
    activation_prompt_ids = archive["prompt_ids"].astype(str)
    if activations.ndim != 3:
        raise ValueError("activations must have shape [prompts, layers, hidden_size]")
    if len(activation_prompt_ids) != activations.shape[0]:
        raise ValueError("Activation prompt IDs do not match activation rows")
    activation_index = {prompt_id: i for i, prompt_id in enumerate(activation_prompt_ids)}
    missing_activations = set(prompt_scores["prompt_id"]) - set(activation_index)
    if missing_activations:
        raise ValueError(f"Missing activations for: {sorted(missing_activations)}")
    order = np.asarray([activation_index[prompt_id] for prompt_id in prompt_scores["prompt_id"]])
    activations = activations[order]

    prompt_scores.to_csv(args.output_dir / "prompt_level_scores.csv", index=False)
    target = prompt_scores["substantive_engagement"].to_numpy(dtype=float)
    ladders = prompt_scores["ladder_id"].to_numpy()
    length_features = prompt_scores[["response_length_tokens"]].to_numpy(dtype=float)
    cap_features = prompt_scores[["hit_token_cap_rate"]].to_numpy(dtype=float)
    rng = np.random.default_rng(args.seed)

    length_predictions = lolo_predictions(
        length_features, target, ladders, args.ridge_alpha, False, rng
    )
    length_metrics = summarize(target, length_predictions)
    cap_predictions = lolo_predictions(
        cap_features, target, ladders, args.ridge_alpha, False, rng
    )
    cap_metrics = summarize(target, cap_predictions)
    prediction_frames = []
    metric_rows = []

    for layer in range(activations.shape[1]):
        features = activations[:, layer, :]
        real_predictions = lolo_predictions(
            features, target, ladders, args.ridge_alpha, False, rng
        )
        real_metrics = summarize(target, real_predictions)
        shuffled_spearman = []
        shuffled_pearson = []
        shuffled_mae = []
        for _ in range(args.shuffle_repeats):
            shuffled_predictions = lolo_predictions(
                features, target, ladders, args.ridge_alpha, True, rng
            )
            metrics = summarize(target, shuffled_predictions)
            shuffled_spearman.append(metrics["spearman"])
            shuffled_pearson.append(metrics["pearson"])
            shuffled_mae.append(metrics["mae"])

        metric_rows.append(
            {
                "layer": layer,
                "activation_spearman": real_metrics["spearman"],
                "activation_pearson": real_metrics["pearson"],
                "activation_mae": real_metrics["mae"],
                "length_spearman": length_metrics["spearman"],
                "length_pearson": length_metrics["pearson"],
                "length_mae": length_metrics["mae"],
                "cap_spearman": cap_metrics["spearman"],
                "cap_pearson": cap_metrics["pearson"],
                "cap_mae": cap_metrics["mae"],
                "shuffled_spearman_mean": float(np.mean(shuffled_spearman)),
                "shuffled_spearman_std": float(np.std(shuffled_spearman)),
                "shuffled_pearson_mean": float(np.mean(shuffled_pearson)),
                "shuffled_mae_mean": float(np.mean(shuffled_mae)),
            }
        )
        prediction_frames.append(
            pd.DataFrame(
                {
                    "prompt_id": prompt_scores["prompt_id"],
                    "ladder_id": ladders,
                    "layer": layer,
                    "true_engagement": target,
                    "activation_prediction": real_predictions,
                    "length_prediction": length_predictions,
                    "hit_token_cap_rate": prompt_scores["hit_token_cap_rate"],
                    "cap_prediction": cap_predictions,
                }
            )
        )

    metrics_frame = pd.DataFrame(metric_rows)
    predictions_frame = pd.concat(prediction_frames, ignore_index=True)
    metrics_frame.to_csv(args.output_dir / "metrics_by_layer.csv", index=False)
    predictions_frame.to_csv(args.output_dir / "predictions.csv", index=False)

    best = metrics_frame.loc[metrics_frame["activation_spearman"].idxmax()].to_dict()
    summary = {
        "selection_metric": "leave-one-ladder-out Spearman correlation",
        "best_layer": int(best["layer"]),
        "best_layer_metrics": {
            key: float(value) for key, value in best.items() if key != "layer"
        },
        "num_prompts": int(len(prompt_scores)),
        "num_ladders": int(prompt_scores["ladder_id"].nunique()),
        "activation_shape": list(activations.shape),
        "ridge_alpha": args.ridge_alpha,
        "shuffle_repeats": args.shuffle_repeats,
        "max_new_tokens": max_new_tokens,
        "capped_responses": int(responses["hit_token_cap"].sum()),
        "capped_response_rate": float(responses["hit_token_cap"].mean()),
        "cap_baseline_metrics": cap_metrics,
        "caution": "Feasibility result only; layer selection and evaluation use the same small dataset.",
    }
    with (args.output_dir / "best_layer_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    category_diagnostics = (
        responses[["prompt_id", "generation_id", "hit_token_cap"]].merge(
            dataset[["prompt_id", "category"]],
            on="prompt_id",
            how="left",
            validate="many_to_one",
        )
        .groupby("category", as_index=False)
        .agg(
            responses=("generation_id", "size"),
            capped_responses=("hit_token_cap", "sum"),
            capped_response_rate=("hit_token_cap", "mean"),
        )
    )
    diagnostics = {
        "max_new_tokens": max_new_tokens,
        "responses": int(len(responses)),
        "capped_responses": int(responses["hit_token_cap"].sum()),
        "capped_response_rate": float(responses["hit_token_cap"].mean()),
        "by_category": category_diagnostics.to_dict("records"),
        "by_prompt": prompt_scores[
            [
                "prompt_id",
                "ladder_id",
                "category",
                "rung",
                "num_generations",
                "capped_generations",
                "hit_token_cap_rate",
            ]
        ].to_dict("records"),
    }
    with (args.output_dir / "generation_diagnostics.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(diagnostics, handle, indent=2)
        handle.write("\n")

    try:
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(figsize=(8, 4.8))
        axis.plot(
            metrics_frame["layer"],
            metrics_frame["activation_spearman"],
            marker="o",
            markersize=3,
            label="Activation ridge",
        )
        axis.plot(
            metrics_frame["layer"],
            metrics_frame["shuffled_spearman_mean"],
            linestyle="--",
            label="Shuffled labels",
        )
        axis.axhline(
            length_metrics["spearman"],
            color="tab:green",
            linestyle=":",
            label="Response length",
        )
        axis.axhline(
            cap_metrics["spearman"],
            color="tab:red",
            linestyle="-.",
            label="Token-cap rate",
        )
        axis.axhline(0, color="black", linewidth=0.7)
        axis.set(
            xlabel="Hidden-state index (0 = embedding output)",
            ylabel="Held-out Spearman correlation",
            title="Substantive engagement prediction by layer",
        )
        axis.legend()
        figure.tight_layout()
        figure.savefig(args.output_dir / "probe_curve.png", dpi=180)
        plt.close(figure)
    except ModuleNotFoundError:
        print("matplotlib is unavailable; CSV results were saved but the plot was skipped.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
