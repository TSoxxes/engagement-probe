"""Pre-generation leakage and design audit for the Round 2 prompt set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from text_features import (
    lolo_text_classification,
    ridge_predict,
    surface_features,
    tfidf_features,
)


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/round2_ladders.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/round2_prompt_audit.json"),
    )
    parser.add_argument("--max-cell-accuracy", type=float, default=0.45)
    parser.add_argument("--max-surface-subtype-accuracy", type=float, default=0.75)
    parser.add_argument("--max-split-cosine", type=float, default=0.55)
    parser.add_argument(
        "--enforce-diagnostic-thresholds",
        action="store_true",
        help="Treat the text-decodability diagnostics as blocking gates.",
    )
    parser.add_argument("--allow-failures", action="store_true")
    return parser.parse_args()


def classification_from_features(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
) -> float:
    classes = np.unique(labels)
    predictions = np.empty(len(labels), dtype=object)
    for held_out in np.unique(groups):
        train = groups != held_out
        test = ~train
        one_hot = np.column_stack(
            [(labels[train] == value).astype(float) for value in classes]
        )
        scores = ridge_predict(
            features[train],
            one_hot,
            features[test],
            alpha=1.0,
        )
        predictions[test] = classes[np.argmax(scores, axis=1)]
    return float(np.mean(predictions == labels))


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.dataset)
    prompts = np.asarray([row["prompt"] for row in rows], dtype=object)
    cells = np.asarray([row["design_cell"] for row in rows], dtype=str)
    ladders = np.asarray([row["ladder_id"] for row in rows], dtype=str)
    rungs = np.asarray([row["rung"] for row in rows], dtype=int)

    _, cell_accuracy = lolo_text_classification(
        prompts.tolist(),
        cells,
        ladders,
    )
    rung_majority_accuracy = max(
        np.mean(cells[rungs == rung] == cell)
        for rung in np.unique(rungs)
        for cell in np.unique(cells)
    )

    refusal = cells == "D"
    refusal_prompts = prompts[refusal]
    refusal_labels = np.asarray(
        [row["intended_refusal_basis"] for row in rows if row["design_cell"] == "D"],
        dtype=str,
    )
    refusal_ladders = ladders[refusal]
    _, full_subtype_accuracy = lolo_text_classification(
        refusal_prompts.tolist(),
        refusal_labels,
        refusal_ladders,
    )
    subtype_surface_accuracy = classification_from_features(
        surface_features(refusal_prompts.tolist()),
        refusal_labels,
        refusal_ladders,
    )

    development = np.asarray(
        [row["split"] == "development" for row in rows],
        dtype=bool,
    )
    confirmation = ~development
    development_features, confirmation_features = tfidf_features(
        prompts[development].tolist(),
        prompts[confirmation].tolist(),
        min_document_frequency=1,
    )
    similarities = confirmation_features @ development_features.T
    development_cells = cells[development]
    confirmation_cells = cells[confirmation]
    same_condition_maxima = []
    nearest_pairs = []
    development_rows = [row for row in rows if row["split"] == "development"]
    confirmation_rows = [row for row in rows if row["split"] == "confirmation"]
    for confirmation_index, confirmation_cell in enumerate(confirmation_cells):
        eligible = np.flatnonzero(development_cells == confirmation_cell)
        local_index = int(np.argmax(similarities[confirmation_index, eligible]))
        development_index = int(eligible[local_index])
        score = float(similarities[confirmation_index, development_index])
        same_condition_maxima.append(score)
        nearest_pairs.append(
            {
                "confirmation_prompt_id": confirmation_rows[confirmation_index][
                    "prompt_id"
                ],
                "development_prompt_id": development_rows[development_index][
                    "prompt_id"
                ],
                "design_cell": str(confirmation_cell),
                "cosine": score,
            }
        )

    scripted_b = sum(
        row["design_cell"] == "B"
        and (
            row["prompt"].lower().startswith("without ")
            or row["prompt"].lower().startswith("i know you cannot")
        )
        for row in rows
    )
    blocking_failures = []
    diagnostic_warnings = []
    maximum_similarity = max(same_condition_maxima)
    if cell_accuracy > args.max_cell_accuracy:
        diagnostic_warnings.append(
            f"text-only cell accuracy {cell_accuracy:.3f} exceeds "
            f"{args.max_cell_accuracy:.3f}"
        )
    if subtype_surface_accuracy > args.max_surface_subtype_accuracy:
        diagnostic_warnings.append(
            f"surface-only D-subtype accuracy {subtype_surface_accuracy:.3f} "
            f"exceeds {args.max_surface_subtype_accuracy:.3f}"
        )
    if maximum_similarity > args.max_split_cosine:
        blocking_failures.append(
            f"maximum same-cell split cosine {maximum_similarity:.3f} exceeds "
            f"{args.max_split_cosine:.3f}"
        )
    if scripted_b:
        blocking_failures.append(
            f"{scripted_b} B prompts retain a scripted disclaimer"
        )
    if rung_majority_accuracy > 0.25:
        blocking_failures.append(
            f"rung predicts cell above chance ({rung_majority_accuracy:.3f})"
        )
    injected_markers = {
        phrase: sum(phrase.lower() in prompt.lower() for prompt in prompts)
        for phrase in (
            "Give a direct response to this question:",
            "I know you cannot",
        )
    }
    for phrase, count in injected_markers.items():
        if count:
            blocking_failures.append(
                f"{count} prompts contain injected marker {phrase!r}"
            )
    if args.enforce_diagnostic_thresholds:
        blocking_failures.extend(diagnostic_warnings)

    report = {
        "dataset": str(args.dataset),
        "prompts": len(rows),
        "ladders": len(set(ladders)),
        "text_only_lolo_design_cell_accuracy": cell_accuracy,
        "chance_design_cell_accuracy": 0.25,
        "rung_best_cell_accuracy": float(rung_majority_accuracy),
        "d_subtype_full_text_lolo_accuracy": full_subtype_accuracy,
        "d_subtype_surface_only_lolo_accuracy": subtype_surface_accuracy,
        "scripted_b_disclaimer_count": scripted_b,
        "injected_marker_counts": injected_markers,
        "maximum_same_cell_development_confirmation_cosine": maximum_similarity,
        "mean_same_cell_development_confirmation_cosine": float(
            np.mean(same_condition_maxima)
        ),
        "nearest_confirmation_pairs": sorted(
            nearest_pairs,
            key=lambda row: row["cosine"],
            reverse=True,
        ),
        "thresholds": {
            "max_cell_accuracy": args.max_cell_accuracy,
            "max_surface_subtype_accuracy": args.max_surface_subtype_accuracy,
            "max_split_cosine": args.max_split_cosine,
        },
        "passed_pre_generation_structure": not blocking_failures,
        "blocking_failures": blocking_failures,
        "diagnostic_warnings": diagnostic_warnings,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    if blocking_failures and not args.allow_failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
