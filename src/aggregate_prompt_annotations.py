"""Aggregate blinded prompt-only harmfulness annotations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from aggregate_judging import read_json_records, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--raw-scores-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--judges", type=int, default=3)
    parser.add_argument("--manual-scores", type=Path)
    return parser.parse_args()


def read_manual(path: Path | None) -> dict[str, float]:
    if path is None:
        return {}
    overrides: dict[str, float] = {}
    for row_number, row in enumerate(read_json_records(path), start=1):
        if {"prompt_id", "prompt_harmfulness"} - row.keys():
            raise ValueError(f"{path}:{row_number} lacks required fields")
        value = row["prompt_harmfulness"]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{path}:{row_number} score must be numeric")
        if not 0 <= float(value) <= 3:
            raise ValueError(f"{path}:{row_number} score is outside 0 to 3")
        prompt_id = str(row["prompt_id"])
        if prompt_id in overrides:
            raise ValueError(f"Duplicate manual prompt score: {prompt_id}")
        overrides[prompt_id] = float(value)
    return overrides


def main() -> None:
    args = parse_args()
    if args.judges < 2:
        raise ValueError("--judges must be at least 2")
    mapping = read_json_records(args.mapping)
    by_case: dict[int, dict[str, str]] = {
        judge: {} for judge in range(1, args.judges + 1)
    }
    prompt_order: list[str] = []
    for row in mapping:
        judge = int(row["judge"])
        if judge not in by_case:
            raise ValueError(f"Unexpected judge {judge}")
        by_case[judge][str(row["case_id"])] = str(row["prompt_id"])
        if judge == 1:
            prompt_order.append(str(row["prompt_id"]))
    reference = set(by_case[1].values())
    if any(set(cases.values()) != reference for cases in by_case.values()):
        raise ValueError("Judges do not cover the same prompt set")

    scores: dict[int, dict[str, int]] = {}
    for judge in range(1, args.judges + 1):
        path = args.raw_scores_dir / f"judge_{judge}.jsonl"
        rows = read_json_records(path)
        indexed: dict[str, int] = {}
        for row_number, row in enumerate(rows, start=1):
            if {"case_id", "prompt_harmfulness"} - row.keys():
                raise ValueError(f"{path}:{row_number} lacks required fields")
            case = str(row["case_id"])
            if case not in by_case[judge]:
                raise ValueError(f"{path}:{row_number} contains unknown case {case}")
            value = row["prompt_harmfulness"]
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 3:
                raise ValueError(f"{path}:{row_number} score must be an integer 0 to 3")
            prompt_id = by_case[judge][case]
            if prompt_id in indexed:
                raise ValueError(f"{path} duplicates prompt {prompt_id}")
            indexed[prompt_id] = value
        if set(indexed) != reference:
            raise ValueError(f"{path} does not cover every prompt")
        scores[judge] = indexed

    overrides = read_manual(args.manual_scores)
    output: list[dict] = []
    disagreements: list[dict] = []
    template: list[dict] = []
    for prompt_id in prompt_order:
        values = [scores[judge][prompt_id] for judge in range(1, args.judges + 1)]
        spread = max(values) - min(values)
        automatic = sum(values) / len(values)
        final = overrides.get(prompt_id, automatic)
        output.append(
            {
                "prompt_id": prompt_id,
                "prompt_harmfulness": round(final, 6),
                "automatic_mean": round(automatic, 6),
                "manual_override": prompt_id in overrides,
            }
        )
        if spread > 1:
            row = {"prompt_id": prompt_id}
            row.update(
                {f"judge_{judge}": value for judge, value in enumerate(values, start=1)}
            )
            row["difference"] = spread
            row["resolved"] = prompt_id in overrides
            disagreements.append(row)
            if prompt_id not in overrides:
                template.append(
                    {"prompt_id": prompt_id, "prompt_harmfulness": None}
                )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "prompt_scores.jsonl", output)
    fields = [
        "prompt_id",
        *(f"judge_{judge}" for judge in range(1, args.judges + 1)),
        "difference",
        "resolved",
    ]
    with (args.output_dir / "disagreements.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(disagreements)
    if template:
        write_jsonl(args.output_dir / "manual_scores.template.jsonl", template)
    summary = {
        "prompts": len(prompt_order),
        "judges": args.judges,
        "flagged_prompts": len(disagreements),
        "unresolved_prompts": sum(not row["resolved"] for row in disagreements),
        "manual_overrides": len(overrides),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
