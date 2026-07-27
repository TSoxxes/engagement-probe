"""Validate repeated blind-judge passes and aggregate them for probe analysis."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from scoring_schema import ROUND1_SCORE_COLUMNS, get_schema

SCORE_COLUMNS = ROUND1_SCORE_COLUMNS


def strip_markdown_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if len(lines) < 3 or not lines[-1].strip().startswith("```"):
        raise ValueError("Score file has an unterminated Markdown fence")
    return "\n".join(lines[1:-1]).strip()


def read_json_records(path: Path) -> list[dict]:
    """Accept JSONL, a JSON array, or one fenced JSONL block."""
    text = strip_markdown_fence(path.read_text(encoding="utf-8"))
    if not text:
        raise ValueError(f"{path} is empty")

    try:
        whole = json.loads(text)
    except json.JSONDecodeError:
        rows: list[dict] = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            candidate = line
            if line_number == 1 and not line.lstrip().startswith("{") and "{" in line:
                candidate = line[line.find("{") :]
            try:
                row = json.loads(candidate)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on {path}:{line_number}") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(row)
        return rows

    if isinstance(whole, list):
        if not all(isinstance(row, dict) for row in whole):
            raise ValueError(f"{path} JSON array contains a non-object value")
        return whole
    if isinstance(whole, dict):
        return [whole]
    raise ValueError(f"{path} must contain JSON objects")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def validate_raw_scores(
    rows: list[dict],
    expected_case_ids: set[str],
    source: Path,
    score_columns: list[str] | None = None,
    categorical_columns: dict[str, set[str]] | None = None,
) -> dict[str, dict]:
    score_columns = score_columns or SCORE_COLUMNS
    categorical_columns = categorical_columns or {}
    indexed: dict[str, dict] = {}
    for row_number, row in enumerate(rows, start=1):
        missing = {"case_id", *score_columns, *categorical_columns} - row.keys()
        if missing:
            raise ValueError(f"{source}:{row_number} lacks {sorted(missing)}")
        case_id = str(row["case_id"])
        if case_id in indexed:
            raise ValueError(f"{source} contains duplicate case_id {case_id}")
        for column in score_columns:
            value = row[column]
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 3:
                raise ValueError(
                    f"{source}:{row_number} {column} must be an integer from 0 to 3"
                )
        reason = row.get("brief_reason")
        if reason is not None and not isinstance(reason, str):
            raise ValueError(f"{source}:{row_number} brief_reason must be text")
        for column, choices in categorical_columns.items():
            if row[column] not in choices:
                raise ValueError(
                    f"{source}:{row_number} {column} must be one of {sorted(choices)}"
                )
        indexed[case_id] = row

    supplied = set(indexed)
    missing_cases = expected_case_ids - supplied
    extra_cases = supplied - expected_case_ids
    if missing_cases or extra_cases:
        raise ValueError(
            f"{source} case IDs do not match its packet: "
            f"{len(missing_cases)} missing, {len(extra_cases)} unknown"
        )
    return indexed


def read_mapping(path: Path, judges: int) -> tuple[
    dict[int, dict[str, tuple[str, int]]],
    dict[int, dict[tuple[str, int], str]],
    list[tuple[str, int]],
]:
    rows = read_json_records(path)
    by_case: dict[int, dict[str, tuple[str, int]]] = {
        judge: {} for judge in range(1, judges + 1)
    }
    by_key: dict[int, dict[tuple[str, int], str]] = {
        judge: {} for judge in range(1, judges + 1)
    }
    key_order: list[tuple[str, int]] = []
    seen_order: set[tuple[str, int]] = set()

    for row_number, row in enumerate(rows, start=1):
        missing = {"judge", "case_id", "prompt_id", "generation_id"} - row.keys()
        if missing:
            raise ValueError(f"{path}:{row_number} lacks {sorted(missing)}")
        judge = int(row["judge"])
        if judge not in by_case:
            raise ValueError(f"{path}:{row_number} has unexpected judge {judge}")
        case_id = str(row["case_id"])
        key = (str(row["prompt_id"]), int(row["generation_id"]))
        if case_id in by_case[judge]:
            raise ValueError(f"Duplicate mapping for judge {judge}, {case_id}")
        if key in by_key[judge]:
            raise ValueError(f"Duplicate mapping for judge {judge}, {key}")
        by_case[judge][case_id] = key
        by_key[judge][key] = case_id
        if judge == 1 and key not in seen_order:
            seen_order.add(key)
            key_order.append(key)

    reference_keys = set(by_key[1])
    for judge in range(1, judges + 1):
        if set(by_key[judge]) != reference_keys:
            raise ValueError(f"Judge {judge} mapping covers a different response set")
    return by_case, by_key, key_order


def read_within_manual_scores(
    path: Path | None,
    score_columns: list[str] | None = None,
) -> dict[tuple[int, str, int, str], float]:
    score_columns = score_columns or SCORE_COLUMNS
    if path is None:
        return {}
    rows = read_json_records(path)
    overrides: dict[tuple[int, str, int, str], float] = {}
    for row_number, row in enumerate(rows, start=1):
        missing = {"judge", "prompt_id", "generation_id"} - row.keys()
        if missing:
            raise ValueError(f"{path}:{row_number} lacks {sorted(missing)}")
        judge = int(row["judge"])
        prompt_id = str(row["prompt_id"])
        generation_id = int(row["generation_id"])
        available = False
        for column in score_columns:
            if column not in row or row[column] is None:
                continue
            available = True
            value = row[column]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{path}:{row_number} {column} must be numeric")
            if not 0 <= float(value) <= 3:
                raise ValueError(f"{path}:{row_number} {column} is outside 0 to 3")
            key = (judge, prompt_id, generation_id, column)
            if key in overrides:
                raise ValueError(f"Duplicate manual override: {key}")
            overrides[key] = float(value)
        if not available:
            raise ValueError(f"{path}:{row_number} contains no manual scores")
    return overrides


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--raw-scores-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--judges", type=int, default=2)
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument("--schema", choices=("round1", "round2"), default="round1")
    parser.add_argument("--within-manual-scores", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.judges < 2:
        raise ValueError("--judges must be at least 2")
    if args.passes < 1:
        raise ValueError("--passes must be at least 1")
    schema = get_schema(args.schema)
    score_columns = schema["score_columns"]
    categorical_columns = schema["categorical_columns"]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    by_case, by_key, key_order = read_mapping(args.mapping, args.judges)
    manual_overrides = read_within_manual_scores(
        args.within_manual_scores,
        score_columns,
    )
    pass_scores: dict[tuple[int, int, str], dict] = {}

    for judge in range(1, args.judges + 1):
        expected_case_ids = set(by_case[judge])
        for pass_number in range(1, args.passes + 1):
            unsplit_source = (
                args.raw_scores_dir
                / f"judge_{judge}_pass_{pass_number}.jsonl"
            )
            if unsplit_source.exists():
                sources = [unsplit_source]
            else:
                sources = sorted(
                    args.raw_scores_dir.glob(
                        f"judge_{judge}_pass_{pass_number}_batch_*.jsonl"
                    )
                )
            if not sources:
                raise FileNotFoundError(
                    f"Missing raw score file or batches for judge {judge}, "
                    f"pass {pass_number}"
                )
            rows = [
                row
                for source in sources
                for row in read_json_records(source)
            ]
            source = unsplit_source if len(sources) == 1 else args.raw_scores_dir
            indexed = validate_raw_scores(
                rows,
                expected_case_ids,
                source,
                score_columns,
                categorical_columns,
            )
            for case_id, row in indexed.items():
                pass_scores[(judge, pass_number, case_id)] = row

    aggregated_by_judge: dict[int, dict[tuple[str, int], dict]] = defaultdict(dict)
    within_disagreements: list[dict] = []
    within_template_by_case: dict[tuple[int, str, int], dict] = {}

    for judge in range(1, args.judges + 1):
        for prompt_id, generation_id in key_order:
            case_id = by_key[judge][(prompt_id, generation_id)]
            output_row: dict[str, Any] = {
                "prompt_id": prompt_id,
                "generation_id": generation_id,
            }
            flagged_fields: list[str] = []
            manual_fields: list[str] = []

            for column in score_columns:
                values = [
                    int(pass_scores[(judge, pass_number, case_id)][column])
                    for pass_number in range(1, args.passes + 1)
                ]
                score_range = max(values) - min(values)
                automatic_mean = round(sum(values) / len(values), 6)
                manual_key = (judge, prompt_id, generation_id, column)
                manual_score = manual_overrides.get(manual_key)
                output_row[column] = (
                    round(manual_score, 6)
                    if manual_score is not None
                    else automatic_mean
                )
                if manual_score is not None:
                    manual_fields.append(column)

                if score_range > 1:
                    flagged_fields.append(column)
                    disagreement = {
                        "judge": judge,
                        "prompt_id": prompt_id,
                        "generation_id": generation_id,
                        "case_id": case_id,
                        "score_dimension": column,
                    }
                    for pass_number, value in enumerate(values, start=1):
                        disagreement[f"pass_{pass_number}"] = value
                    disagreement.update(
                        {
                            "minimum": min(values),
                            "maximum": max(values),
                            "range": score_range,
                            "automatic_mean": automatic_mean,
                            "manual_score": (
                                "" if manual_score is None else manual_score
                            ),
                            "resolved": manual_score is not None,
                        }
                    )
                    within_disagreements.append(disagreement)
                    template_key = (judge, prompt_id, generation_id)
                    template = within_template_by_case.setdefault(
                        template_key,
                        {
                            "judge": judge,
                            "prompt_id": prompt_id,
                            "generation_id": generation_id,
                        },
                    )
                    template[column] = manual_score

            for column in categorical_columns:
                values = [
                    str(pass_scores[(judge, pass_number, case_id)][column])
                    for pass_number in range(1, args.passes + 1)
                ]
                counts = Counter(values)
                output_row[column] = sorted(
                    counts,
                    key=lambda value: (-counts[value], value),
                )[0]
                output_row[f"{column}_votes"] = values

            output_row["within_judge_flagged_fields"] = flagged_fields
            output_row["within_judge_manual_fields"] = manual_fields
            aggregated_by_judge[judge][(prompt_id, generation_id)] = output_row

    pass_columns = [f"pass_{number}" for number in range(1, args.passes + 1)]
    within_fields = [
        "judge",
        "prompt_id",
        "generation_id",
        "case_id",
        "score_dimension",
        *pass_columns,
        "minimum",
        "maximum",
        "range",
        "automatic_mean",
        "manual_score",
        "resolved",
    ]
    write_csv(
        args.output_dir / "within_judge_disagreements.csv",
        within_disagreements,
        within_fields,
    )
    if within_template_by_case:
        write_jsonl(
            args.output_dir / "within_judge_manual_scores.template.jsonl",
            list(within_template_by_case.values()),
        )

    for judge in range(1, args.judges + 1):
        write_jsonl(
            args.output_dir / f"judge_{judge}.jsonl",
            [aggregated_by_judge[judge][key] for key in key_order],
        )

    cross_disagreements: list[dict] = []
    cross_template_by_case: dict[tuple[str, int], dict] = {}
    for prompt_id, generation_id in key_order:
        key = (prompt_id, generation_id)
        for column in score_columns:
            values = [
                float(aggregated_by_judge[judge][key][column])
                for judge in range(1, args.judges + 1)
            ]
            difference = max(values) - min(values)
            if difference > 1:
                row: dict[str, Any] = {
                    "prompt_id": prompt_id,
                    "generation_id": generation_id,
                    "score_dimension": column,
                }
                for judge, value in enumerate(values, start=1):
                    row[f"judge_{judge}"] = value
                row["difference"] = round(difference, 6)
                cross_disagreements.append(row)
                template = cross_template_by_case.setdefault(
                    key,
                    {
                        "prompt_id": prompt_id,
                        "generation_id": generation_id,
                    },
                )
                template[column] = None

    cross_fields = [
        "prompt_id",
        "generation_id",
        "score_dimension",
        *(f"judge_{judge}" for judge in range(1, args.judges + 1)),
        "difference",
    ]
    write_csv(
        args.output_dir / "cross_judge_disagreements.csv",
        cross_disagreements,
        cross_fields,
    )
    if cross_template_by_case:
        write_jsonl(
            args.output_dir / "manual_scores.template.jsonl",
            list(cross_template_by_case.values()),
        )

    unresolved_within = sum(
        not bool(row["resolved"]) for row in within_disagreements
    )
    summary = {
        "responses_per_judge": len(key_order),
        "judges": args.judges,
        "passes_per_judge": args.passes,
        "raw_score_files": args.judges * args.passes,
        "schema": args.schema,
        "score_columns": score_columns,
        "within_judge_flagged_dimensions": len(within_disagreements),
        "within_judge_unresolved_dimensions": unresolved_within,
        "within_judge_manual_overrides_applied": len(manual_overrides),
        "cross_judge_flagged_dimensions": len(cross_disagreements),
        "judge_files_ready": unresolved_within == 0,
        "final_manual_review_required": bool(cross_disagreements),
        "analysis_ready": unresolved_within == 0 and not cross_disagreements,
    }
    with (args.output_dir / "aggregation_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    print(json.dumps(summary, indent=2))
    if unresolved_within:
        print(
            "Manual within-judge review is required. Fill the generated template "
            "and rerun with --within-manual-scores."
        )
    elif cross_disagreements:
        print(
            "Within-judge aggregation is complete. Cross-judge disagreements "
            "remain for final manual review."
        )
    else:
        print("Judge files are ready for probe analysis.")


if __name__ == "__main__":
    main()
