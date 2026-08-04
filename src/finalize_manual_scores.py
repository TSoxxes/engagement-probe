"""Apply audited manual-adjudication overrides to a complete score draft."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in rows
    )
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--overrides", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = read_jsonl(args.draft)
    overrides = read_jsonl(args.overrides)
    by_key: dict[tuple[str, int], dict[str, Any]] = {}

    for row in rows:
        key = (str(row["prompt_id"]), int(row["generation_id"]))
        if key in by_key:
            raise ValueError(f"Duplicate draft row: {key}")
        score_columns = [
            name for name in row if name not in {"prompt_id", "generation_id"}
        ]
        if not score_columns:
            raise ValueError(f"Draft row has no score columns: {key}")
        for dimension in score_columns:
            score = row[dimension]
            if score is None or not isinstance(score, (int, float)) or not 0 <= score <= 3:
                raise ValueError(
                    f"Invalid draft score for {key} {dimension}: {score!r}"
                )
        by_key[key] = row

    seen_overrides: set[tuple[str, int, str]] = set()
    for override in overrides:
        key = (str(override["prompt_id"]), int(override["generation_id"]))
        dimension = str(override["dimension"])
        override_key = (*key, dimension)
        if override_key in seen_overrides:
            raise ValueError(f"Duplicate override: {override_key}")
        seen_overrides.add(override_key)
        if key not in by_key:
            raise ValueError(f"Override does not match a draft row: {override_key}")
        if dimension not in by_key[key]:
            raise ValueError(
                f"Override dimension was not flagged in the draft: {override_key}"
            )
        score = override["score"]
        if not isinstance(score, (int, float)) or not 0 <= score <= 3:
            raise ValueError(f"Invalid override score for {override_key}: {score!r}")
        by_key[key][dimension] = score

    write_jsonl(args.output, rows)
    print(
        f"Wrote {len(rows)} adjudicated cases with "
        f"{len(overrides)} audited overrides to {args.output}"
    )


if __name__ == "__main__":
    main()
