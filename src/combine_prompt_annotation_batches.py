"""Combine validated web-chat prompt-annotation batches by judge."""

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
            raise ValueError(f"{path}:{line_number}: expected an object")
        if {"case_id", "prompt_harmfulness"} - row.keys():
            raise ValueError(f"{path}:{line_number}: missing required fields")
        value = row["prompt_harmfulness"]
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 3:
            raise ValueError(f"{path}:{line_number}: invalid harmfulness score")
        rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parts-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--judges", type=int, default=3)
    parser.add_argument("--batches", type=int, default=4)
    args = parser.parse_args()

    for judge in range(1, args.judges + 1):
        rows: list[dict[str, Any]] = []
        for batch in range(1, args.batches + 1):
            path = args.parts_dir / f"judge_{judge}_batch_{batch:02d}.jsonl"
            if not path.exists():
                raise FileNotFoundError(f"Missing batch: {path}")
            rows.extend(read_jsonl(path))

        case_ids = [str(row["case_id"]) for row in rows]
        duplicates = sorted(
            case_id for case_id in set(case_ids) if case_ids.count(case_id) > 1
        )
        if duplicates:
            raise ValueError(f"Judge {judge} has duplicate cases: {duplicates}")

        output = args.output_dir / f"judge_{judge}.jsonl"
        write_jsonl(output, rows)
        print(f"Wrote {len(rows)} Judge {judge} scores to {output}")


if __name__ == "__main__":
    main()
