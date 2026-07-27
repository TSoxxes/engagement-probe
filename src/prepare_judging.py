"""Create blinded, position-balanced response packets for repeated judging."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path


REQUIRED_RESPONSE_FIELDS = {
    "prompt_id",
    "generation_id",
    "prompt",
    "response",
}


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on {path}:{line_number}") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def validate_responses(rows: list[dict]) -> None:
    if not rows:
        raise ValueError("Response file is empty")
    seen: set[tuple[str, int]] = set()
    for row_number, row in enumerate(rows, start=1):
        missing = REQUIRED_RESPONSE_FIELDS - row.keys()
        if missing:
            raise ValueError(f"Response {row_number} lacks {sorted(missing)}")
        key = (str(row["prompt_id"]), int(row["generation_id"]))
        if key in seen:
            raise ValueError(f"Duplicate response key: {key}")
        seen.add(key)
        if not str(row["prompt"]).strip():
            raise ValueError(f"{key} has an empty prompt")
        if not str(row["response"]).strip():
            raise ValueError(f"{key} has an empty response")


def opaque_case_id(seed: int, judge: int, prompt_id: str, generation_id: int) -> str:
    payload = f"{seed}|{judge}|{prompt_id}|{generation_id}".encode()
    digest = hashlib.blake2s(payload, digest_size=5).hexdigest().upper()
    return f"J{judge}-{digest}"


def has_prompt_dispersion(
    order: list[int],
    rows: list[dict],
    minimum_gap: int,
) -> bool:
    positions: dict[str, list[int]] = defaultdict(list)
    for position, row_index in enumerate(order):
        positions[str(rows[row_index]["prompt_id"])].append(position)
    return all(
        right - left >= minimum_gap
        for prompt_positions in positions.values()
        for left, right in zip(prompt_positions, prompt_positions[1:])
    )


def balanced_orders(
    rows: list[dict],
    passes: int,
    seed: int,
    minimum_gap: int,
) -> list[list[int]]:
    """Put every case in each positional block once across all passes."""
    if passes < 1:
        raise ValueError("--passes must be at least 1")
    if len(rows) % passes:
        raise ValueError(
            f"{len(rows)} responses cannot be evenly balanced across {passes} passes"
        )
    if minimum_gap < 1:
        raise ValueError("--min-prompt-gap must be at least 1")

    rng = random.Random(seed)
    base = list(range(len(rows)))
    rng.shuffle(base)
    if passes == 1:
        for _ in range(10_000):
            rng.shuffle(base)
            if has_prompt_dispersion(base, rows, minimum_gap):
                return [base.copy()]
        raise RuntimeError(
            "Could not create a prompt-dispersed ordering; lower --min-prompt-gap"
        )
    block_size = len(rows) // passes
    blocks = [
        base[block_number * block_size : (block_number + 1) * block_size]
        for block_number in range(passes)
    ]

    orders: list[list[int]] = []
    for pass_index in range(passes):
        rotated = blocks[pass_index:] + blocks[:pass_index]
        for _ in range(10_000):
            candidate: list[int] = []
            for block in rotated:
                shuffled_block = block.copy()
                rng.shuffle(shuffled_block)
                candidate.extend(shuffled_block)
            if has_prompt_dispersion(candidate, rows, minimum_gap):
                orders.append(candidate)
                break
        else:
            raise RuntimeError(
                "Could not create a prompt-dispersed ordering; lower --min-prompt-gap"
            )

    if len({tuple(order) for order in orders}) != passes:
        raise RuntimeError("Repeated passes unexpectedly received the same ordering")
    return orders


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--judges", type=int, default=2)
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Split each judge pass into context-sized packet files.",
    )
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument(
        "--min-prompt-gap",
        type=int,
        default=2,
        help="Minimum positions between generations of the same prompt.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.judges < 2:
        raise ValueError("--judges must be at least 2")
    if args.batch_size is not None and args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")

    rows = read_jsonl(args.responses)
    validate_responses(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    packet_dir = args.output_dir / "packets"
    raw_score_dir = args.output_dir / "raw_scores"
    packet_dir.mkdir(exist_ok=True)
    raw_score_dir.mkdir(exist_ok=True)

    mapping_rows: list[dict] = []
    packet_files: list[str] = []
    cross_judge_orders = None
    if args.passes == 1 and len(rows) % args.judges == 0:
        cross_judge_orders = balanced_orders(
            rows,
            passes=args.judges,
            seed=args.seed + 9_999_991,
            minimum_gap=args.min_prompt_gap,
        )
    for judge in range(1, args.judges + 1):
        case_ids: dict[int, str] = {}
        used_case_ids: set[str] = set()
        for row_index, row in enumerate(rows):
            case_id = opaque_case_id(
                args.seed,
                judge,
                str(row["prompt_id"]),
                int(row["generation_id"]),
            )
            if case_id in used_case_ids:
                raise RuntimeError(f"Opaque case ID collision: {case_id}")
            used_case_ids.add(case_id)
            case_ids[row_index] = case_id
            mapping_rows.append(
                {
                    "judge": judge,
                    "case_id": case_id,
                    "prompt_id": row["prompt_id"],
                    "generation_id": int(row["generation_id"]),
                }
            )

        if cross_judge_orders is not None:
            orders = [cross_judge_orders[judge - 1]]
        else:
            orders = balanced_orders(
                rows,
                passes=args.passes,
                seed=args.seed + judge * 1_000_003,
                minimum_gap=args.min_prompt_gap,
            )
        for pass_number, order in enumerate(orders, start=1):
            packet_rows = [
                {
                    "case_id": case_ids[row_index],
                    "user_prompt": rows[row_index]["prompt"],
                    "model_response": rows[row_index]["response"],
                }
                for row_index in order
            ]
            if args.batch_size is None:
                batches = [packet_rows]
            else:
                batches = [
                    packet_rows[start : start + args.batch_size]
                    for start in range(0, len(packet_rows), args.batch_size)
                ]
            for batch_number, batch_rows in enumerate(batches, start=1):
                if len(batches) == 1:
                    filename = f"judge_{judge}_pass_{pass_number}.jsonl"
                else:
                    filename = (
                        f"judge_{judge}_pass_{pass_number}_"
                        f"batch_{batch_number:02d}.jsonl"
                    )
                write_jsonl(packet_dir / filename, batch_rows)
                packet_files.append(str(Path("packets") / filename))

    write_jsonl(args.output_dir / "private_mapping.jsonl", mapping_rows)
    manifest = {
        "source_responses": str(args.responses),
        "num_responses": len(rows),
        "judges": args.judges,
        "passes_per_judge": args.passes,
        "seed": args.seed,
        "minimum_same_prompt_gap": args.min_prompt_gap,
        "batch_size": args.batch_size,
        "position_balancing": (
            (
                f"Each case appears once in each of {args.judges} positional "
                "blocks across judges."
            )
            if cross_judge_orders is not None
            else (
                f"Each case appears once in each of {args.passes} positional "
                "blocks for its judge."
            )
        ),
        "packet_files": packet_files,
        "raw_score_filename_pattern": "raw_scores/judge_{judge}_pass_{pass}.jsonl",
        "private_mapping": "private_mapping.jsonl",
        "judge_visible_fields": ["case_id", "user_prompt", "model_response"],
    }
    with (args.output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")

    print(
        f"Created {len(packet_files)} blinded packets for {len(rows)} responses "
        f"under {args.output_dir}"
    )
    print(f"Keep {args.output_dir / 'private_mapping.jsonl'} away from judges.")


if __name__ == "__main__":
    main()
