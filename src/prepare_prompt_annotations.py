"""Create blinded prompt-only harmfulness annotation packets."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from prepare_judging import read_jsonl, write_jsonl


def case_id(seed: int, judge: int, prompt_id: str) -> str:
    digest = hashlib.blake2s(
        f"{seed}|prompt|{judge}|{prompt_id}".encode(),
        digest_size=5,
    ).hexdigest().upper()
    return f"P{judge}-{digest}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--judges", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Also write web-chat-sized packet batches while retaining full packets.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.judges < 2:
        raise ValueError("--judges must be at least 2")
    if args.batch_size is not None and args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    rows = read_jsonl(args.dataset)
    required = {"prompt_id", "prompt"}
    if not rows:
        raise ValueError("Dataset is empty")
    if any(required - row.keys() for row in rows):
        raise ValueError("Every prompt must contain prompt_id and prompt")
    if len({str(row["prompt_id"]) for row in rows}) != len(rows):
        raise ValueError("prompt_id values must be unique")

    packet_dir = args.output_dir / "packets"
    raw_dir = args.output_dir / "raw_scores"
    packet_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    batch_dir = packet_dir / "batches"
    if args.batch_size is not None:
        batch_dir.mkdir(parents=True, exist_ok=True)
    mapping: list[dict] = []
    for judge in range(1, args.judges + 1):
        rng = random.Random(args.seed + judge * 1_000_003)
        order = list(range(len(rows)))
        rng.shuffle(order)
        ids = {
            index: case_id(args.seed, judge, str(row["prompt_id"]))
            for index, row in enumerate(rows)
        }
        packet_rows = [
            {
                "case_id": ids[index],
                "user_prompt": rows[index]["prompt"],
            }
            for index in order
        ]
        write_jsonl(packet_dir / f"judge_{judge}.jsonl", packet_rows)
        if args.batch_size is not None:
            for batch_number, start in enumerate(
                range(0, len(packet_rows), args.batch_size), start=1
            ):
                write_jsonl(
                    batch_dir
                    / f"judge_{judge}_batch_{batch_number:02d}.jsonl",
                    packet_rows[start : start + args.batch_size],
                )
        mapping.extend(
            {
                "judge": judge,
                "case_id": ids[index],
                "prompt_id": row["prompt_id"],
            }
            for index, row in enumerate(rows)
        )
    write_jsonl(args.output_dir / "private_mapping.jsonl", mapping)
    print(
        f"Created {args.judges} prompt-only packets for {len(rows)} prompts "
        f"under {args.output_dir}"
    )
    if args.batch_size is not None:
        batches_per_judge = (len(rows) + args.batch_size - 1) // args.batch_size
        print(
            f"Also created {batches_per_judge} web-chat batches per judge "
            f"with at most {args.batch_size} prompts each."
        )


if __name__ == "__main__":
    main()
