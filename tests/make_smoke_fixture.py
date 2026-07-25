"""Create tiny synthetic files for testing analyze.py without downloading Gemma."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "smoke_fixture"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def main() -> None:
    FIXTURE.mkdir(parents=True, exist_ok=True)
    dataset = [
        json.loads(line)
        for line in (ROOT / "data" / "ladders.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    rng = np.random.default_rng(42)
    engagement = np.asarray([(row["rung"] - 1) / 3 for row in dataset], dtype=np.float32)
    activations = rng.normal(size=(len(dataset), 5, 16)).astype(np.float32)
    activations[:, 3, 0] += engagement * 4
    np.savez_compressed(
        FIXTURE / "activations.npz",
        activations=activations,
        prompt_ids=np.asarray([row["prompt_id"] for row in dataset]),
        final_prompt_token_ids=np.zeros(len(dataset), dtype=np.int64),
    )

    responses = []
    judge_1 = []
    judge_2 = []
    for row in dataset:
        for generation_id in range(2):
            base_score = row["rung"] - 1
            responses.append(
                {
                    **row,
                    "generation_id": generation_id,
                    "response": "Synthetic response",
                    "response_length_tokens": 20 + row["rung"],
                    "response_length_chars": 50,
                    "hit_token_cap": False,
                    "finish_reason": "eos",
                }
            )
            score = {
                "prompt_id": row["prompt_id"],
                "generation_id": generation_id,
                "substantive_engagement": base_score,
                "direct_refusal": 0,
                "safety_caution": int(row["category"] == "safety"),
                "epistemic_uncertainty": int(row["category"] == "harmless_uncertainty"),
                "professional_redirection": int(row["category"] == "professional_advice"),
                "prompt_harmfulness": int(row["category"] == "safety"),
            }
            judge_1.append(score)
            judge_2.append(score)
    write_jsonl(FIXTURE / "responses.jsonl", responses)
    write_jsonl(FIXTURE / "judge_1.jsonl", judge_1)
    write_jsonl(FIXTURE / "judge_2.jsonl", judge_2)
    (FIXTURE / "run_config.json").write_text(
        json.dumps(
            {
                "max_new_tokens": 600,
                "generation_eos_token_id": [1, 107],
                "finish_reason_counts": {
                    "eos": len(responses),
                    "length": 0,
                    "other": 0,
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(FIXTURE)


if __name__ == "__main__":
    main()
