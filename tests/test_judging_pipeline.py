"""End-to-end tests for blinded packet preparation and score aggregation."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCORE_COLUMNS = [
    "substantive_engagement",
    "direct_refusal",
    "safety_caution",
    "epistemic_uncertainty",
    "professional_redirection",
    "prompt_harmfulness",
]


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


class JudgingPipelineTest(unittest.TestCase):
    def test_balanced_blinding_and_disagreement_resolution(self) -> None:
        workspace = ROOT / "tests" / "judging_test_workspace"
        if workspace.exists():
            shutil.rmtree(workspace)
        workspace.mkdir()
        self.addCleanup(lambda: shutil.rmtree(workspace, ignore_errors=True))
        try:
            responses_path = workspace / "responses.jsonl"
            responses = []
            for prompt_number in range(6):
                for generation_id in range(2):
                    responses.append(
                        {
                            "prompt_id": f"prompt_{prompt_number}",
                            "generation_id": generation_id,
                            "ladder_id": "SECRET",
                            "category": "SECRET",
                            "rung": 4,
                            "seed": 123,
                            "prompt": f"User prompt {prompt_number}",
                            "response": f"Model response {generation_id}",
                            "response_length_tokens": 20,
                        }
                    )
            write_jsonl(responses_path, responses)

            judging_dir = workspace / "judging"
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "src" / "prepare_judging.py"),
                    "--responses",
                    str(responses_path),
                    "--output-dir",
                    str(judging_dir),
                    "--min-prompt-gap",
                    "2",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            mapping = read_jsonl(judging_dir / "private_mapping.jsonl")
            self.assertEqual(len(mapping), 24)
            mapping_by_case = {
                row["case_id"]: (row["prompt_id"], row["generation_id"])
                for row in mapping
            }
            block_size = len(responses) // 3
            for judge in (1, 2):
                block_appearances: dict[str, set[int]] = {}
                for pass_number in (1, 2, 3):
                    packet = read_jsonl(
                        judging_dir
                        / "packets"
                        / f"judge_{judge}_pass_{pass_number}.jsonl"
                    )
                    self.assertEqual(len(packet), len(responses))
                    for position, row in enumerate(packet):
                        self.assertEqual(
                            set(row),
                            {"case_id", "user_prompt", "model_response"},
                        )
                        block_appearances.setdefault(row["case_id"], set()).add(
                            position // block_size
                        )
                    prompt_ids = [
                        mapping_by_case[row["case_id"]][0] for row in packet
                    ]
                    for left, right in zip(prompt_ids, prompt_ids[1:]):
                        self.assertNotEqual(left, right)
                self.assertTrue(
                    all(blocks == {0, 1, 2} for blocks in block_appearances.values())
                )

            raw_dir = judging_dir / "raw_scores"
            flagged_case = next(
                row["case_id"] for row in mapping if row["judge"] == 1
            )
            flagged_key = mapping_by_case[flagged_case]
            for judge in (1, 2):
                for pass_number in (1, 2, 3):
                    packet = read_jsonl(
                        judging_dir
                        / "packets"
                        / f"judge_{judge}_pass_{pass_number}.jsonl"
                    )
                    score_rows = []
                    for case in packet:
                        score = {
                            "case_id": case["case_id"],
                            **{column: 1 for column in SCORE_COLUMNS},
                            "brief_reason": "Synthetic score.",
                        }
                        if judge == 1 and case["case_id"] == flagged_case:
                            score["substantive_engagement"] = [0, 2, 3][
                                pass_number - 1
                            ]
                        if judge == 2 and mapping_by_case[case["case_id"]] == flagged_key:
                            score["substantive_engagement"] = 3
                        score_rows.append(score)
                    write_jsonl(
                        raw_dir / f"judge_{judge}_pass_{pass_number}.jsonl",
                        score_rows,
                    )

            aggregate_dir = judging_dir / "aggregated"
            base_command = [
                sys.executable,
                str(ROOT / "src" / "aggregate_judging.py"),
                "--mapping",
                str(judging_dir / "private_mapping.jsonl"),
                "--raw-scores-dir",
                str(raw_dir),
                "--output-dir",
                str(aggregate_dir),
            ]
            subprocess.run(
                base_command,
                check=True,
                capture_output=True,
                text=True,
            )
            summary = json.loads(
                (aggregate_dir / "aggregation_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(summary["within_judge_flagged_dimensions"], 1)
            self.assertEqual(summary["within_judge_unresolved_dimensions"], 1)
            self.assertEqual(summary["cross_judge_flagged_dimensions"], 1)
            self.assertFalse(summary["analysis_ready"])

            manual_path = aggregate_dir / "within_manual.jsonl"
            write_jsonl(
                manual_path,
                [
                    {
                        "judge": 1,
                        "prompt_id": flagged_key[0],
                        "generation_id": flagged_key[1],
                        "substantive_engagement": 2.5,
                    }
                ],
            )
            subprocess.run(
                [*base_command, "--within-manual-scores", str(manual_path)],
                check=True,
                capture_output=True,
                text=True,
            )
            resolved = json.loads(
                (aggregate_dir / "aggregation_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(resolved["within_judge_unresolved_dimensions"], 0)
            self.assertEqual(resolved["cross_judge_flagged_dimensions"], 0)
            self.assertTrue(resolved["analysis_ready"])
        finally:
            shutil.rmtree(workspace, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
