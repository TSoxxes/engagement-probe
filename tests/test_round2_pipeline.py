"""Round 2 dataset, judging, and locked-analysis regression tests."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from build_round2_dataset import build_pilot_rows, build_rows
from aggregate_judging import read_json_records
from scoring_schema import ROUND2_SCORE_COLUMNS


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


class Round2PipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.workspace = ROOT / "tests" / "round2_test_workspace"
        shutil.rmtree(self.workspace, ignore_errors=True)
        self.workspace.mkdir()

    def tearDown(self) -> None:
        shutil.rmtree(self.workspace, ignore_errors=True)

    def test_frozen_dataset_shape(self) -> None:
        rows = build_rows()
        self.assertEqual(len(rows), 160)
        self.assertEqual(len({row["ladder_id"] for row in rows}), 40)
        self.assertEqual(sum(row["generation_count"] for row in rows), 576)
        self.assertEqual(sum(row["variability_subset"] for row in rows), 32)
        self.assertTrue(
            all(row["prompt_id"].startswith("r2v2_") for row in rows)
        )
        self.assertTrue(
            all(
                row["prompt_id"].endswith(
                    f"_{row['design_cell'].lower()}"
                )
                for row in rows
            )
        )
        refusal_rows = [row for row in rows if row["design_cell"] == "D"]
        self.assertEqual(
            sum(
                row["intended_refusal_basis"] == "capability"
                for row in refusal_rows
            ),
            20,
        )
        self.assertEqual(
            sum(
                row["intended_refusal_basis"] == "harmful_request"
                for row in refusal_rows
            ),
            20,
        )
        for rung in range(1, 5):
            self.assertEqual(
                {
                    cell: sum(
                        row["rung"] == rung and row["design_cell"] == cell
                        for row in rows
                    )
                    for cell in "ABCD"
                },
                {cell: 10 for cell in "ABCD"},
            )

    def test_pilot_is_separate_and_balanced(self) -> None:
        main_ids = {row["prompt_id"] for row in build_rows()}
        pilot = build_pilot_rows()
        self.assertEqual(len(pilot), 16)
        self.assertFalse(main_ids & {row["prompt_id"] for row in pilot})
        self.assertEqual(
            {
                cell: sum(row["design_cell"] == cell for row in pilot)
                for cell in "ABCD"
            },
            {cell: 4 for cell in "ABCD"},
        )
        required_by_generator = {
            "prompt_id",
            "ladder_id",
            "category",
            "rung",
            "prompt",
        }
        self.assertTrue(
            all(required_by_generator <= row.keys() for row in pilot)
        )
        self.assertTrue(
            all(1 <= row["rung"] <= 4 for row in pilot)
        )

    def test_score_reader_accepts_one_leading_preamble(self) -> None:
        path = self.workspace / "preamble.txt"
        path.write_text(
            'I will score the cases.{"case_id":"x","score":1}\n'
            '{"case_id":"y","score":2}\n',
            encoding="utf-8",
        )
        self.assertEqual(
            [row["case_id"] for row in read_json_records(path)],
            ["x", "y"],
        )

    def test_one_pass_batched_round2_judging(self) -> None:
        responses = [
            {
                "prompt_id": f"p{prompt}",
                "generation_id": generation,
                "prompt": f"Prompt {prompt}",
                "response": f"Response {generation}",
            }
            for prompt in range(6)
            for generation in range(2)
        ]
        responses_path = self.workspace / "responses.jsonl"
        write_jsonl(responses_path, responses)
        judging = self.workspace / "judging"
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "src" / "prepare_judging.py"),
                "--responses",
                str(responses_path),
                "--output-dir",
                str(judging),
                "--judges",
                "3",
                "--passes",
                "1",
                "--batch-size",
                "5",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        mapping = read_jsonl(judging / "private_mapping.jsonl")
        self.assertEqual(len(mapping), 36)
        mapped_key = {
            row["case_id"]: (row["prompt_id"], row["generation_id"])
            for row in mapping
        }
        cross_judge_blocks: dict[tuple[str, int], set[int]] = {}
        raw = judging / "raw_scores"
        for judge in (1, 2, 3):
            packet_files = sorted(
                (judging / "packets").glob(
                    f"judge_{judge}_pass_1_batch_*.jsonl"
                )
            )
            self.assertEqual([len(read_jsonl(path)) for path in packet_files], [5, 5, 2])
            ordered_packet = [
                row for path in packet_files for row in read_jsonl(path)
            ]
            for position, row in enumerate(ordered_packet):
                cross_judge_blocks.setdefault(
                    mapped_key[row["case_id"]], set()
                ).add(position // 4)
            for packet_path in packet_files:
                batch = packet_path.stem.rsplit("_", 1)[-1]
                score_rows = []
                for row in read_jsonl(packet_path):
                    score_rows.append(
                        {
                            "case_id": row["case_id"],
                            **{column: 2 for column in ROUND2_SCORE_COLUMNS},
                            "response_mode": "partial_answer",
                            "brief_reason": "Synthetic partial answer.",
                        }
                    )
                write_jsonl(
                    raw / f"judge_{judge}_pass_1_batch_{batch}.jsonl",
                    score_rows,
                )
        self.assertTrue(
            all(blocks == {0, 1, 2} for blocks in cross_judge_blocks.values())
        )
        output = judging / "aggregated"
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "src" / "aggregate_judging.py"),
                "--mapping",
                str(judging / "private_mapping.jsonl"),
                "--raw-scores-dir",
                str(raw),
                "--output-dir",
                str(output),
                "--judges",
                "3",
                "--passes",
                "1",
                "--schema",
                "round2",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        summary = json.loads(
            (output / "aggregation_summary.json").read_text(encoding="utf-8")
        )
        self.assertEqual(summary["schema"], "round2")
        self.assertEqual(summary["cross_judge_flagged_dimensions"], 0)
        row = read_jsonl(output / "judge_1.jsonl")[0]
        self.assertEqual(row["response_mode"], "partial_answer")

    def test_prompt_only_annotation(self) -> None:
        dataset = [
            {"prompt_id": f"p{number}", "prompt": f"Prompt {number}"}
            for number in range(5)
        ]
        dataset_path = self.workspace / "dataset.jsonl"
        write_jsonl(dataset_path, dataset)
        annotation = self.workspace / "prompt_annotation"
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "src" / "prepare_prompt_annotations.py"),
                "--dataset",
                str(dataset_path),
                "--output-dir",
                str(annotation),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        raw = annotation / "raw_scores"
        for judge in (1, 2, 3):
            rows = [
                {
                    "case_id": row["case_id"],
                    "prompt_harmfulness": 1,
                    "brief_reason": "Synthetic prompt.",
                }
                for row in read_jsonl(annotation / "packets" / f"judge_{judge}.jsonl")
            ]
            write_jsonl(raw / f"judge_{judge}.jsonl", rows)
        output = annotation / "aggregated"
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "src" / "aggregate_prompt_annotations.py"),
                "--mapping",
                str(annotation / "private_mapping.jsonl"),
                "--raw-scores-dir",
                str(raw),
                "--output-dir",
                str(output),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        scores = read_jsonl(output / "prompt_scores.jsonl")
        self.assertEqual(len(scores), 5)
        self.assertTrue(all(row["prompt_harmfulness"] == 1 for row in scores))

    def test_locked_round2_analysis_smoke(self) -> None:
        dataset = []
        responses = []
        judge_rows = {1: [], 2: []}
        prompt_ids = []
        activation_rows = []
        for ladder_number in range(6):
            split = "development" if ladder_number < 4 else "confirmation"
            for rung in range(1, 5):
                prompt_id = f"ladder_{ladder_number}_{rung}"
                prompt_ids.append(prompt_id)
                target = float(rung - 1)
                dataset.append(
                    {
                        "prompt_id": prompt_id,
                        "ladder_id": f"ladder_{ladder_number}",
                        "split": split,
                        "category": "synthetic",
                        "rung": rung,
                        "design_cell": "ABCD"[rung - 1],
                        "design_condition": (
                            "D_capability" if rung == 4 else "ABCD"[rung - 1]
                        ),
                        "prompt": f"Synthetic prompt {ladder_number} {rung}",
                        "intended_information_supply": (
                            "high" if rung < 3 else "low"
                        ),
                        "intended_refusal_language": (
                            "high" if rung in {2, 4} else "low"
                        ),
                        "intended_refusal_basis": (
                            "capability" if rung == 4 else "none"
                        ),
                        "variability_subset": False,
                        "generation_count": 1,
                    }
                )
                responses.append(
                    {
                        "prompt_id": prompt_id,
                        "generation_id": 0,
                        "response_length_tokens": 20 + rung,
                        "hit_token_cap": False,
                        "finish_reason": "eos",
                    }
                )
                score = {
                    "prompt_id": prompt_id,
                    "generation_id": 0,
                    **{column: target for column in ROUND2_SCORE_COLUMNS},
                }
                judge_rows[1].append(score)
                judge_rows[2].append(score)
                layer_zero = np.zeros(8, dtype=np.float16)
                layer_one = np.full(8, target, dtype=np.float16)
                activation_rows.append(np.stack([layer_zero, layer_one]))

        dataset_path = self.workspace / "dataset.jsonl"
        responses_path = self.workspace / "responses.jsonl"
        write_jsonl(dataset_path, dataset)
        write_jsonl(responses_path, responses)
        judge_paths = []
        for judge in (1, 2):
            path = self.workspace / f"judge_{judge}.jsonl"
            write_jsonl(path, judge_rows[judge])
            judge_paths.append(path)
        activations_path = self.workspace / "activations.npz"
        np.savez_compressed(
            activations_path,
            activations=np.stack(activation_rows),
            prompt_ids=np.asarray(prompt_ids),
            final_prompt_token_ids=np.zeros(len(prompt_ids), dtype=np.int64),
        )
        output = self.workspace / "analysis"
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "src" / "analyze_round2.py"),
                "--dataset",
                str(dataset_path),
                "--responses",
                str(responses_path),
                "--activations",
                str(activations_path),
                "--judge-files",
                *(str(path) for path in judge_paths),
                "--output-dir",
                str(output),
                "--shuffle-repeats",
                "1",
                "--bootstrap-repeats",
                "20",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        summary = json.loads(
            (output / "round2_summary.json").read_text(encoding="utf-8")
        )
        self.assertFalse(summary["confirmation_used_for_layer_selection"])
        self.assertEqual(summary["development_ladders"], 4)
        self.assertEqual(summary["confirmation_ladders"], 2)
        self.assertTrue((output / "target_summary.csv").exists())
        self.assertIn("primary_activation_over_text", summary)


if __name__ == "__main__":
    unittest.main()
