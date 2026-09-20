"""Verify that every file frozen at the Round 2 protocol freeze is byte-identical.

The Round 2 change policy is: "Any change to a hashed file requires a new
protocol version and a new confirmation set." That policy is only meaningful if
a violation is actually detected, so it is checked here rather than trusted.

A failure here is not a routine test failure. It means a frozen artifact
has been edited, and the correct response is to restore the file -- not to
update the manifest.
"""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs" / "round_2_freeze_manifest.json"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


class FreezeManifestTest(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    def test_manifest_declares_frozen_status(self) -> None:
        self.assertEqual(self.manifest["status"], "frozen")
        self.assertEqual(self.manifest["protocol_version"], "0.3")

    def test_hashed_files_are_unchanged(self) -> None:
        for relative, expected in self.manifest["sha256"].items():
            with self.subTest(file=relative):
                path = ROOT / relative
                self.assertTrue(path.is_file(), f"frozen file is missing: {relative}")
                self.assertEqual(
                    sha256_of(path),
                    expected.upper(),
                    f"{relative} no longer matches the frozen protocol. Restore the "
                    f"file; do not update the manifest.",
                )

    def test_frozen_dataset_is_unchanged(self) -> None:
        dataset = self.manifest["dataset"]
        path = ROOT / dataset["path"]
        self.assertTrue(path.is_file(), f"frozen dataset is missing: {dataset['path']}")
        self.assertEqual(sha256_of(path), dataset["sha256"].upper())

    def test_frozen_dataset_has_the_declared_shape(self) -> None:
        dataset = self.manifest["dataset"]
        rows = [
            json.loads(line)
            for line in (ROOT / dataset["path"]).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(len(rows), dataset["prompts"])
        self.assertEqual(len({row["ladder_id"] for row in rows}), dataset["ladders"])
        self.assertEqual(
            sum(row["generation_count"] for row in rows),
            dataset["total_generations"],
        )

        splits = {"development": set(), "confirmation": set()}
        for row in rows:
            if row["split"] in splits:
                splits[row["split"]].add(row["ladder_id"])
        self.assertEqual(len(splits["development"]), dataset["development_ladders"])
        self.assertEqual(len(splits["confirmation"]), dataset["confirmation_ladders"])


if __name__ == "__main__":
    unittest.main()
