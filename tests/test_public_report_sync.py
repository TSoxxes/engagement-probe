from __future__ import annotations

import shutil
import subprocess
import sys
import unittest
from pathlib import Path
from uuid import uuid4


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from sync_public_report_site import (  # noqa: E402
    ASSET_PATHS,
    EXPECTED_REMOTE,
    synchronize,
)


TEST_TEMP_ROOT = Path(__file__).parents[1] / ".tmp" / "tests"


class PublicReportSyncTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        TEST_TEMP_ROOT.mkdir(parents=True, exist_ok=True)

    def make_workspaces(self, temporary: Path) -> tuple[Path, Path]:
        root = Path(temporary) / "research"
        site = Path(temporary) / "site"
        subprocess.run(
            ["git", "init", str(site)], check=True, capture_output=True
        )
        subprocess.run(
            ["git", "-C", str(site), "remote", "add", "origin", EXPECTED_REMOTE],
            check=True,
            capture_output=True,
        )
        for source_relative, _ in ASSET_PATHS:
            source = root / source_relative
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes(f"canonical:{source_relative}".encode())
        return root, site

    def make_temporary_directory(self) -> Path:
        temporary = TEST_TEMP_ROOT / uuid4().hex
        temporary.mkdir()
        self.addCleanup(shutil.rmtree, temporary, True)
        return temporary

    def test_sync_copies_and_check_verifies_assets(self) -> None:
        root, site = self.make_workspaces(self.make_temporary_directory())
        results = synchronize(root, site, check=False)
        self.assertTrue(all(result.startswith("UPDATED") for result in results))
        self.assertEqual(len(synchronize(root, site, check=True)), 2)

    def test_check_rejects_drift(self) -> None:
        root, site = self.make_workspaces(self.make_temporary_directory())
        synchronize(root, site, check=False)
        destination = site / ASSET_PATHS[0][1]
        destination.write_text("drift", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "out of sync"):
            synchronize(root, site, check=True)

    def test_rejects_wrong_deployment_repository(self) -> None:
        root, site = self.make_workspaces(self.make_temporary_directory())
        subprocess.run(
            [
                "git",
                "-C",
                str(site),
                "remote",
                "set-url",
                "origin",
                "https://example.com/wrong",
            ],
            check=True,
            capture_output=True,
        )
        with self.assertRaisesRegex(ValueError, "wrong repository"):
            synchronize(root, site, check=True)


if __name__ == "__main__":
    unittest.main()
