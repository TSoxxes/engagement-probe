"""Synchronize canonical report assets into the GitHub Pages repository."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_REMOTE = "https://github.com/TSoxxes/willingness-probe-report"
ASSET_PATHS = (
    (
        Path("report/willingness_probe_report_v3.html"),
        Path("index.html"),
    ),
    (
        Path("output/pdf/willingness_probe_report.pdf"),
        Path("willingness_probe_report.pdf"),
    ),
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def normalize_remote(remote: str) -> str:
    value = remote.strip().lower().removesuffix(".git").rstrip("/")
    if value.startswith("git@github.com:"):
        value = "https://github.com/" + value.removeprefix("git@github.com:")
    if value.startswith("ssh://git@github.com/"):
        value = "https://github.com/" + value.removeprefix(
            "ssh://git@github.com/"
        )
    return value


def validate_site_checkout(site_dir: Path) -> None:
    if not site_dir.is_dir():
        raise FileNotFoundError(
            f"Sites deployment checkout does not exist: {site_dir}"
        )
    if not (site_dir / ".git").exists():
        raise ValueError(
            f"Public-report checkout is not a standalone Git repository: {site_dir}"
        )
    remote = subprocess.run(
        ["git", "-C", str(site_dir), "remote", "get-url", "origin"],
        check=False,
        capture_output=True,
        text=True,
    )
    if remote.returncode != 0:
        raise ValueError(
            f"Public-report checkout has no readable origin remote: {site_dir}"
        )
    actual = normalize_remote(remote.stdout)
    expected = normalize_remote(EXPECTED_REMOTE)
    if actual != expected:
        raise ValueError(
            "Refusing to synchronize the wrong repository: "
            f"expected {EXPECTED_REMOTE}, found {remote.stdout.strip()}"
        )


def synchronize(root: Path, site_dir: Path, *, check: bool) -> list[str]:
    """Copy or verify report assets, returning human-readable results."""
    validate_site_checkout(site_dir)
    results: list[str] = []
    mismatches: list[str] = []

    for source_relative, destination_relative in ASSET_PATHS:
        source = root / source_relative
        destination = site_dir / destination_relative
        if not source.is_file():
            raise FileNotFoundError(f"Canonical report asset is missing: {source}")

        matches = destination.is_file() and digest(source) == digest(destination)
        if check:
            if matches:
                results.append(f"OK {source_relative}")
            else:
                mismatches.append(str(destination_relative))
            continue

        destination.parent.mkdir(parents=True, exist_ok=True)
        if not matches:
            shutil.copy2(source, destination)
        if digest(source) != digest(destination):
            raise OSError(f"Synchronized asset failed verification: {destination}")
        action = "UNCHANGED" if matches else "UPDATED"
        results.append(f"{action} {destination_relative}")

    if mismatches:
        joined = ", ".join(mismatches)
        raise ValueError(f"Deployment report assets are out of sync: {joined}")
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--site-dir",
        type=Path,
        default=ROOT / "public-report-pages",
        help="Standalone GitHub Pages repository checkout.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify synchronization without writing files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for result in synchronize(ROOT, args.site_dir.resolve(), check=args.check):
        print(result)


if __name__ == "__main__":
    main()
