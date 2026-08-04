# Public short-report publishing

## Source of truth

The research repository owns the authoritative report assets:

- `report/willingness_probe_report_v3.html`
- `output/pdf/willingness_probe_report.pdf`

The public report is hosted through GitHub Pages at
<https://tsoxxes.github.io/willingness-probe-report/>. Its source is the
separate `TSoxxes/willingness-probe-report` repository. A local checkout may be
placed at `public-report-pages/`; the outer repository ignores that directory
so the two Git histories do not overlap.

The files below are deployment copies and must not be edited directly:

- `public-report-pages/index.html`
- `public-report-pages/willingness_probe_report.pdf`

## Update workflow

1. Edit and review the authoritative HTML under `report/`.
2. Regenerate and visually verify the authoritative PDF under `output/pdf/`.
3. Copy both assets into the deployment repository:

   ```powershell
   python scripts/sync_public_report_site.py
   ```

4. Verify synchronization without writing:

   ```powershell
   python scripts/sync_public_report_site.py --check
   ```

5. Review the deployment-repository diff and verify its links locally.
6. Commit and push the deployment repository's `main` branch. GitHub Pages then
   updates the existing public URL without changing it.

The sync command fails clearly when the deployment checkout is missing or its
`origin` remote is not `TSoxxes/willingness-probe-report`. Use `--site-dir` if
the standalone repository is checked out somewhere else.
