# Public short-report publishing

## Source of truth

The research repository owns the authoritative report assets:

- `report/willingness_probe_report_v3.html`
- `output/pdf/willingness_probe_report.pdf`

`public-report-site/` is a separate nested Git repository connected to the
existing OpenAI Sites project. The outer repository ignores that directory so
the two Git histories do not overlap. Keeping the existing Sites project ID
preserves the report's existing URL when a new version is deployed.

The files below are deployment copies and must not be edited directly:

- `public-report-site/public/report.html`
- `public-report-site/public/willingness_probe_report.pdf`

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

5. Build and test from `public-report-site/`.
6. Commit the deployment repository separately and deploy a new version to the
   project ID already stored in `public-report-site/.openai/hosting.json`.

The sync command fails clearly when the deployment checkout is missing or when
its Sites project metadata is absent. Use `--site-dir` if the standalone site
repository is checked out somewhere else.
