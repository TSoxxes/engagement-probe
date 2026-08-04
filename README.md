# Willingness Probe — Free Pilot Prototype

This repository is a small, runnable proof of concept for one question:

> Among responses that still answer the user, can a model's pre-answer
> activations predict how fully it engages with the request?

It intentionally does **not** claim that the model has a distinct "willingness"
mechanism. The pilot only checks whether a candidate engagement signal is worth
studying further.

Round 1 is complete. Round 2 is frozen after a successful behavioral pilot,
with 40 ladders, a locked
confirmation split, revised behavioral targets, variable prespecified
generation counts, three independent judges, text-leakage controls, and
activation-over-text incremental analysis. It predicts prompt-conditioned
response behavior rather than claiming a prompt-independent willingness
variable. The full Round 2 generation is authorized against the frozen
protocol. See:

- `docs/round_1_report.md`
- `docs/round_2_audit_response.md`
- `docs/round_2_freeze_manifest.json`
- `docs/round_2_pilot_report.md`
- `docs/round_2_protocol.md`
- `docs/round_2_workflow.md`
- `docs/design_review_response.md`
- `docs/project_roadmap.md`

## What is included

- 24 prompts: 6 matched ladders × 4 rungs
- Three ladder categories: safety, professional advice, harmless uncertainty
- Gemma-2-2B-it generation on a free Kaggle GPU
- Residual-stream snapshots from every layer at the final prompt token
- A ready-to-paste rubric for two independent frontier-model judges
- Per-layer ridge probes with leave-one-ladder-out evaluation
- Shuffled-label, response-length, and token-cap baselines

## Repository layout

```text
willingness-probe/
  data/
    ladders.jsonl
    round2_ladders.jsonl
    round2_pilot_prompts.jsonl
    round2_prompt_audit.json
    judge_scores.example.jsonl
  docs/
    round_1_report.md
    round_2_audit_response.md
    round_2_freeze_manifest.json
    round_2_pilot_report.md
    round_2_protocol.md
    round_2_workflow.md
    design_review_response.md
    project_roadmap.md
  scoring/
    JUDGING_WORKFLOW.md
    batch_judge_prompt.md
    judge_prompt.md
    round2_batch_judge_prompt.md
    round2_prompt_annotation_prompt.md
  src/
    aggregate_judging.py
    aggregate_prompt_annotations.py
    analyze_round2.py
    audit_round2_prompts.py
    build_round2_dataset.py
    check_round2_pilot.py
    generation_utils.py
    generate.py
    analyze.py
    prepare_prompt_annotations.py
    prepare_judging.py
    scoring_schema.py
    text_features.py
  results/
  scripts/
    sync_public_report_site.py
  tests/
    make_smoke_fixture.py
    test_generation_utils.py
    test_judging_pipeline.py
    test_round2_pipeline.py
  requirements.txt
  README.md
```

## Public short-report site

The master short report is `report/willingness_probe_report_v3.html`, and its
master PDF is `output/pdf/willingness_probe_report.pdf`. The public URL is
<https://tsoxxes.github.io/willingness-probe-report/>, deployed from the
separate `TSoxxes/willingness-probe-report` GitHub repository. A local checkout
of that repository belongs at `public-report-pages/`; do not edit its report
copies directly.

Synchronize the deployment checkout from the master files with:

```powershell
python scripts/sync_public_report_site.py
```

Use `--check` to verify that the two repositories agree without changing
files. The sync command refuses to operate on a different Git remote. The full
publishing workflow is documented in `docs/public_report_publishing.md`.

## Local checks

The regression tests do not download or load Gemma:

```bash
python src/build_round2_dataset.py
python src/audit_round2_prompts.py
python -m unittest discover -s tests -v
```

To exercise the complete analysis path with tiny synthetic activations:

```bash
python tests/make_smoke_fixture.py
python src/analyze.py \
    --dataset data/ladders.jsonl \
    --responses tests/smoke_fixture/responses.jsonl \
    --activations tests/smoke_fixture/activations.npz \
    --run-config tests/smoke_fixture/run_config.json \
    --judge-files tests/smoke_fixture/judge_1.jsonl tests/smoke_fixture/judge_2.jsonl \
    --output-dir tests/smoke_output
```

## Important measurement detail

The activation is recorded at the last token of the fully formatted chat prompt,
immediately before generation begins. It is therefore the same for all sampled
answers to a prompt. The analysis averages judge scores and response length
across generations before fitting a probe. It does not pretend that a
pre-generation snapshot can explain random differences between sampled answers.

## Run on Kaggle

### 1. Prepare model access

Gemma-2-2B-it is gated on Hugging Face:

1. Accept the model terms on the Hugging Face model page.
2. Create a read token.
3. In Kaggle, add it as a secret named `HF_TOKEN`.
4. Turn on a T4 GPU and internet access for the notebook.

Upload this repository as a Kaggle dataset or clone it into the notebook
session. In a notebook cell, expose the secret without printing it:

```python
import os
from kaggle_secrets import UserSecretsClient
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
```

### 2. Install the small dependency set

From the repository directory:

```python
!pip install -q -r requirements.txt
```

Restart the notebook kernel if Kaggle asks you to after installation.

### 3. Generate responses and capture activations

```python
!python src/generate.py \
    --dataset data/ladders.jsonl \
    --output-dir results/run_main \
    --num-generations 4 \
    --max-new-tokens 600 \
    --temperature 0.7
```

The script uses float16 on a Kaggle T4 and bfloat16 on GPUs with native support.
It stops on the model's complete EOS configuration, including Gemma's
`<end_of_turn>` token. Expected outputs:

- `results/run_main/responses.jsonl`
- `results/run_main/activations.npz`
- `results/run_main/run_config.json`

Every response records `finish_reason` (`eos`, `length`, or `other`) and a
boolean `hit_token_cap`. The run configuration summarizes those outcomes.

The activation file stores float16 arrays shaped:

```text
[number of prompts, embedding plus transformer layers, hidden size]
```

### 4. Score the responses

The recommended workflow uses two judge models, three independently ordered
passes per judge, opaque case IDs, and automatic within- and cross-judge
disagreement checks.

Prepare the six blinded packets:

```bash
python src/prepare_judging.py \
    --responses results/run_main/responses.jsonl \
    --output-dir results/run_main/judging
```

Give each packet to its assigned model in a fresh conversation using
`scoring/batch_judge_prompt.md`, save the six raw outputs, and aggregate them:

```bash
python src/aggregate_judging.py \
    --mapping results/run_main/judging/private_mapping.jsonl \
    --raw-scores-dir results/run_main/judging/raw_scores \
    --output-dir results/run_main/judging/aggregated
```

Follow `scoring/JUDGING_WORKFLOW.md` for exact collection, validation,
within-judge adjudication, and cross-judge adjudication steps. The private
mapping and all experimental metadata must remain hidden from judges.

### 5. Fit and evaluate the probes

```python
!python src/analyze.py \
    --dataset data/ladders.jsonl \
    --responses results/run_main/responses.jsonl \
    --activations results/run_main/activations.npz \
    --run-config results/run_main/run_config.json \
    --judge-files \
      results/run_main/judging/aggregated/judge_1.jsonl \
      results/run_main/judging/aggregated/judge_2.jsonl \
    --manual-scores results/run_main/judging/aggregated/manual_scores.jsonl \
    --output-dir results/run_main/analysis
```

Omit `--manual-scores` if no manual corrections are needed.

Main outputs:

- `metrics_by_layer.csv`: held-out predictions summarized at every layer
- `predictions.csv`: every held-out prediction
- `best_layer_summary.json`: the strongest real-label layer
- `judge_disagreements.csv`: cases needing manual review
- `generation_diagnostics.json`: token-cap rates by prompt and category
- `probe_curve.png`: engagement probe versus all sensitivity baselines

The primary metric is held-out Spearman correlation. The plot compares the
activation probe with shuffled labels, response length, and token-cap rate.
With only six ladders this is a feasibility signal, not a stable scientific
estimate.

## Observed pilot diagnostics

The corrected `run_003` pilot produced all 96 expected responses and finite
activations. Twenty-three responses (24.0%) reached the uniform 600-token
ceiling. The capped responses were long, on-topic, and showed no obvious
repetitive degeneration, but most ended mid-list or mid-sentence. They were
concentrated in professional-advice and harmless-uncertainty prompts, with none
in the safety category. For that reason, token-cap rate is retained as an
explicit sensitivity baseline rather than silently treated as ordinary EOS.

## What counts as a useful result

A promising pilot would show that activation-based predictions outperform both
baselines on entirely unseen ladders. It would still **not** establish a new
mechanism: this prototype does not yet project out refusal, harmfulness, or
uncertainty directions, and it does not include causal steering.

A null result is also useful. It may mean the behavior is not stable, the small
model is too crude, or the apparent signal is mostly topic or response length.

## Scope deliberately left out

- No steering intervention
- No second model family
- No API automation for judges
- No hyperparameter search
- No claim of statistical significance
- No attempt to identify model experience or preferences

Those belong after this end-to-end pipeline has run successfully.
