# Released Round 2 artifacts

This directory holds the data behind the Round 2 numbers reported in
[the public short report](https://tsoxxes.github.io/engagement-probe-report/)
and in [`docs/round_2_results_summary.md`](../docs/round_2_results_summary.md).

Everything here is the output of the frozen protocol described in
[`docs/round_2_protocol.md`](../docs/round_2_protocol.md). Nothing in this
directory was edited after the fact; the manual adjudication files record
corrections as separate overlay files rather than by rewriting judge output.

## What is here

```text
results/round2_main/
  validation_summary.json                  run-level integrity checks
  run_round2/
    responses.jsonl                        576 generated responses + metadata
    activations.npz                        160 x 27 x 2304 pre-answer states
    run_config.json                        model, seeds, decoding settings
    analysis/                              all analysis outputs (see below)
    judging_web/
      aggregated/                          the 3 judges' final per-case scores
      manifest.json                        judging run provenance
      PROTOCOL_DEVIATIONS.md               every recorded deviation
    prompt_annotations/
      aggregated/                          prompt-level harmfulness annotation
```

### The files that carry the headline result

| File | What it holds |
|---|---|
| `analysis/round2_summary.json` | Every reported primary and secondary number |
| `analysis/confirmation_predictions.csv` | Per-prompt held-out predictions on the untouched confirmation set |
| `analysis/development_metrics_by_layer.csv` | Layer sweep used to select layer 25 |
| `analysis/repeated_split_sensitivity/` | The 200-split post-hoc stability analysis |
| `analysis/length_diagnostics/` | Response-length confound diagnostics |
| `analysis/reliability_summary.json` | Inter-judge reliability estimates |
| `judging_web/aggregated/judge_{1,2,3}.jsonl` | 1,728 blinded ratings before adjudication |
| `judging_web/aggregated/manual_scores.jsonl` | The 209 adjudicated cases |

## What is deliberately not here

| Not released | Size | Why |
|---|---|---|
| `judging_web/packets/` | 3.5 MB | Blinded input packets — process material, reconstructible from `responses.jsonl` |
| `judging_web/manual_review/`, `raw_scores/` | 1.3 MB | Per-pass raw judge transcripts behind the aggregates |
| `results/archive/` | 7 MB | Round 1 pilot runs, superseded by Round 2 |

Everything the confirmatory analysis consumes **is** here, so the primary result
can be re-derived from this repository on a laptop with no GPU and no model
access:

```bash
python src/analyze_round2.py \
    --dataset data/round2_ladders.jsonl \
    --responses results/round2_main/run_round2/responses.jsonl \
    --activations results/round2_main/run_round2/activations.npz \
    --run-config results/round2_main/run_round2/run_config.json \
    --judge-files \
      results/round2_main/run_round2/judging_web/aggregated/judge_1.jsonl \
      results/round2_main/run_round2/judging_web/aggregated/judge_2.jsonl \
      results/round2_main/run_round2/judging_web/aggregated/judge_3.jsonl \
    --manual-scores results/round2_main/run_round2/judging_web/aggregated/manual_scores.jsonl \
    --prompt-scores results/round2_main/run_round2/prompt_annotations/aggregated/prompt_scores.jsonl \
    --output-dir /tmp/round2_recheck
```

Then compare `/tmp/round2_recheck/round2_summary.json` against the committed
`analysis/round2_summary.json`. The bootstrap is seeded (`--seed`, default
20260801).

This has been checked: re-running the command above reproduces all 70 numeric
fields, with every reported figure — the selected layer, both confirmation
Spearman values, and the primary delta — identical to the last digit. The
largest disagreement anywhere in the file is 3e-15, in a bootstrap interval
bound, which is floating-point summation order and not a difference in result.
Compare with a tolerance rather than exact string equality.

## Verifying the headline numbers

The primary confirmation-set result, straight from the released predictions:

```bash
python - <<'EOF'
import json
summary = json.load(open(
    "results/round2_main/run_round2/analysis/round2_summary.json"))
print("confirmation prompts:", summary["confirmation_prompts"])
print("activation Spearman:", round(
    summary["primary_confirmation_metrics"]["spearman"], 4))
print("sparse-text Spearman:", round(
    summary["text_only_confirmation_metrics"]["spearman"], 4))
print("advantage:", round(
    summary["primary_activation_over_text"]["spearman_delta"]["observed"], 4))
EOF
```

Expected: 40 confirmation prompts, activation Spearman 0.8679, sparse-text
Spearman 0.8219, advantage +0.0460. That advantage did **not** clear the
preregistered rule — its paired ladder-bootstrap interval crossed zero — which
is why the report calls the primary test inconclusive.

## A note on the released generations

`responses.jsonl` contains every response the model produced, including its
responses to the 40 deliberately harmful-request prompts. Those prompts were
audited and softened for unnecessary operational detail before use (see
[`docs/round_2_protocol.md`](../docs/round_2_protocol.md)), and the
overwhelming majority of the resulting responses are refusals.

A small number are not, and those are kept rather than removed. A dataset about
how fully a model answers would be dishonest if it quietly dropped the cases
where the model answered something it should have declined — those cases are
the finding, not an accident. The compliant responses here are generic,
widely-published-knowledge material from a 2B model, not operational uplift.

## Citing or reusing

Released under the repository's MIT license. The generations are outputs of
`google/gemma-2-2b-it` and are also subject to the
[Gemma Terms of Use](https://ai.google.dev/gemma/terms).
