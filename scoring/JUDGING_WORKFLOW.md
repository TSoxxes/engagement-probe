# Round 1 repeated blind-judging workflow

This document records the completed Round 1 procedure: two judges with three
passes each. It is not the Round 2 judging plan. For frozen Round 2, use
`docs/round_2_workflow.md`, which specifies three different judge models with
one pass each and protects the development/confirmation split.

This workflow scores every generated response three times with each of two
independent judge models. The preparation script strips experimental metadata,
gives each judge opaque IDs, disperses generations from the same prompt, and
balances every case across the beginning, middle, and end of that judge's three
passes.

## Files produced

```text
run_003/
  judging/
    manifest.json
    private_mapping.jsonl
    packets/
      judge_1_pass_1.jsonl
      judge_1_pass_2.jsonl
      judge_1_pass_3.jsonl
      judge_2_pass_1.jsonl
      judge_2_pass_2.jsonl
      judge_2_pass_3.jsonl
    raw_scores/
      judge_1_pass_1.jsonl
      judge_1_pass_2.jsonl
      judge_1_pass_3.jsonl
      judge_2_pass_1.jsonl
      judge_2_pass_2.jsonl
      judge_2_pass_3.jsonl
    aggregated/
      judge_1.jsonl
      judge_2.jsonl
      within_judge_disagreements.csv
      cross_judge_disagreements.csv
      aggregation_summary.json
```

Never give `private_mapping.jsonl`, `manifest.json`, another pass's scores, or
any aggregation output to a judge.

## 1. Prepare the six packets

From the repository root in PowerShell:

```powershell
$run = "results/archive/willingness-probe-run-003/run_003"

python src/prepare_judging.py `
  --responses "$run/responses.jsonl" `
  --output-dir "$run/judging"
```

Preparation uses a fixed seed by default. Do not regenerate the packets after
judging begins, because doing so could invalidate the private mapping.

## 2. Collect six fresh judge passes

Use two genuinely different judge models if possible. Keep the same model and
settings for all three passes belonging to a particular judge.

For each of the six packet files:

1. Start a fresh conversation with the appropriate judge model. Do not reuse a
   conversation from another pass.
2. Attach exactly one packet from `judging/packets/`.
3. Paste the complete contents of `scoring/batch_judge_prompt.md`.
4. Save the model's complete response under the matching filename in
   `judging/raw_scores/`.

For example, the scores produced from `packets/judge_1_pass_2.jsonl` must be
saved as `raw_scores/judge_1_pass_2.jsonl`.

The aggregator accepts plain JSONL, a JSON array, or one Markdown-fenced JSONL
block, although the prompt asks for plain JSONL. Do not manually reorder lines
or edit score values.

If a judge omits cases or its response is truncated, repeat that entire pass in
a new conversation. The validator rejects missing, duplicate, and unknown case
IDs.

## 3. Validate and aggregate the passes

```powershell
python src/aggregate_judging.py `
  --mapping "$run/judging/private_mapping.jsonl" `
  --raw-scores-dir "$run/judging/raw_scores" `
  --output-dir "$run/judging/aggregated"
```

For each judge, response, and score dimension, the script computes the mean of
the three pass scores. A pass range greater than one point is written to
`within_judge_disagreements.csv`.

## 4. Resolve within-judge disagreements

If `aggregation_summary.json` reports unresolved within-judge dimensions:

1. Copy
   `aggregated/within_judge_manual_scores.template.jsonl` to
   `aggregated/within_judge_manual_scores.jsonl`.
2. Review the original prompt and response plus the three raw reasons.
3. Replace each `null` flagged score with the adjudicated score from 0 to 3.
4. Rerun aggregation with the override file:

```powershell
python src/aggregate_judging.py `
  --mapping "$run/judging/private_mapping.jsonl" `
  --raw-scores-dir "$run/judging/raw_scores" `
  --within-manual-scores "$run/judging/aggregated/within_judge_manual_scores.jsonl" `
  --output-dir "$run/judging/aggregated"
```

Do not alter unflagged dimensions. The script applies manual values only to the
dimensions present in the override file.

## 5. Review cross-judge disagreements

After within-judge adjudication, inspect
`aggregated/cross_judge_disagreements.csv`. It flags any dimension where the
two aggregated judges differ by more than one point.

If flags exist:

1. Copy `aggregated/manual_scores.template.jsonl` to
   `aggregated/manual_scores.jsonl`.
2. Replace each `null` with the final adjudicated score.
3. Supply this file to `analyze.py` as `--manual-scores`.

## 6. Run the probe analysis

If cross-judge manual scores are required:

```powershell
python src/analyze.py `
  --dataset data/ladders.jsonl `
  --responses "$run/responses.jsonl" `
  --activations "$run/activations.npz" `
  --run-config "$run/run_config.json" `
  --judge-files `
    "$run/judging/aggregated/judge_1.jsonl" `
    "$run/judging/aggregated/judge_2.jsonl" `
  --manual-scores "$run/judging/aggregated/manual_scores.jsonl" `
  --output-dir "$run/analysis"
```

If no cross-judge manual scores are required, omit the `--manual-scores` line.

The two `judge_*.jsonl` files contain restored `prompt_id` and `generation_id`
values plus the within-judge three-pass means. The analyzer averages those two
judge values, then applies any final manual overrides.
