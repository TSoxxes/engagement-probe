# Round 2 execution workflow

This document gives the exact order of operations. Do not inspect confirmation
results until the development analysis settings and manual scoring decisions
are frozen.

## 1. Validate the repository locally

```powershell
python src/build_round2_dataset.py
python src/audit_round2_prompts.py
python -m unittest discover -s tests -v
```

Do not continue to Kaggle unless the audit exits successfully. Its full JSON
record is saved to `data/round2_prompt_audit.json`.

The text-only cell and surface-only D-subtype accuracies are diagnostic
warnings, not structural failures under protocol 0.3. Preserve and report
them; do not describe the experiment as finding a prompt-independent
willingness variable.

Expected dataset invariants:

- 160 prompts
- 40 ladders
- 30 development ladders
- 10 confirmation ladders
- 32 prompts in the six-generation variability subset
- variability allocation B=16, C=8, D=8
- 576 total generations
- every rung contains ten A, ten B, ten C, and ten D prompts
- D conditions contain 20 capability and 20 harmful-request prompts

## 2. Run and judge the behavioral pilot

Before freezing the main dataset, run the separate pilot described in
`docs/round_2_protocol.md`. Do not use pilot prompts in the main dataset.
Check that B actually produces more response-level refusal language than A,
that C usually produces engagement below 2, and that judges can separate the
four potentially overlapping safety dimensions.

Generate its 48 responses first:

```powershell
python src/generate.py `
  --dataset data/round2_pilot_prompts.jsonl `
  --output-dir results/round2_pilot `
  --num-generations 3 `
  --generation-count-field generation_count `
  --max-new-tokens 800 `
  --temperature 0.7 `
  --top-p 0.95
```

Prepare and score the pilot with the same Round 2 rubric and at least two
independent judges:

```powershell
python src/prepare_judging.py `
  --responses results/round2_pilot/responses.jsonl `
  --output-dir results/round2_pilot/judging `
  --judges 2 `
  --passes 1 `
  --batch-size 48

python src/aggregate_judging.py `
  --mapping results/round2_pilot/judging/private_mapping.jsonl `
  --raw-scores-dir results/round2_pilot/judging/raw_scores `
  --output-dir results/round2_pilot/judging/aggregated `
  --judges 2 `
  --passes 1 `
  --schema round2

python src/check_round2_pilot.py `
  --judge-files `
    results/round2_pilot/judging/aggregated/judge_1.jsonl `
    results/round2_pilot/judging/aggregated/judge_2.jsonl
```

Complete the three manual checks listed in `pilot_check.json` as well.

If the pilot fails, revise and rebuild the main dataset, rerun the text audit,
and repeat the pilot with new pilot prompts. When both pass:

1. record the SHA-256 of `data/round2_ladders.jsonl`;
2. change the protocol status to frozen; and
3. upload that exact repository state to Kaggle.

## 3. Configure Kaggle

Use a T4 accelerator, enable internet, accept the Gemma model license on
Hugging Face, and add a Kaggle secret named `HF_TOKEN`.

```python
import os
from kaggle_secrets import UserSecretsClient

os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
```

Copy or attach the repository, change into its writable working copy, and
install requirements:

```python
!pip install -q -r requirements.txt
```

## 4. Run generation as a saved Kaggle version

Use Kaggle's non-interactive **Save Version / Run All** execution:

```python
!python -u src/generate.py \
    --dataset data/round2_ladders.jsonl \
    --output-dir results/run_round2 \
    --num-generations 3 \
    --generation-count-field generation_count \
    --max-new-tokens 800 \
    --temperature 0.7 \
    --top-p 0.95
```

Validate the result:

```python
import json
from pathlib import Path

run = Path("results/run_round2")
responses = [
    json.loads(line)
    for line in (run / "responses.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
config = json.loads((run / "run_config.json").read_text(encoding="utf-8"))

assert len(responses) == 576
assert config["num_prompts"] == 160
assert config["generation_count_minimum"] == 3
assert config["generation_count_maximum"] == 6
assert config["total_generations"] == 576
print(config["finish_reason_counts"])
```

Download `responses.jsonl`, `activations.npz`, and `run_config.json`, or archive
the complete `results/run_round2` directory.

## 5. Create response-judging packets

The 576 responses are divided into six 96-response batches for each of three
judges:

```powershell
$run = "results/run_round2"

python src/prepare_judging.py `
  --responses "$run/responses.jsonl" `
  --output-dir "$run/judging" `
  --judges 3 `
  --passes 1 `
  --batch-size 96 `
  --min-prompt-gap 2
```

Use `scoring/round2_batch_judge_prompt.md`. Each judge should be a different
frontier model. Keep `private_mapping.jsonl` hidden.

Save every returned batch under the matching filename:

```text
raw_scores/judge_1_pass_1_batch_01.jsonl
...
raw_scores/judge_3_pass_1_batch_06.jsonl
```

The output must contain all seven numerical scores, `response_mode`, and
`brief_reason`.

Aggregate:

```powershell
python src/aggregate_judging.py `
  --mapping "$run/judging/private_mapping.jsonl" `
  --raw-scores-dir "$run/judging/raw_scores" `
  --output-dir "$run/judging/aggregated" `
  --judges 3 `
  --passes 1 `
  --schema round2
```

If `manual_scores.template.jsonl` is created, review only the flagged dimensions,
replace `null` values with adjudicated scores, and save the result as
`manual_scores.jsonl`.

## 6. Annotate prompts separately

```powershell
python src/prepare_prompt_annotations.py `
  --dataset data/round2_ladders.jsonl `
  --output-dir "$run/prompt_annotations" `
  --judges 3
```

Use `scoring/round2_prompt_annotation_prompt.md`, saving outputs as:

```text
prompt_annotations/raw_scores/judge_1.jsonl
prompt_annotations/raw_scores/judge_2.jsonl
prompt_annotations/raw_scores/judge_3.jsonl
```

Aggregate:

```powershell
python src/aggregate_prompt_annotations.py `
  --mapping "$run/prompt_annotations/private_mapping.jsonl" `
  --raw-scores-dir "$run/prompt_annotations/raw_scores" `
  --output-dir "$run/prompt_annotations/aggregated" `
  --judges 3
```

If prompt disagreements remain, fill the generated template and rerun with:

```powershell
  --manual-scores "$run/prompt_annotations/aggregated/manual_scores.jsonl"
```

## 7. Freeze adjudication and run analysis

Do not revise the rubric, split, ridge alpha, or target definitions at this
point.

```powershell
python src/analyze_round2.py `
  --dataset data/round2_ladders.jsonl `
  --responses "$run/responses.jsonl" `
  --activations "$run/activations.npz" `
  --run-config "$run/run_config.json" `
  --judge-files `
    "$run/judging/aggregated/judge_1.jsonl" `
    "$run/judging/aggregated/judge_2.jsonl" `
    "$run/judging/aggregated/judge_3.jsonl" `
  --manual-scores "$run/judging/aggregated/manual_scores.jsonl" `
  --prompt-scores "$run/prompt_annotations/aggregated/prompt_scores.jsonl" `
  --output-dir "$run/analysis"
```

Omit `--manual-scores` only when no cross-judge response dimensions were
flagged.

## 8. Primary outputs

- `round2_summary.json`: locked confirmation result and baselines
- `target_summary.csv`: development-selected layer and confirmation metrics for
  every target
- `development_metrics_by_layer.csv`: development-only layer selection
- `confirmation_predictions.csv`: untouched confirmation predictions
- `confirmation_per_ladder.csv`: fold-level calibration and rank behavior
- `prompt_level_scores.csv`: final prompt-level targets and diagnostics
- `development_probe_curve.png`: development layer-selection curve

The summary includes the text-only confirmation baseline, the
design-condition-adjusted engagement result, and the realized
engagement-withholding correlation. Its headline field is
`primary_activation_over_text`, containing activation-minus-text Spearman and
Pearson deltas, text-minus-activation MAE improvement, and paired
ladder-bootstrap intervals. Always report that comparison before the raw
activation result.

The confirmation set must not be rerun with revised settings and presented as a
fresh confirmatory result. Any post-result modification starts a new protocol
version.
