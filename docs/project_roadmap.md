# Willingness Probe project roadmap

Status updated: 2026-07-27

## Project objective

Test whether a language model's pre-answer hidden state predicts how
substantively it will engage with a prompt, beyond what can already be
predicted from prompt text. The current project makes a prompt-conditioned
behavioral prediction claim. It does not yet claim a universal willingness
representation, causal control mechanism, or reliable distinction between
"cannot answer" and "chooses not to answer."

## Completed work

### Round 1: feasibility pilot

- Generated 96 responses from 24 prompts in six semantic ladders using
  `google/gemma-2-2b-it`.
- Collected prompt-final-token activations at the embedding output and every
  transformer layer.
- Built a blinded repeated-judging and adjudication workflow.
- Found a later-layer rank signal for substantive engagement, alongside a
  serious calibration failure and ceiling-saturated labels.
- Established that repeated generations are useful, but that additional
  independent prompts are generally more valuable than a fourth generation
  everywhere.

See `docs/round_1_report.md`.

### Round 2: design, audit, and behavioral pilot

- Expanded to 40 matched semantic ladders and 160 prompts.
- Locked 30 development ladders and 10 untouched confirmation ladders.
- Split refusal language from actual information withholding and separated
  safety framing, uncertainty, underdetermination, and redirection.
- Added a sparse prompt-text baseline and a primary incremental-activation
  decision rule.
- Passed three external prompt-design reviews and the automated structural
  audit.
- Ran a separate 48-response behavioral pilot and clarified the judging rubric.
- Froze protocol 0.3, dataset hashes, generation settings, and analysis code.

The B manipulation met its prespecified acceptance threshold exactly, without
margin. Round 2 remains valid for its frozen activation-over-text question, but
the pilot should not be described as demonstrating a robust warm-refusal cell.

See:

- `docs/round_2_protocol.md`
- `docs/round_2_audit_response.md`
- `docs/round_2_pilot_report.md`
- `docs/round_2_freeze_manifest.json`

## Current milestone: execute frozen Round 2

### 1. Main generation

Run the frozen 576-generation Kaggle job using
`docs/round_2_workflow.md`. Preserve:

- `responses.jsonl`
- `activations.npz`
- `run_config.json`
- the saved Kaggle notebook version and logs

Do not change a hashed file, prompt, split, generation count, or model setting.

### 2. Pre-judging diagnostics

Before committing the full judge-API budget:

- plot engagement, explicit refusal, and withholding jointly for the 48 Round 2
  pilot responses;
- analyze the 96 Round 1 responses separately using its legacy
  `direct_refusal` score;
- do not pool the two rounds unless Round 1 is rescored under the seven-axis
  Round 2 rubric; and
- estimate the judge-token and dollar budget using the actual Round 2 response
  lengths.

These are exploratory resource-planning diagnostics, not gates on frozen
Round 2.

### 3. Judging and confirmatory analysis

- Use three different judge models with one blinded pass each.
- Lock model versions, prompts, and settings before scoring.
- Either score everything in one locked collection while keeping confirmation
  outputs sealed, or stage by development/confirmation split.
- Never use shuffled packet batch order as a substitute for the dataset split.
- Apply only the written adjudication rules.
- Run the frozen development selection and one-time confirmation analysis.

### 4. Round 2 deliverables

- A concise technical report with the preregistered primary result
- Development and confirmation metrics with uncertainty
- Sparse-text comparison and calibration plots
- A limitations section covering cell imbalance, prompt-form leakage, judge
  dependence, and model-scale limits
- A small, privacy-checked result package containing derived tables, figures,
  run configuration, and provenance hashes

## Parallel exploratory track: behavioral-cell diagnostic

This track informs later design but does not alter Round 2.

1. Define response-level thresholds before plotting:
   low = 0–1 and high = 2–3.
2. Quantify the four engagement/refusal cells on the 48 Round 2 pilot
   responses.
3. Optionally rescore the 96 Round 1 responses under the seven-axis rubric.
4. Design a small targeted pilot for answerable-but-thin responses:
   low explicit refusal, high information withholding, and low substantive
   engagement.
5. Retain matched semantic ladders and a sparse text baseline so cell identity
   is not reducible to benchmark source or prompt wording.

## Round 3: causal and scale-up work

Round 3 should be separately preregistered after Round 2 results are known.

### Causal harness

- Reproduce a published refusal-steering effect as a positive control.
- Include a matched-magnitude random-direction negative control.
- Separate the best predictive probe from the steering direction.
- Run a dose-response sweep with fixed scoring and stopping rules.
- Test whether a direction intended to reduce over-refusal also weakens
  justified safety refusal.

No operationally harmful generations should be released.

### Stronger ground truth

For claims about "will not" versus "cannot," use tasks with externally
established capability:

- known-answer or known-capability task sets;
- controlled system-prompt underperformance;
- controlled fine-tuned sandbagging; or
- held-out behavioral capability validation.

Output-judged engagement remains appropriate for predicting output behavior,
but it is not by itself ground truth about latent capability or intent.

### Model scaling

Start with a small cross-model pilot rather than assuming a scale threshold:

- the current 2B model for continuity;
- at least one 7–9B instruction-tuned model;
- if affordable, a second model family or frontier API model.

Use the same targeted ladders and rubric across models. Scale the prompt count
only after confirming that the rare behavioral cells are populated.

## Publication plan

1. Publish an honest Round 1/2 project report before beginning a large third
   redesign.
2. Release the protocol, audit record, derived findings, and exact provenance.
3. Clearly separate confirmatory Round 2 results from exploratory behavioral
   cells and later causal work.
4. Complete a cited prior-art review covering activation probes, refusal
   directions, concept erasure, sandbagging, XSTest, OR-Bench, CoCoNot,
   HarmBench, and related engagement/over-refusal evaluations.

The original Engagement Probe Review Brief should be archived beside
`docs/design_review_response.md` so readers can assess the response fairly.

## Funding roadmap

### Immediate BlueDot request

BlueDot Impact's Rapid Grants program currently supports concrete AI-safety
projects, including compute, API credits, research access, and project tooling.
The public program page lists grants from $50 to $10,000:

- Program and application route:
  https://bluedot.org/programs/rapid-grants
- Technical AI Safety Project Sprint form for eligible current or past
  participants:
  https://airtable.com/appMVNtdBtvtJvu5E/pag9G3oF4DYAyassX/form

Links verified on 2026-07-27.

The first request should fund a specific bottleneck rather than general project
support:

1. three-model API judging for Round 2;
2. prompt annotation and any required blinded adjudication calls;
3. a small 7–9B cross-model pilot;
4. limited compute/API costs for the positive-control steering harness; and
5. result hosting or research tooling if directly required.

The application should include an itemized token and compute estimate. Current
free Kaggle capacity means GPU spending is not the primary Round 2 bottleneck;
that conclusion should be revisited for larger-model or steering experiments.

### Evidence to include

- Round 1 completed findings
- Passed Round 2 behavioral pilot
- Frozen protocol and recorded hashes
- Exact next experiment and decision rule
- Existing code and reproducible workflow
- Requested amount tied to judge calls, model scale, and concrete deliverables
- A short explanation of how the work improves evaluation integrity without
  claiming that internal probes uniquely reveal model intent

## Change control

Round 2's hashed protocol artifacts remain frozen. New diagnostics,
documentation, automation, or future protocols may be added, but changing a
hashed Round 2 file requires a new protocol version and a new confirmation set.

The immediate priority is execution, not another redesign.
