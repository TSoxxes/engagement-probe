# Willingness Probe project roadmap

Status updated: 2026-08-03

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
- Completed the frozen 576-generation Kaggle run and preserved responses,
  activations, configuration, and run provenance.
- Completed three-model blinded response judging: 1,728 judge-case ratings in
  total, followed by manual adjudication of all 209 flagged responses.
- Recorded one narrow judge-model deviation: Opus 4.6 scored one case that the
  assigned Opus 4.8 web product refused to classify. That case received
  mandatory manual review and is designated for an exclusion sensitivity
  check.
- Completed three-model prompt-harmfulness annotation and adjudication.
- Completed the locked development/confirmation analysis. The activation probe
  achieved confirmation Spearman 0.868 versus 0.822 for sparse prompt text, but
  the +0.046 difference had a ladder-bootstrap interval crossing zero and did
  not pass the preregistered primary rule.
- Completed a separately specified post-hoc repeated-split sensitivity analysis.
  All 200 alternative splits favored activations; the median advantage was
  +0.077, and the original result was at the 16.5th percentile. This supports
  split stability but does not replace the frozen primary result.

The B manipulation met its prespecified acceptance threshold exactly, without
margin. Round 2 remains valid for its frozen activation-over-text question, but
the pilot should not be described as demonstrating a robust warm-refusal cell.

See:

- `docs/round_2_protocol.md`
- `docs/round_2_audit_response.md`
- `docs/round_2_pilot_report.md`
- `docs/round_2_freeze_manifest.json`

## Current milestone: finish the public Round 2 report and prepare the next preregistration

### 1. Main generation — complete

The frozen 576-generation Kaggle job is complete. Preserve:

- `responses.jsonl`
- `activations.npz`
- `run_config.json`
- the saved Kaggle notebook version and logs

Do not change a hashed file, prompt, split, generation count, or model setting.

### 2. Pre-judging diagnostics — complete

The exploratory diagnostic was completed before full judging and is recorded
under `results/round2_main/prejudge_diagnostic/`. It was used for resource
planning, not as a gate or modification to frozen Round 2.

### 3. Response judging and adjudication — complete

- Three independent web-chat judge families each scored all 576 responses.
- All raw ratings, mappings, source transcripts, and provenance are preserved
  under `results/round2_main/run_round2/judging_web/`.
- Every response dimension with a cross-judge range greater than one point was
  manually reviewed. The finalized scores are in
  `judging_web/aggregated/manual_scores.jsonl`.

### 4. Prompt-only annotation — complete

Three blinded judge families scored all 160 prompts for harmfulness without
seeing model responses. Six prompts were flagged and adjudicated; none remained
unresolved. Artifacts are preserved under
`results/round2_main/run_round2/prompt_annotations/`.

### 5. Locked confirmatory analysis — complete

The one-time confirmation analysis was run exactly as specified in
`docs/round_2_workflow.md`. The preregistered Spearman decision rule was not met;
Pearson, absolute error, condition-adjusted, and repeated-split diagnostics
favored activations. Confirmatory and post-hoc results remain explicitly
separated.

The repeated-split procedure and results are documented in:

- `docs/round2_repeated_split_sensitivity_spec.md`
- `docs/round2_repeated_split_sensitivity_report.md`
- `results/round2_main/run_round2/analysis/repeated_split_sensitivity/`

### 6. Round 2 deliverables — in final review

- A concise technical report with the preregistered primary result
- Development and confirmation metrics with uncertainty
- Sparse-text comparison and calibration plots
- A limitations section covering cell imbalance, prompt-form leakage, judge
  dependence, and model-scale limits
- A small, privacy-checked result package containing derived tables, figures,
  run configuration, and provenance hashes

The technical summary is in `docs/round_2_results_summary.md`. The public HTML
report is undergoing final wording review before the revision markings are
removed.

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

- Pilot a published refusal-direction procedure on the same open-weight model
  selected for the causal follow-up. Reproduce existing results when they cover
  the exact checkpoint; otherwise derive the direction on that model using the
  published procedure. Require the expected refusal change on separate held-out
  prompts before proceeding to the engagement intervention.
- Include a matched-magnitude random-direction negative control.
- Separate the best predictive probe from the steering direction.
- Run a dose-response sweep with fixed scoring and stopping rules.
- Test whether a direction intended to reduce over-refusal also weakens
  justified safety refusal.

No operationally harmful generations should be released.

### Stable prediction and evaluation

- Keep a locked confirmation set as the primary test.
- Use more than 40 semantic ladders, prioritizing a larger independent
  confirmation set. Set the exact count prospectively from power analysis and
  the scoring and compute budget.
- Preregister grouped repeated-split analysis as a supporting robustness check,
  rebuilding both fitted predictors within every development split.
- Use a separate pilot with different semantic ladders to evaluate every layer
  of a new model. Before examining the pilot results, specify the band width and
  a rule for finding a contiguous region that remains strong across grouped
  pilot resamples. Freeze the selected layers and averaging rule before the
  main study, and use that neighboring-layer average as the primary
  representation. Treat a regularized combination of a few widely separated
  layers as a prespecified secondary comparison; it cannot be promoted to the
  primary result after outcomes are known. In the Round 2 sensitivity analysis,
  layer 25 was chosen in only 28 of 200 splits, while layer 18 was chosen in 60.
- Prespecify a regularization sensitivity check. Round 2 fixed ridge penalties
  in advance and never varied them: alpha 10.0 for the activation arm and 1.0
  for the sparse text arm. Fixing them before analysis avoided tuning toward a
  positive result, but leaves open whether the activation-minus-text delta holds
  across other reasonable penalties, and whether the arms' different values
  affected the comparison. The check re-runs development-only layer selection
  and fitting across a range of penalties for both arms and reports whether the
  delta is stable; confirmation data is not involved. Deferred from the Round 2
  revision on 2026-08-20 as out of scope for a published report, not because the
  question was resolved.

- Compare a prompt-only predictor with an augmented predictor that uses
  activations to predict what the prompt-only estimate misses. Use the
  augmented-minus-prompt-only ranking improvement after accounting for broad
  prompt type as the single primary endpoint; retain pooled and per-type results
  and an activation-only linear probe as prespecified secondary analyses.

### Stronger ground truth

For claims about "will not" versus "cannot," first run a separate capability
pilot. Use safe, fully specified tasks with objective answer keys, repeat them,
and retain only tasks the model solves reliably. The main study can then use
matched variants that introduce missing information, an impossible action, or
a safety constraint while preserving the underlying knowledge or skill. For
safety cases, verify that skill with a benign matched task. Other possible
capability controls include:

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

1. three-model API judging for the next controlled prompt set;
2. checklist drafting, prompt annotation, and independent LLM adjudication;
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

The immediate priority is finishing the clean public report, then freezing the
next study's prompt set, checklist rubric, predictor comparison, split-stability
analysis, and intervention controls before new data are generated.
