# Engagement Probe

**Can a language model's internal state, read after it sees a prompt but before
it writes a single token, predict how fully it will answer?**

Most evaluations score what a model chose to show us. A refusal check only
catches the cases where the model announces itself. This project measures
something a refusal check walks straight past: how *substantively* a request was
met, graded on a 0–3 scale, predicted from the model's pre-answer activations.

In this study's own 576 responses, **55 gave little of substance and withheld
most of what was asked while using little or no refusal language** — and none of
those 55 were on harmful prompts. They were ordinary requests, quietly
under-answered.

📄 **[Read the short report](https://tsoxxes.github.io/willingness-probe-report/)**
(~15 minutes, written for a general technical audience) ·
[PDF](output/pdf/willingness_probe_report.pdf)

---

## The result, stated honestly

Two preregistered rounds on `google/gemma-2-2b-it`. Round 2 locked the protocol,
dataset hashes, rubric, and analysis code — with recorded SHA-256 checksums —
before a single main-study response was generated. Ten of 40 subject areas were
held back untouched until a one-time confirmation analysis.

On that untouched confirmation set:

| Predictor | Spearman | Pearson | MAE (0–3 scale) |
|---|---:|---:|---:|
| **Pre-answer activations** (layer 25) | **0.868** | **0.949** | **0.283** |
| Sparse prompt text (TF-IDF) | 0.822 | 0.802 | 0.615 |
| Judge-rated refusal language + safety framing | 0.827 | 0.790 | 0.533 |
| Response length | 0.766 | 0.910 | 0.367 |
| Token-cap rate | 0.190 | 0.236 | 1.047 |

The preregistered primary test was whether activations *beat prompt wording* on
rank correlation. The observed advantage was **+0.046**, with a paired
ladder-bootstrap interval of **−0.014 to +0.130**. That interval crosses zero,
so under the rule committed to in advance, **the primary result is
inconclusive** — and that is how it is reported, here and in the public report.

Two prespecified secondary measures clearly favoured activations (Pearson +0.147,
interval +0.082 to +0.228; MAE improvement +0.332, interval +0.241 to +0.428),
and a post-hoc 200-split stability check found all 200 splits positive with a
median advantage of +0.077. Neither rescues the primary test, and neither is
presented as doing so.

### What this does not show

- No universal or model-independent engagement representation
- No causal mechanism — a predictive probe cannot show the direction *does*
  anything
- No independence from prompt wording: sparse text identifies the intended
  design cell with 97.5% leave-one-ladder-out accuracy
- No reliable separation of "cannot answer" from "chooses not to answer"
- Engagement and withholding scores came out near mirror images (r = −0.95), and
  longer answers scored higher, so "said more" and "said it at greater length"
  are not cleanly separated

That last point is the most load-bearing weakness, and it is what the next study
is designed to settle first. See [`docs/project_roadmap.md`](docs/project_roadmap.md).

## How it was measured

For every prompt, the model's representation of the **final prompt token** is
saved at 27 internal indices — the embedding output and every transformer layer
— each 2,304 values wide, recorded *before the first answer token exists*. A
linear ridge probe predicts a prompt-level behavioural score from those values,
evaluated leave-one-subject-area-out.

Because the snapshot is taken before generation, it is identical for every
sampled answer to a prompt. The analysis therefore averages judge scores and
response length across generations before fitting. It does not pretend a
pre-generation snapshot can explain random variation between sampled answers.

Labels come from **three independent LLM judge families**, 1,728 blinded ratings,
with opaque case IDs and three independently ordered passes per judge. Every
dimension where judges disagreed by more than one point went to documented
case-by-case adjudication — 209 responses reviewed. Deviations are recorded in
[`results/round2_main/run_round2/judging_web/PROTOCOL_DEVIATIONS.md`](results/round2_main/run_round2/judging_web/PROTOCOL_DEVIATIONS.md).

Every label in this project comes from LLM judges. Human-rater validation of
those judges has not been done, and it is the most common objection the work has
received.

## Try it in two minutes

```bash
pip install numpy pandas matplotlib
python -m unittest discover -s tests
```

24 tests, no model download, no GPU. To exercise the full analysis path on tiny
synthetic activations:

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

To rebuild and re-audit the Round 2 prompt set from its generating source:

```bash
python src/build_round2_dataset.py
python src/audit_round2_prompts.py
```

## Inspect the actual data

The Round 2 responses, judge scores, and analysis outputs are committed under
[`results/`](results/README.md) — 576 responses, 1,728 ratings, and every
reported number. That directory's README explains what is there, what is not
(the 18 MB activation tensor, available on request), and how to check the
headline figures yourself.

## Repository layout

```text
data/           Prompt sets. round2_ladders.jsonl is the frozen Round 2 set.
docs/           Protocol, reports, audit responses, roadmap. Start with
                round_2_results_summary.md.
report/         Master HTML short report (v3 is current).
output/pdf/     Master PDF of the short report.
results/        Released Round 2 artifacts — see results/README.md.
scoring/        Judge rubrics and the judging workflow.
src/            Generation, judging aggregation, and analysis entry points.
scripts/        Publishing helper for the separate report site.
tests/          Regression tests. No model download required.
```

### Key documents

| Document | What it is |
|---|---|
| [`docs/round_2_results_summary.md`](docs/round_2_results_summary.md) | The technical write-up. Read this first. |
| [`docs/round_2_protocol.md`](docs/round_2_protocol.md) | The preregistered protocol |
| [`docs/round_2_freeze_manifest.json`](docs/round_2_freeze_manifest.json) | SHA-256 checksums recorded at freeze time |
| [`docs/round_2_audit_response.md`](docs/round_2_audit_response.md) | Response to external design review |
| [`docs/design_review_response.md`](docs/design_review_response.md) | Response to the review brief |
| [`docs/round2_repeated_split_sensitivity_report.md`](docs/round2_repeated_split_sensitivity_report.md) | The 200-split stability analysis |
| [`docs/round_1_report.md`](docs/round_1_report.md) | Round 1, including its ceiling-saturation failure |
| [`docs/project_roadmap.md`](docs/project_roadmap.md) | What is next and why |

### Main analysis entry points

| Script | Purpose |
|---|---|
| `src/generate.py` | Generation + all-layer activation capture |
| `src/analyze_round2.py` | The locked confirmatory analysis |
| `src/analyze_round2_multisplit.py` | Repeated-split sensitivity |
| `src/analyze_round2_length.py` | Response-length confound diagnostics |
| `src/analyze_round2_reliability.py` | Inter-judge reliability |
| `src/prepare_judging.py` / `src/aggregate_judging.py` | Blinded judging pipeline |

## Reproducing the generation run

Generation needs a GPU and gated model access; everything else does not.

Gemma-2-2B-it is gated on Hugging Face: accept the model terms, create a read
token, and add it in Kaggle as a secret named `HF_TOKEN`. Turn on a T4 GPU and
internet access. Then, in a notebook cell, expose the secret without printing it:

```python
import os
from kaggle_secrets import UserSecretsClient
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
```

Install dependencies and run:

```python
!pip install -q -r requirements.txt
!python src/generate.py \
    --dataset data/round2_ladders.jsonl \
    --output-dir results/run_round2 \
    --generation-count-field generation_count \
    --num-generations 3 \
    --max-new-tokens 800 \
    --temperature 0.7
```

`--generation-count-field` makes each prompt use the prespecified
`generation_count` from the frozen dataset (3 for most, 6 for the
variability subset); `--num-generations` is the fallback for prompts without
one. Omitting the field flag silently gives every prompt the same count and
will not reproduce the published run.

The script uses float16 on a T4 and bfloat16 where natively supported, and stops
on Gemma's complete EOS configuration including `<end_of_turn>`. It writes
`responses.jsonl`, `activations.npz`, and `run_config.json`. Every response
records `finish_reason` and `hit_token_cap`; 11 of 576 hit the 800-token ceiling
in the main run. The activation file is shaped
`[prompts, embedding + transformer layers, hidden size]`.

Scoring and analysis then follow [`docs/round_2_workflow.md`](docs/round_2_workflow.md)
and [`scoring/JUDGING_WORKFLOW.md`](scoring/JUDGING_WORKFLOW.md). The private
case-ID mapping and all experimental metadata must stay hidden from judges.

## How this project was built

This is a human-directed, LLM-assisted project, and the public report says so
explicitly — including which parts of the protocol, rubric, and interpretation
were human decisions and which drafting and coding was model-assisted. Three
separate model families supplied the judge scores.

## License

[MIT](LICENSE). Generations under `results/` are outputs of
`google/gemma-2-2b-it` and are also subject to the
[Gemma Terms of Use](https://ai.google.dev/gemma/terms).
