# Round 1 pilot report

## Research question

Can the final prompt-token hidden state of an instruction-tuned language model
predict how substantively the model will engage with a prompt before it begins
generating its answer?

## Model and data

- Model: `google/gemma-2-2b-it`
- Prompts: 24
- Independent semantic ladders: 6
- Rungs per ladder: 4
- Sampled responses per prompt: 4
- Maximum new tokens: 600
- Recorded representation: final prompt token at the embedding output and every
  transformer layer
- Activation shape: `[24, 27, 2304]`

The six ladders covered household safety, medication safety, medical advice,
employment advice, travel recommendations, and software troubleshooting. The
four responses for a prompt shared one deterministic pre-generation activation
but supplied repeated behavioral observations.

## Scoring

Two independent model judges scored every response on six 0–3 dimensions:

1. substantive engagement;
2. direct refusal;
3. safety caution;
4. epistemic uncertainty;
5. professional redirection; and
6. prompt harmfulness.

Each judge scored three position-balanced permutations. No score dimension
varied by more than one point within either judge. Twelve dimensions across
eleven responses differed by more than one point between judges and were
manually adjudicated under the written rubric.

The primary prompt-level target was the mean substantive-engagement score across
the four generations.

## Primary analysis

A standardized ridge probe was fit separately at each hidden-state index.
Evaluation used leave-one-ladder-out prediction: five ladders trained the probe,
and the sixth was predicted without appearing in training. This was repeated
until every ladder had been held out.

The strongest layer selected on the pilot data was hidden-state index 19:

| Metric | Value |
|---|---:|
| Held-out Spearman correlation | 0.615 |
| Held-out Pearson correlation | 0.118 |
| Mean absolute error | 0.318 |
| Response-length Spearman baseline | 0.035 |
| Token-cap Spearman baseline | -0.175 |
| Shuffled-label Spearman, mean | -0.038 |
| Shuffled-label Spearman, standard deviation | 0.161 |

The positive rank association persisted across many later layers rather than
appearing only at one isolated index. It was not explained by average response
length or by the proportion of generations that hit the token cap.

## Calibration failure and label imbalance

The headline rank correlation overstated practical refusal detection. Eighteen
of 24 prompts received an engagement score of exactly 3. Only two prompts scored
below 2, and both belonged to the travel ladder.

When that entire ladder was held out, the probe had no training examples below
2.5. It ranked the two low-engagement travel prompts slightly below the
high-engagement travel prompts, but predicted scores near 2.86 for true scores
of 1.25 and 0. This explains the combination of a moderately strong Spearman
correlation and a weak Pearson correlation.

The defensible primary conclusion is therefore:

> Later hidden states contained a decodable signal associated with relative
> substantive engagement, but the pilot did not establish a calibrated,
> refusal-independent, or out-of-domain willingness measure.

## Generation variability

Substantive engagement varied across the four generations for 5 of 24 prompts.
About 15% of total engagement variance was within prompts and 85% was between
prompts. Two prompts showed large behavioral changes:

- `safety_ibuprofen_2`: engagement scores of 1, 3, 3, and 3;
- `uncertainty_travel_3`: approximately 2.33, 0.17, 0.17, and 2.33.

Direct refusal was less stable: 9 of 24 prompts varied, and approximately 28% of
its variance was within prompts. Repeated generations were therefore useful,
but a fourth generation everywhere offered less information than additional
independent prompts and ladders would.

## Exploratory analyses

These analyses were conducted after inspecting the primary result and are not
confirmatory.

- Human engagement and refusal scores were strongly related (Pearson -0.905;
  Spearman -0.624).
- The layer-19 engagement predictions were almost unrelated to prompt-level
  refusal scores (Spearman -0.061).
- Separate post-hoc probes found best held-out Spearman correlations of 0.615
  for direct refusal, 0.415 for epistemic uncertainty, and 0.865 for safety
  caution, with different selected layer indices.

These results suggest that the representation contains multiple prompt- and
response-planning signals, but the pilot cannot isolate engagement from topic,
risk recognition, safety framing, or refusal behavior.

## Limitations

- Only six independent ladders were available.
- Low engagement was concentrated in one ladder.
- Layer selection and evaluation used the same small dataset.
- The rubric combined explicit refusal language with actual information
  withholding.
- Epistemic uncertainty combined expressed doubt with underdetermination.
- Safety caution did not distinguish requested risk information from additional
  safety framing.
- Twenty-three of 96 responses reached the 600-token limit.
- Exploratory target comparisons were post-hoc.

## Round 2 motivation

Round 2 is designed to increase the number of independent ladders, separate
refusal language from information withholding, distinguish expressed
uncertainty from underdetermination, use a locked confirmation split, and test
whether engagement remains predictable after refusal-related behavior is
accounted for.

Primary artifacts for the pilot are under
`results/archive/willingness-probe-run-003/run_003/analysis/`.
