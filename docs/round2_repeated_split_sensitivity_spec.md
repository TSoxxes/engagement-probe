# Round 2 repeated-split sensitivity analysis specification

**Status:** Post-hoc robustness analysis. This specification was frozen on
2026-08-02 before any repeated-split results were calculated. The analysis
cannot change the outcome of the prespecified Round 2 primary test.

## Question

How sensitive is the observed activation-versus-words result to the particular
30-ladder development / 10-ladder confirmation assignment used in Round 2?

## Data and target

- Use the final 160-prompt Round 2 prompt-level dataset and adjudicated scores.
- Keep all four prompts from a semantic ladder in the same split.
- Predict mean substantive engagement for each prompt.
- Use the existing saved pre-answer activations without alteration.

## Repeated splits

- Random seed: `20260802`.
- Generate 200 unique ladder-level splits.
- Each split uses 30 development ladders and 10 held-out evaluation ladders.
- Each held-out set contains five ladders whose D prompt is a capability-limit
  request and five whose D prompt is a harmful request. This matches the balance
  of the original confirmation set.
- The original frozen split is reported separately and is not included among
  the 200 repeated-split summaries.

## Fitting inside each split

All fitting and selection must be repeated from scratch using only the 30
development ladders in that split.

### Activation predictor

1. For each of the 27 saved activation indices, produce leave-one-ladder-out
   predictions within the 30 development ladders.
2. Select the layer with the highest development Spearman correlation; break
   exact ties by choosing the lower layer index.
3. Fit the standardized ridge predictor on all 30 development ladders at the
   selected layer using the locked Round 2 ridge penalty `alpha = 10`.
4. Predict engagement for the ten held-out ladders.

### Words-only predictor

1. Fit unigram/bigram TF-IDF using only the 30 development ladders, retaining
   terms appearing in at least two development prompts.
2. Fit the locked words-only ridge model with `alpha = 1`.
3. Predict engagement for the ten held-out ladders.

No held-out score, prompt text, or activation may influence layer selection,
feature construction, calibration, or model fitting for its split.

## Saved outcomes

For each split, save:

- held-out activation and words-only Spearman correlations;
- activation-minus-words Spearman difference;
- held-out Pearson correlations and their difference;
- uncalibrated mean absolute errors and activation improvement;
- selected activation layer;
- the ten held-out ladder identifiers; and
- every held-out prompt prediction.

Across the 200 splits, report:

- mean, median, standard deviation, 2.5th/25th/75th/97.5th percentiles, minimum,
  and maximum of the activation-minus-words Spearman difference;
- the fraction of splits with a positive difference;
- where the original `+0.046` result falls in the repeated-split distribution;
- how often each activation layer was selected; and
- cross-fitted aggregate performance obtained by averaging each prompt's
  held-out predictions across the repetitions in which it was evaluated.

## Interpretation constraints

- The distribution across overlapping random splits is a stability diagnostic,
  not a confidence interval, p-value, or new confirmatory test.
- Results will be reported regardless of direction.
- The original prespecified confirmation result remains the primary result.
- Any public-report addition will be labeled post hoc.
