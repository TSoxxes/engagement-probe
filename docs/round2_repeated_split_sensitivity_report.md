# Round 2 repeated-split sensitivity report

## Status

This is a post-hoc robustness analysis. Its procedure was frozen in
`docs/round2_repeated_split_sensitivity_spec.md` before any alternative-split
results were calculated. It does not replace the prespecified Round 2 primary
test.

## Question

Did the observed activation-versus-words result depend unusually heavily on the
particular 30-ladder development / 10-ladder confirmation assignment?

## Procedure

The complete analysis was repeated across 200 unique alternative splits. Each
split held out ten whole ladders, balanced between five capability-limit and
five harmful-request D prompts. Using only that split's 30 development ladders,
the analysis:

1. reselected the activation layer by leave-one-ladder-out development
   performance;
2. refit the activation probe;
3. rebuilt the words-only vocabulary and predictor; and
4. evaluated both predictors on the ten held-out ladders.

The original frozen split was also reconstructed as an implementation check.

## Results

The reconstructed original split exactly reproduced the saved primary values:

- activation Spearman: **0.868**;
- words-only Spearman: **0.822**; and
- activation advantage: **+0.046**.

Across the 200 alternative splits:

- **200 of 200** activation-minus-words Spearman differences were positive;
- mean advantage: **+0.081**;
- median advantage: **+0.077**;
- minimum and maximum: **+0.005 to +0.167**;
- middle 95% of split results: **+0.021 to +0.151**; and
- the original +0.046 result was at the **16.5th percentile**, meaning 83.5% of
  the alternative splits produced a larger activation advantage.

The same directional result appeared on the two other saved performance
measures. Activations had the higher Pearson correlation in **200 of 200**
splits (median advantage **+0.107**) and the lower uncalibrated mean absolute
error in **200 of 200** splits (median improvement **0.278 points** on the 0–3
engagement scale).

Each prompt appeared in held-out evaluation an average of 50 times. Averaging
its held-out predictions across those repetitions produced:

| Predictor | Cross-fitted Spearman |
|---|---:|
| Activations | 0.928 |
| Words only | 0.848 |
| Difference | **+0.080** |

This cross-fitted aggregate is ensemble-like rather than a direct estimate for
the frozen single-layer probe. Averaging roughly 50 held-out predictions per
prompt reduces prediction noise, and the contributing activation probes often
use different selected layers. The 0.928 / 0.848 correlations therefore should
not be placed beside the original single-split correlations without this
qualification.

Layer selection was not concentrated on one unique layer. Selected activation
indices ranged from 17 to 25. The frozen layer 25 won 28 of 200 times (14%),
while layer 18 won 60 times (30%):

| Layer | Selections |
|---:|---:|
| 17 | 3 |
| 18 | 60 |
| 19 | 23 |
| 20 | 41 |
| 21 | 1 |
| 22 | 10 |
| 23 | 33 |
| 24 | 1 |
| 25 | 28 |

## Interpretation

The original split does not appear to have been unusually favorable to the
activation probe. If anything, its observed advantage was smaller than most
alternative-split estimates. The positive activation advantage was stable
across the 200 tested assignments, while the exact size of the advantage and
the selected late layer varied.

This strengthens the case that the qualitative activation advantage was not an
accident of the original ladder assignment. It does **not** convert the primary
test into a confirmatory success: the analysis was designed after the primary
result was known, the random splits overlap, and all repetitions reuse the same
40 ladders. Their distribution is therefore a stability diagnostic rather than
a confidence interval, p-value, or independent replication.

## Saved artifacts

Machine-readable outputs are under
`results/round2_main/run_round2/analysis/repeated_split_sensitivity/`:

- `summary.json`
- `split_metrics.csv`
- `heldout_predictions.csv`
- `prompt_crossfitted_predictions.csv`
- `layer_selection_counts.csv`
- `original_split_predictions.csv`
