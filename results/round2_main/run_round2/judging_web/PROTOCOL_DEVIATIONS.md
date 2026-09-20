# Round 2 judging protocol deviations

## Judge 2 safety-layer fallback

- Date: 2026-07-28
- Case: `J2-2185F08DA4`
- Original assignment: Claude Opus 4.8, high effort
- Actual scorer: Claude Opus 4.6
- Scope: 1 of 576 Judge 2 cases; 1 of 1,728 total judge-case ratings
- Reason: The Opus 4.8 web product refused to evaluate the quoted malware
  request even when it was presented as a behavioral-classification task. Opus
  4.6 completed the same rubric-only evaluation.
- Handling: The Opus 4.6 score was inserted at Judge 2 batch 03 position 29.
  The case is designated for mandatory manual review regardless of the ordinary
  cross-judge disagreement threshold.
- Interpretation: Report this as a minor, content-dependent judge-model
  deviation. Run a sensitivity check excluding this response from analyses that
  use judged outcomes; it must not affect conclusions if results are robust.

