# Round 2 external-audit response

Status: behavioral pilot passed on 2026-07-26; the frozen main run is
authorized.

This document records the disposition of three independent review passes
received on 2026-07-25.

## Changes accepted and implemented

| Finding | Disposition |
|---|---|
| Rung identified design cell | Fixed with a Latin square. Every rung contains ten A, B, C, and D prompts. |
| D pooled capability and harmful refusal | Fixed. `D_capability` and `D_harmful_request` are separate five-level conditions, balanced across splits and reported separately. |
| B scripted refusal language | Fixed. No B prompt contains `Without...` or `I know you cannot...`. |
| B became an ordinary A-like request after disclaimer removal | Fixed in protocol 0.3 prompts. B now requests a personalized assessment or unavailable context plus useful explanation or next steps. The separate pilot must verify B-A refusal lift and retained engagement. |
| Mechanical paraphrasing produced malformed prompts | Fixed by deleting the mechanical prefix transformation. All prompts are authored complete sentences. The injected phrase audit currently reports zero. |
| Prompt IDs drifted when rung was permuted | Fixed by minting `r2v2_<ladder>_<cell>` IDs. Rung remains separate metadata. Future incompatible prompt changes require a new ID version. |
| Repeat budget excluded B | Fixed. Six-generation prompts are allocated B=16, C=8, D=8. |
| Several C prompts remained answerable or likely to elicit refusal | Medication, investing, and cryptocurrency C prompts were replaced with missing-record questions that support clarification without a safety or policy refusal. Earlier flagged C prompts remain revised. |
| Fire C used a live-emergency frame | Fixed. It now asks retrospectively about a faint burning smell, retaining underdetermination without an active emergency. |
| Skin and exercise-injury capability prompts bundled diagnosis or interpretation | Fixed. They now request only an unavailable appointment/upload or record-transfer action. |
| Harmful prompts were unnecessarily operational | The specifically flagged harmful prompts remain softened. |
| Confirmation prompts were template siblings | Confirmation wording was re-themed. Current maximum same-cell development/confirmation cosine is 0.494, below 0.55. |
| Engagement and withholding were incorrectly treated as independent nuisance and target variables | Fixed. Withholding is not regressed out. Their development correlation is reported with a 0.90 identifiability threshold. |
| Raw activation correlation was vulnerable to prompt-form shortcuts | Fixed by changing the claim and primary analysis. Protocol 0.3 studies prompt-conditioned behavior. Its primary estimand is confirmation activation Spearman minus sparse-text Spearman, with a 2,000-resample paired ladder-bootstrap interval. |

## Protocol-copy discrepancy

The second review stated that the supplied protocol still used four cells,
regressed out withholding, omitted the text baseline, and lacked a version
bump. Those statements did not match the local protocol at the time: protocol
0.2 already contained five conditions, excluded withholding from the nuisance
model, and specified the sparse-text baseline. No substantive change was made
to reproduce the obsolete language.

Protocol 0.3 now makes the activation-over-text delta the explicit primary
estimand rather than merely requiring that the two results be shown together.
The repository contains only the canonical `round_2_protocol.md` and
`round_2_audit_response.md`; there are no hashed duplicate documents. The
reviewer's stale plain-name copies were created by its upload workspace. Future
reviews should use a fresh conversation or remove older uploads before adding
the current files.

## Current automated audit

The canonical machine-readable record is
`data/round2_prompt_audit.json`.

| Check | Current | Interpretation |
|---|---:|---|
| Rung best-cell accuracy | 0.250 | structural pass |
| Scripted B disclaimers | 0 | structural pass |
| Injected mechanical markers | 0 | structural pass |
| Maximum same-cell split cosine | 0.494 | structural pass; threshold 0.55 |
| Text-only cell accuracy | 0.975 | retained diagnostic; confirms prompt type is highly visible |
| Surface-only D-subtype accuracy | 0.950 | retained diagnostic; prohibits a representation-level subtype claim |

The last two values are not being disguised or optimized to chance. Prompt
text must encode the requested behavior. They motivate the prompt-conditioned
framing and activation-over-text primary comparison. The audit tool treats them
as diagnostic warnings; `--enforce-diagnostic-thresholds` is available if a
future protocol chooses to reinstate them as gates.

## Behavioral pilot

`data/round2_pilot_prompts.jsonl` contains 16 prompts and produces 48 responses.
Those prompts are excluded from the main dataset. The pilot showed:

1. B minus A explicit-refusal score of at least 0.25;
2. mean B substantive engagement of at least 2.0;
3. at least two thirds of C responses below engagement 2; and
4. acceptable judge separation of engagement, safety framing,
   underdetermination, and professional redirection.

All four gates passed after manual adjudication and a targeted rubric recheck.
The exact results are recorded in `docs/round_2_pilot_report.md`; the main run
is authorized against the frozen protocol and rubric.

The pilot contains four ladders and exactly four prompts per condition
(A=4, B=4, C=4, D=4). It covers personalized health/legal assessment,
account-security access, and physical-safety boundaries. Its C prompts use
missing information rather than policy refusal, and its D prompts include two
capability and two harmful-request cases.
