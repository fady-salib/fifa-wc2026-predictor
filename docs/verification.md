# Verification — 2026-10-29

Baseline: `bec8cbf`; **12 existing Rust tests passed** before source edits.
No original Python suite existed. The legacy notebook was read and syntax-
inspected, not executed as a valid evaluation or used to claim an improvement.

Reviewed checks executed locally:

| Check | Result |
|---|---|
| Python unittest suite | 15 passed |
| Rust suite (`cargo test --locked`) | 18 passed |
| Rust formatting | Applied; check required in CI |
| Clippy with `-D warnings` | Passed after fixing comparison-chain lint |
| Chronological evaluation | Completed: 1,859 matches; 60 rejected groups; six-model holdout comparison |
| ML pair matrix export | Completed: 2,256 directed rows; normalized and symmetric |
| Exact Python/Rust parity, Elo input | Five scenarios × 200 tournaments; every team/stage count identical |
| Exact Python/Rust parity, exported ML input | Same five scenarios × 200 tournaments; all counts identical |

Scenarios: baseline, France −100 Elo shock, Brazil knockout-only withdrawal,
Brazil full withdrawal, and Iraq replaced by Italy at explicitly assumed 1900
Elo. Both ports use SplitMix64/Knuth Poisson, seed 42, the same source snapshot
and the same **simplified** ranking/bracket/goal assumptions. Exact agreement
does not prove those shared assumptions correct or official.

Python tests cover future-result invariance, same-day batching, cutoff exclusion,
contradictory records, disjoint periods, test-label nonparticipation in fitting,
probability reversal/normalization, repeatability, stage conservation, invalid
events/matrices, withdrawal/replacement and all-draw knockout behavior. Rust adds
atomic matrix validation, reversed pairs, invalid Elo/shocks and extreme tilt
checks to its original event/RNG/stage tests.

Actual evaluation results and class/calibration details are in `evaluation.json`;
raw exclusions/name mappings are in `data-audit.json`. Elo-only log loss 1.0302
beats calibrated stack 1.0374 in this particular small holdout. No test tuning,
independent World Cup validation or accuracy-improvement claim is made. Test
features update from earlier test-date results; this is rolling one-step-ahead
evaluation, not a frozen 2023–2026 prediction exercise.

Environment: Python 3.12, NumPy 2.2.6, pandas 2.3.3, SciPy 1.16.3, scikit-learn
1.6.1, Rust 1.85.1. SciPy emits L-BFGS-B deprecation warnings through sklearn;
they were not globally suppressed. Rustfmt/clippy initially needed installation;
the first clippy pass found an existing comparison-chain lint, then passed after
the style fix. No GPU, live ratings service or physical hardware was needed.

Notebook module operations were run through the CLI. Full notebook UI/rendering
and the archived heavy ensemble were not executed. Official rules and upstream
CSV provenance remain unverified. No passing remote CI or published PR is
claimed: GitHub writes are blocked by the session's current permissions.
