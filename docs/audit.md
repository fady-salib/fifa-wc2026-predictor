# Audit — 2026-10-29

Base `bec8cbf` on `main`. All 13 original tracked files were inspected. All
notebook code cells were extracted/read before changes, the source CSV was
profiled, and the existing Rust suite ran before edits (12 passed). There were
no Python tests, root ignore file or CI workflow.

## Original file-by-file findings

| File | Role and finding | Treatment |
|---|---|---|
| `FIFA_World_Cup_2026.ipynb` | 29-cell, 1,103-code-line training/prediction/simulation experiment; overlapping validation, future strength/H2H/rest lookups, shuffled OOF, in-sample upper layers and no-op calibration wrapper | Preserve exact original in `notebooks/legacy/`; replace entry notebook with tested module calls and honest results |
| `data/international_football_2018_present.csv` | 3,866 rows / 60 columns; historical Elo/xG entirely absent, malformed name codes, contradictory mirrors and duplicate groups | Raw file unchanged; validate/quarantine whole groups; no guessed outcome repairs |
| `data/README.md` | Incorrectly says CSV absent; no upstream provenance/license | Correct inventory, integrity procedure and unresolved source rights |
| `requirements.txt` | Unpinned notebook packages; unnecessary heavy models for default reviewed workflow | Pin tested core versions; separate optional legacy dependencies |
| `rust/Cargo.toml` | Dependency-free simulator metadata/profile | Preserve; commit generated Cargo.lock for application builds |
| `rust/src/data.rs` | Hard-coded 12 groups / 48 Elo values without dated source | Preserve snapshot; Python copy and port checks; explicitly unverified scenario inputs |
| `rust/src/rng.rs` | Deterministic SplitMix64 and small-rate Knuth Poisson | Preserve algorithm; identical Python port |
| `rust/src/model.rs` | Elo fallback, matrix and event tilt; accepts NaN/negative/zero/duplicate matrix rows; partial mutation on failed load; extreme tilt changes supposedly fixed draw probability | Strict atomic load and reverse-pair consistency/fallback; finite Elo validation; decisive-odds tilt preserves draw |
| `rust/src/events.rs` | CLI grammar for shock/withdraw/replace/KO events; empty/nonfinite shocks allowed | Reject invalid team/delta input |
| `rust/src/sim.rs` | Group ranking, toy R32, event simulation/stage counts; multiple withdrawals can advance withdrawn teams; all-draw KO always selects B | Reject unsupported multiple vacancies; fair-coin all-draw fallback; retain and label toy rules |
| `rust/src/main.rs` | CLI output, baseline/scenario comparison; unsafe rating inputs; rounded output only | Validated Elo, explicit toy warning, raw CSV stage-count export |
| `README.md` | Correctly admits several leaks but still presents advanced ensemble as prediction model; simplified bracket and wrong independence implication for same-seed comparisons | Rewrite around evaluated protocol, baseline comparison, data evidence and unresolved official rules |
| `LICENSE` | MIT, 2026 fady botrous | Retain; does not establish third-party CSV rights |

## Methodological defects, in priority order

1. **P1 — Invalid labels/data:** some reciprocal records assign both teams the
   same non-draw result/score; ambiguous duplicate groups also exist. Consistency
   cannot establish the true score, so entire groups are quarantined with line
   references. 1,859 matches retained; 60 groups rejected.
2. **P1 — Test overlap:** original train dates before 2026 include the purported
   2023–2025 validation window. Revised base/meta/calibration/test periods are
   disjoint and respect whole match dates.
3. **P1 — Future-feature leakage:** live/current Elo/rank/xG applied to history;
   H2H uses all outcomes including target/future; rest uses the final match date
   and a single 2026 reference. Global imputation uses future rows. Revised
   features use only earlier-date outcomes and no unverified external values.
4. **P1 — Meta/calibration leakage:** shuffled stratified OOF allows future dates
   and mirrored matches across folds. L3 consumes in-sample L2 predictions;
   FourLayerStack.fit is a no-op over already-fitted globals. Revised default is
   a smaller held-out stack with separate 2023 meta and 2024 calibration blocks.
5. **P2 — Other legacy feature errors:** opponent-form rename does not invert
   outcomes; missing labels become draws; `tournament` absent in a CSV using
   `competition`, making specialist/context logic degenerate. Missing values are
   not fabricated in the new pipeline; unnecessary features/specialist removed
   from the default, preserved in the labeled archive.
6. **P2 — Probability/event defects:** invalid matrices and NaN ratings accepted;
   advertised fixed-draw tilt changes draw at extremes; all-draw knockout picks
   B; multi-withdrawal vacancy behavior unspecified. Guarded/fixed in both ports.
7. **P2 — Simulation validity:** simplified R32, incomplete tiebreaks and forced
   score/outcome reconciliation are not official/probabilistic goal models.
   Retained as toy assumptions and exposed in every CLI/README, not silently
   upgraded to an unverified official implementation.
8. **P3 — Reproducibility:** notebook global random state, live scrapes, unpinned
   dependencies and port mismatch risk. Reviewed modules are offline, seeded,
   version-pinned and checked for exact unrounded stage-count parity.

## New file roles

`wc2026/history.py`: strict loader, quarantine and date-batched history.
`evaluate.py`: disjoint fitting, baselines, normalized calibration, metrics.
`simulation.py`: shared-assumption Python port, validated matrices/events.
`tournament-data.json`: labeled source snapshot mirrored from Rust.
`export.py`: dated symmetric model probability export.
`tools/compare_simulators.py`: exact raw-count comparison; tests cover both
default Elo and exported ML matrix. `tests/`: leakage, integrity, event,
probability and deterministic-stage regressions. `docs/evaluation.json` and
`data-audit.json`: executed evidence. Root workflow, requirements and ignore
rules make validation reproducible without executing the archived notebook.

## Limits requiring new evidence

The final test has only 80 matches (160 correlated rows). Accepted scores and
match coverage still lack authoritative verification. Reconstructed Elo starts
at equal 1500 in partial 2018 history, not true historical national-team Elo.
Class-specific calibration/conclusions are uncertain at this sample size. The
default toy goal generator enforces an independently sampled outcome and does
not fit a coherent score distribution. It is not a new measured xG model.

Official FIFA conformance remains incomplete. A format-page request returned a
client-rendered shell, not a verified full regulation/bracket table. A sourced,
versioned official allocation table, head-to-head/discipline rules, bracket-path
tests and bronze-match handling are future work. Data-source/license provenance
and original individual/team contributions need owner confirmation.
