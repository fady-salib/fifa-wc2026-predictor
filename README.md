# FIFA World Cup 2026 — Prediction Model

A 4-layer stacked ensemble (XGBoost / RF / ExtraTrees / LightGBM / MLP → LR + LightGBM → MLP blender → sigmoid
calibration, blended 70/30 with a competitive-match specialist) that predicts every match of the 48-team
World Cup, simulates the tournament, and answers **"what if something happens to a team?"**

```
├── FIFA_World_Cup_2026.ipynb   # training, prediction, Monte Carlo, what-if scenarios
├── data/                       # put the training CSV here (see data/README.md)
└── rust/                       # dependency-free Rust port of the simulator + team events
```

## Python notebook

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab FIFA_World_Cup_2026.ipynb
```

### Team events

```python
EVENTS = [
    TeamEvent('elo_shock', 'France', delta=-100),                    # star player injured
    TeamEvent('withdraw',  'Iraq', replacement='Italy'),             # swapped for another team
    TeamEvent('withdraw',  'Brazil', knockout_only=True),            # pulls out before the R32
]
compare_scenarios(EVENTS, sims=1000)   # baseline vs scenario, same seed
```

| Event | Effect |
|-------|--------|
| `elo_shock` (± delta) | Team's strength shifts by `delta` Elo points; P(draw) stays fixed. |
| `withdraw` | Team forfeits every match (3–0 walkovers, so it always finishes last in its group); in the knockouts its opponent advances. |
| `withdraw` + `replacement` | Another team takes the slot before the group stage. |
| `knockout_only=True` | Event only applies from the Round of 32. |

Events adjust the trained model's probabilities after the fact — no retraining.

## Rust

```bash
cd rust
cargo test
cargo run --release -- --sims 20000 --event shock:France:-100 --event withdraw:Iraq=Italy --elo "Italy=1880"
```

By default the Rust version uses Elo-only probabilities. To use the full ML ensemble, run the last notebook
cell (writes `wc2026_prob_matrix.csv`) and pass `--probs ../wc2026_prob_matrix.csv`. Run `cargo run -- --help`
for all options. The ML training itself stays in Python; Rust handles the simulation.

## What changed from V11

- Monte Carlo: the four "250-run" cells each reset the counters, so only 250 runs were reported. Now one cell, 1,000 runs, and model predictions are cached (each pair is predicted once, not thousands of times).
- Monte Carlo third-place teams were ranked by Elo; now by points / GD / GF like the main bracket. Goal difference now uses real goals.
- Removed `%pip install`, a duplicate import, the deprecated `use_label_encoder`, the unused `WorldCupMatches.csv` load, and literal `\n` printing in several `print` calls. Broken title cell fixed. Outputs cleared.

## Known limitations (not changed — worth knowing)

- **Validation is in-sample.** The 2023–2025 validation window is inside the training set, so the reported accuracy is optimistic. A time-based holdout would fix this.
- **Calibration is in-sample.** `FourLayerStack.fit()` does nothing, so the sigmoid calibrators see the stack's own training predictions.
- **Elo/FIFA rank/xG are current values applied to historical matches**, which leaks present-day strength into the past.
- Groups, Elo and FIFA ranks are hard-coded; the Round of 32 pairing is simplified rather than FIFA's official bracket.
