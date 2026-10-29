"""Export the held-out model at a declared forecast date for the toy simulator."""
import argparse
import csv
from datetime import date
from itertools import permutations
from pathlib import Path
import numpy as np
from .evaluate import run
from .simulation import GROUPS


def export_matrix(model, history, path, forecast_date=date(2026, 6, 11)):
    if history.last and max(history.last.values()) >= forecast_date:
        raise ValueError("forecast must follow every history observation date")
    teams = [team for group in GROUPS.values() for team in group]
    pairs = list(permutations(teams, 2))
    x = np.asarray([history.features(a, b, forecast_date) for a, b in pairs])
    probabilities = model.predict_proba(x)
    with open(path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["team_a", "team_b", "p_loss", "p_draw", "p_win"])
        for (a, b), p in zip(pairs, probabilities):
            writer.writerow([a, b, *(f"{v:.6f}" for v in p)])
    return len(pairs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/international_football_2018_present.csv"))
    parser.add_argument("--output", type=Path, default=Path("wc2026_prob_matrix.csv"))
    args = parser.parse_args()
    _, model, history = run(args.data, "outputs/export-evaluation")
    print(f"Exported {export_matrix(model, history, args.output)} toy-scenario pair rows to {args.output}")


if __name__ == "__main__":
    main()
