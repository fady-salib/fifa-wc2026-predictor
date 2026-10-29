"""Compare unrounded Python/Rust counts using identical toy rules and SplitMix64."""
import argparse
import csv
from pathlib import Path
import subprocess
import tempfile
from wc2026.simulation import Model, Scenario, TeamEvent, STAGES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rust-bin", type=Path, default=Path("rust/target/debug/wc2026"))
    parser.add_argument("--probs", type=Path)
    parser.add_argument("--sims", type=int, default=200)
    args = parser.parse_args()
    scenarios = [([], []),
                 (["shock:France:-100"], [TeamEvent("elo_shock", "France", delta=-100)]),
                 (["withdraw:Brazil@ko"], [TeamEvent("withdraw", "Brazil", knockout_only=True)]),
                 (["withdraw:Brazil"], [TeamEvent("withdraw", "Brazil")]),
                 (["withdraw:Iraq=Italy"], [TeamEvent("withdraw", "Iraq", replacement="Italy")])]
    for specs, events in scenarios:
        model = Model.from_csv(args.probs) if args.probs else Model()
        model.elo["Italy"] = 1900
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "counts.csv"
            command = [str(args.rust_bin.resolve()), "--sims", str(args.sims), "--seed", "42",
                       "--counts", str(output), "--top", "0", "--elo", "Italy=1900"]
            if args.probs:
                command += ["--probs", str(args.probs)]
            for spec in specs:
                command += ["--event", spec]
            subprocess.run(command, check=True, capture_output=True, text=True)
            with output.open() as file:
                rust = {row["team"]: [int(row[stage]) for stage in STAGES] for row in csv.DictReader(file)}
            python = Scenario(model, events).run(args.sims, 42)
            if rust != python:
                raise AssertionError(f"stage counts differ: {specs}")
            print(f"PASS exact stage-count parity: {specs or ['baseline']}")


if __name__ == "__main__":
    main()
