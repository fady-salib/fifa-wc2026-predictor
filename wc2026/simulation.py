"""Deterministic toy tournament, intentionally matching the Rust assumptions.

Not official FIFA group tiebreaks/knockout mapping. No live data or network I/O.
"""
import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

DATA = json.loads(Path(__file__).with_name("tournament-data.json").read_text())
GROUPS = DATA["groups"]
STAGES = ["R32", "R16", "QF", "SF", "Final", "Champion"]


class Rng:
    """SplitMix64 and Knuth Poisson with the same stream as rust/src/rng.rs."""
    def __init__(self, seed):
        if not 0 <= seed < 2**64:
            raise ValueError("seed must fit u64")
        self.state = seed

    def next_u64(self):
        mask = 2**64 - 1
        self.state = (self.state + 0x9E3779B97F4A7C15) & mask
        value = self.state
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
        return value ^ (value >> 31)

    def random(self):
        return (self.next_u64() >> 11) / 2**53

    def poisson(self, rate):
        limit, count, product = math.exp(-min(rate, 30)), 0, 1.0
        while True:
            product *= self.random()
            if product <= limit:
                return count
            count += 1

    def pick3(self, weights):
        value = self.random()
        return 0 if value < weights[0] else (1 if value < weights[0] + weights[1] else 2)


def elo_probs(difference):
    expected = 1 / (1 + 10 ** (-max(-10000, min(10000, difference)) / 400))
    draw = .10 + .14 * (1 - (2 * expected - 1) ** 2)
    values = [max(1 - expected - .5 * draw, 0), draw, max(expected - .5 * draw, 0)]
    total = sum(values)
    return [v / total for v in values]


def tilt_probs(probabilities, delta):
    if not math.isfinite(delta):
        raise ValueError("Elo shock must be finite")
    if delta == 0:
        return list(probabilities)
    loss, draw, win = probabilities
    decisive = 1 - draw
    if decisive <= 0:
        return list(probabilities)
    q = max(1e-12, min(1 - 1e-12, win / decisive))
    odds = max(-700, min(700, math.log(q / (1 - q)) + delta * math.log(10) / 400))
    shifted = decisive / (1 + math.exp(-odds))
    return [decisive - shifted, draw, shifted]


class Model:
    def __init__(self, elo=None, matrix=None):
        self.elo = dict(DATA["elo"] if elo is None else elo)
        if any(not t.strip() or not math.isfinite(v) or not 0 <= v <= 4000 for t, v in self.elo.items()):
            raise ValueError("invalid team or Elo (requires finite 0..4000)")
        self.matrix = dict(matrix or {})
        for (a, b), values in self.matrix.items():
            if not a or not b or a == b or len(values) != 3 or any(not math.isfinite(v) or not 0 <= v <= 1 for v in values) or abs(sum(values) - 1) > 2e-6:
                raise ValueError("invalid pairwise probability row")
            reverse = self.matrix.get((b, a))
            if reverse is not None and any(abs(values[i] - reverse[2-i]) > 2e-6 for i in range(3)):
                raise ValueError("inconsistent reverse probabilities")

    def probs(self, a, b):
        if (a, b) in self.matrix:
            p = self.matrix[a, b]
            return [v / sum(p) for v in p]
        if (b, a) in self.matrix:
            p = self.matrix[b, a][::-1]
            return [v / sum(p) for v in p]
        return elo_probs(self.elo.get(a, 1500) - self.elo.get(b, 1500))

    @classmethod
    def from_csv(cls, path):
        matrix = {}
        with open(path, newline="") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames != ["team_a", "team_b", "p_loss", "p_draw", "p_win"]:
                raise ValueError("invalid probability header")
            for row in reader:
                key = (row["team_a"].strip(), row["team_b"].strip())
                if key in matrix:
                    raise ValueError("duplicate pair")
                matrix[key] = [float(row[name]) for name in ("p_loss", "p_draw", "p_win")]
        if not matrix:
            raise ValueError("empty probability matrix")
        return cls(matrix=matrix)


@dataclass(frozen=True)
class TeamEvent:
    kind: str
    team: str
    delta: float = 0.0
    knockout_only: bool = False
    replacement: str | None = None


class Scenario:
    def __init__(self, model, events=()):
        self.model = model
        self.groups = [list(teams) for teams in GROUPS.values()]
        self.shock_group, self.shock_ko = defaultdict(float), defaultdict(float)
        self.out_group, self.out_ko = set(), set()
        for event in events:
            location = next(((g, i) for g, group in enumerate(self.groups) for i, team in enumerate(group) if team == event.team), None)
            if location is None:
                raise ValueError(f"{event.team!r} is not in the tournament")
            if event.kind == "elo_shock":
                if not math.isfinite(event.delta):
                    raise ValueError("shock must be finite")
                if not event.knockout_only:
                    self.shock_group[event.team] += event.delta
                self.shock_ko[event.team] += event.delta
            elif event.kind == "withdraw":
                if event.replacement is not None:
                    if event.knockout_only or event.replacement in [t for group in self.groups for t in group] or event.replacement not in model.elo:
                        raise ValueError("replacement needs an unused team with Elo, before groups")
                    self.groups[location[0]][location[1]] = event.replacement
                else:
                    if not event.knockout_only:
                        self.out_group.add(event.team)
                    self.out_ko.add(event.team)
            else:
                raise ValueError("unknown event kind")
        if len(self.out_ko) > 1:
            raise ValueError("toy bracket supports at most one unreplaced withdrawal")
        if any(not math.isfinite(v) for v in [*self.shock_group.values(), *self.shock_ko.values()]):
            raise ValueError("combined shock must be finite")
        self.teams = [team for group in self.groups for team in group]

    def group_match(self, a, b, rng):
        if a in self.out_group:
            return 0, 3
        if b in self.out_group:
            return 3, 0
        p = tilt_probs(self.model.probs(a, b), self.shock_group[a] - self.shock_group[b])
        outcome = rng.pick3([p[2], p[1], p[0]])
        difference = self.model.elo.get(a, 1500) - self.model.elo.get(b, 1500)
        ga, gb = rng.poisson(max(.3, 1.25 + difference / 900)), rng.poisson(max(.3, 1.25 - difference / 900))
        if outcome == 0:
            ga = max(ga, gb + 1)
        elif outcome == 2:
            gb = max(gb, ga + 1)
        else:
            ga = gb = min(ga, gb)
        return ga, gb

    def knockout(self, a, b, rng):
        if a in self.out_ko:
            return b
        if b in self.out_ko:
            return a
        p = tilt_probs(self.model.probs(a, b), self.shock_ko[a] - self.shock_ko[b])
        decisive = p[0] + p[2]
        win = p[2] / decisive if decisive > 0 else .5
        return a if rng.random() < win else b

    def run(self, sims=1000, seed=42):
        if not isinstance(sims, int) or sims < 1:
            raise ValueError("sims must be a positive integer")
        rng = Rng(seed)
        counts = {team: [0] * 6 for team in self.teams}
        for _ in range(sims):
            winners, runners, thirds = [], [], []
            for group in self.groups:
                table = {team: [0, 0, 0, rng.random()] for team in group}
                for a, b in combinations(group, 2):
                    ga, gb = self.group_match(a, b, rng)
                    table[a][1] += ga - gb
                    table[b][1] += gb - ga
                    table[a][2] += ga
                    table[b][2] += gb
                    table[a][0] += 3 if ga > gb else (1 if ga == gb else 0)
                    table[b][0] += 3 if gb > ga else (1 if ga == gb else 0)
                key = lambda team: (-table[team][0], -table[team][1], -table[team][2], table[team][3])
                order = sorted(group, key=key)
                winners.append(order[0])
                runners.append(order[1])
                thirds.append((order[2], table[order[2]]))
            thirds.sort(key=lambda item: (-item[1][0], -item[1][1], -item[1][2], item[1][3]))
            # Retained toy mapping: adjacent winner/runner pairs, then third-v-third.
            pairs = [(winners[i], runners[i ^ 1]) for i in range(12)]
            pairs += [(thirds[i][0], thirds[i + 1][0]) for i in range(0, 8, 2)]
            for a, b in pairs:
                counts[a][0] += 1
                counts[b][0] += 1
            teams = [self.knockout(a, b, rng) for a, b in pairs]
            for stage in range(1, 6):
                for team in teams:
                    counts[team][stage] += 1
                if len(teams) == 1:
                    break
                teams = [self.knockout(teams[i], teams[i+1], rng) for i in range(0, len(teams), 2)]
        return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sims", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--probs", type=Path)
    args = parser.parse_args()
    model = Model.from_csv(args.probs) if args.probs else Model()
    result = Scenario(model).run(args.sims, args.seed)
    print("TOY rules: not official bracket odds")
    for team in sorted(result, key=lambda t: -result[t][5])[:15]:
        print(f"{team}: {100 * result[team][5] / args.sims:.1f}% toy championship frequency")


if __name__ == "__main__":
    main()
