"""Validate mirrored match records and construct strictly pre-date features.

No current ratings, rankings, xG, same-match statistics or global imputation are
used. Contradictory source records are quarantined rather than guessed/repaired.
"""
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
import csv
import re
import numpy as np

LABELS = {"Loss": 0, "Draw": 1, "Win": 2}
FEATURES = ["elo_diff", "form_diff", "goal_balance_diff", "rest_days_diff", "h2h_centered"]


@dataclass(frozen=True)
class Match:
    date: date
    competition: str
    a: str
    b: str
    goals_a: int
    goals_b: int


def clean_names(names):
    names = set(names)
    result = {}
    for name in names:
        prefix = re.fullmatch(r"[a-z]{2}([A-Z].+)", name)
        cleaned = prefix.group(1) if prefix else name
        for length in (3, 2):
            if len(cleaned) > length and cleaned[-length:].islower() and cleaned[:-length] in names:
                cleaned = cleaned[:-length]
                break
        result[name] = cleaned
    return result


def load_matches(path):
    with open(path, newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    names = clean_names([r[key] for r in rows for key in ("team", "opponent")])
    grouped = defaultdict(list)
    rejected = []
    for line, row in enumerate(rows, 2):
        try:
            when = date.fromisoformat(row["date"])
            a, b = names[row["team"]], names[row["opponent"]]
            if not a or not b or a == b:
                raise ValueError("invalid teams")
            key = (when, row.get("competition", ""), *sorted((a, b)))
            grouped[key].append((line, row, a, b))
        except (ValueError, KeyError) as exc:
            rejected.append({"lines": [line], "reason": str(exc)})
    matches = []
    for (when, competition, a, b), records in sorted(grouped.items()):
        try:
            if len(records) != 2 or {r[2] for r in records} != {a, b}:
                raise ValueError("expected exactly two reciprocal team perspectives")
            scores = []
            for line, row, team, opponent in records:
                score = re.fullmatch(r"\s*(\d+)\s*[-–]\s*(\d+)\s*", row["match_score"])
                if not score or row["match_result"] not in LABELS:
                    raise ValueError("missing/invalid result or score")
                ga, gb = map(int, score.groups())
                expected = 2 if ga > gb else (0 if ga < gb else 1)
                if LABELS[row["match_result"]] != expected:
                    raise ValueError("result disagrees with perspective score")
                scores.append((ga, gb) if team == a else (gb, ga))
            if scores[0] != scores[1]:
                raise ValueError("opponent records disagree on match score")
            matches.append(Match(when, competition, a, b, *scores[0]))
        except ValueError as exc:
            rejected.append({"date": when.isoformat(), "teams": [a, b],
                             "lines": [r[0] for r in records], "reason": str(exc)})
    report = {"source_rows": len(rows), "accepted_matches": len(matches),
              "rejected_groups": len(rejected), "rejected": rejected,
              "name_changes": {k: v for k, v in sorted(names.items()) if k != v}}
    return matches, report


class History:
    def __init__(self):
        self.elo = defaultdict(lambda: 1500.0)
        self.form = defaultdict(list)
        self.goals = defaultdict(list)
        self.last = {}
        self.h2h = defaultdict(list)

    def features(self, a, b, when):
        def mean(values, default):
            return sum(values[-5:]) / len(values[-5:]) if values else default
        def rest(team):
            return min((when - self.last[team]).days, 90) if team in self.last else 30
        pair = tuple(sorted((a, b)))
        h2h = mean(self.h2h[pair], .5)
        if a != pair[0]:
            h2h = 1 - h2h
        return [self.elo[a] - self.elo[b], mean(self.form[a], .5) - mean(self.form[b], .5),
                mean(self.goals[a], 0) - mean(self.goals[b], 0), rest(a) - rest(b), h2h - .5]

    def update(self, match):
        a, b = match.a, match.b
        outcome = 1.0 if match.goals_a > match.goals_b else (0.0 if match.goals_a < match.goals_b else .5)
        expected = 1 / (1 + 10 ** (-(self.elo[a] - self.elo[b]) / 400))
        change = 20 * (outcome - expected)
        self.elo[a] += change
        self.elo[b] -= change
        self.form[a].append(outcome)
        self.form[b].append(1 - outcome)
        balance = match.goals_a - match.goals_b
        self.goals[a].append(balance)
        self.goals[b].append(-balance)
        self.last[a] = self.last[b] = match.date
        pair = tuple(sorted((a, b)))
        self.h2h[pair].append(outcome if a == pair[0] else 1 - outcome)


def historical_features(matches, before=date(2026, 6, 11)):
    history = History()
    dates, x, y, identities = [], [], [], []
    by_date = defaultdict(list)
    for match in matches:
        if match.date < before:
            by_date[match.date].append(match)
    for when, daily in sorted(by_date.items()):
        # All same-day feature rows precede every same-day update. Mirror rows
        # therefore cannot reveal their own outcome or enter different splits.
        for match in sorted(daily, key=lambda m: (m.a, m.b)):
            features = history.features(match.a, match.b, when)
            target = 2 if match.goals_a > match.goals_b else (0 if match.goals_a < match.goals_b else 1)
            for sign in (1, -1):
                x.append([sign * value for value in features])
                y.append(target if sign == 1 else 2 - target)
                dates.append(when.isoformat())
                identities.append((when.isoformat(), match.a, match.b))
        for match in sorted(daily, key=lambda m: (m.a, m.b)):
            history.update(match)
    return np.asarray(x, dtype=float), np.asarray(y, dtype=int), np.asarray(dates), identities, history
