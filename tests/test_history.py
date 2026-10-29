import csv
from datetime import date
from pathlib import Path
import tempfile
import unittest
import numpy as np
from wc2026.history import Match, load_matches, historical_features, clean_names
from wc2026.evaluate import chronological_masks, metrics, HeldOutStack


class HistoricalFeatures(unittest.TestCase):
    def test_names_use_evidence_not_arbitrary_suffix_trimming(self):
        names = clean_names(["Russia", "Russiaru", "Spain", "esSpain", "England", "Englandeng", "Canada"])
        self.assertEqual(names["Russiaru"], "Russia")
        self.assertEqual(names["esSpain"], "Spain")
        self.assertEqual(names["Englandeng"], "England")
        self.assertEqual(names["Canada"], "Canada")

    def test_future_results_cannot_change_earlier_features(self):
        past = Match(date(2020, 1, 1), "fixture", "A", "B", 2, 0)
        future = Match(date(2021, 1, 1), "fixture", "A", "B", 0, 9)
        first = historical_features([past])[0]
        both = historical_features([past, future])[0]
        np.testing.assert_array_equal(first, both[:2])
        np.testing.assert_array_equal(both[::2], -both[1::2])

    def test_same_day_does_not_see_other_same_day_result(self):
        matches = [Match(date(2020, 1, 1), "fixture", "A", "B", 9, 0),
                   Match(date(2020, 1, 1), "fixture", "A", "C", 0, 1)]
        x, *_ = historical_features(matches)
        np.testing.assert_array_equal(x, np.zeros_like(x))

    def test_forecast_cutoff_excludes_future_matches(self):
        match = Match(date(2026, 6, 11), "fixture", "A", "B", 1, 0)
        self.assertEqual(len(historical_features([match])[1]), 0)

    def test_contradictory_mirrors_are_quarantined(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "data.csv"
            with path.open("w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["date", "competition", "team", "opponent", "match_result", "match_score"])
                writer.writerows([["2020-01-01", "fixture", "A", "B", "Loss", "0-1"],
                                  ["2020-01-01", "fixture", "B", "A", "Loss", "0-1"],
                                  ["2021-01-01", "fixture", "A", "B", "Win", "1-0"],
                                  ["2021-01-01", "fixture", "B", "A", "Loss", "0-1"]])
            matches, report = load_matches(path)
            self.assertEqual(len(matches), 1)
            self.assertEqual(report["rejected_groups"], 1)

    def test_chronological_partitions_are_disjoint(self):
        dates = np.array(["2022-12-31", "2023-01-01", "2024-01-01", "2025-01-01"])
        masks = chronological_masks(dates)
        np.testing.assert_array_equal(sum(masks.values()), np.ones(4))

    def test_probability_metrics_boundaries(self):
        y = np.array([0, 1, 2])
        result = metrics(y, np.eye(3))
        self.assertEqual(result["multiclass_brier_sum"], 0)
        self.assertEqual(result["accuracy"], 1)
        with self.assertRaises(ValueError):
            metrics(y, np.full((3, 3), -1))

    def test_test_labels_do_not_fit_models_or_calibrators(self):
        rng = np.random.default_rng(42)
        x, y = rng.normal(size=(120, 5)), np.tile([0, 1, 2], 40)
        dates = np.repeat(["2020-01-01", "2023-01-01", "2024-01-01", "2025-01-01"], 30)
        masks = chronological_masks(dates)
        original = HeldOutStack().fit(x, y, masks)
        changed_y = y.copy()
        changed_y[masks["test"]] = 2
        changed = HeldOutStack().fit(x, changed_y, masks)
        np.testing.assert_array_equal(original.predict_proba(x), changed.predict_proba(x))
        np.testing.assert_allclose(original.predict_proba(x), original.predict_proba(-x)[:, ::-1])
        np.testing.assert_allclose(original.predict_proba(x).sum(axis=1), 1)


if __name__ == "__main__":
    unittest.main()
