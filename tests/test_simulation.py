import unittest
from wc2026.simulation import Model, Scenario, Rng, TeamEvent, tilt_probs


class Simulation(unittest.TestCase):
    def test_seed_and_stage_conservation(self):
        scenario = Scenario(Model())
        a, b = scenario.run(30, 42), scenario.run(30, 42)
        self.assertEqual(a, b)
        for col, expected in enumerate([32, 16, 8, 4, 2, 1]):
            self.assertEqual(sum(row[col] for row in a.values()), 30 * expected)
        for row in a.values():
            self.assertEqual(row, sorted(row, reverse=True))

    def test_withdrawal_and_replacement(self):
        self.assertEqual(Scenario(Model(), [TeamEvent("withdraw", "Brazil")]).run(20)["Brazil"], [0]*6)
        model = Model({**Model().elo, "Italy": 1900})
        scenario = Scenario(model, [TeamEvent("withdraw", "Iraq", replacement="Italy")])
        self.assertIn("Italy", scenario.teams)
        self.assertNotIn("Iraq", scenario.teams)

    def test_knockout_only_withdrawal_cannot_reach_round16(self):
        result = Scenario(Model(), [TeamEvent("withdraw", "Argentina", knockout_only=True)]).run(30)
        self.assertEqual(result["Argentina"][1:], [0]*5)

    def test_rejects_invalid_event_combinations(self):
        for events in ([TeamEvent("elo_shock", "France", delta=float("nan"))],
                       [TeamEvent("withdraw", "Iraq", replacement="Spain")],
                       [TeamEvent("withdraw", "Iraq", replacement="Italy")],
                       [TeamEvent("withdraw", "Brazil"), TeamEvent("withdraw", "France")]):
            with self.assertRaises(ValueError):
                Scenario(Model(), events)

    def test_tilt_preserves_draw_even_at_extremes(self):
        for delta in [-10000, -100, 0, 100, 10000]:
            p = tilt_probs([.2, .25, .55], delta)
            self.assertEqual(p[1], .25)
            self.assertAlmostEqual(sum(p), 1)
            self.assertTrue(all(v >= 0 for v in p))

    def test_all_draw_knockout_uses_a_fair_coin(self):
        scenario = Scenario(Model(matrix={("A", "B"): [0, 1, 0]}))
        rng = Rng(42)
        wins = sum(scenario.knockout("A", "B", rng) == "A" for _ in range(1000))
        self.assertTrue(450 < wins < 550)

    def test_invalid_probabilities_and_zero_simulations(self):
        for p in [[-1, 1, 1], [float("nan"), 0, 1], [0, 0, 0]]:
            with self.assertRaises(ValueError):
                Model(matrix={("A", "B"): p})
        with self.assertRaises(ValueError):
            Scenario(Model()).run(0)
