# -*- coding: utf-8 -*-
"""预算闸门单测：覆盖未超限、四项各自触发、环境变量解析、快照结构。"""

import unittest

from vidlens.budget import BudgetExceeded, BudgetLimits, BudgetTracker


class FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance_ms(self, ms: int) -> None:
        self.now += ms / 1000.0


class BudgetLimitsTest(unittest.TestCase):
    def test_zero_limits_are_inactive(self) -> None:
        self.assertFalse(BudgetLimits().active)

    def test_any_limit_activates(self) -> None:
        self.assertTrue(BudgetLimits(max_rounds=1).active)
        self.assertTrue(BudgetLimits(max_tokens=100).active)
        self.assertTrue(BudgetLimits(max_duration_ms=1000).active)
        self.assertTrue(BudgetLimits(max_cost_usd=0.01).active)

    def test_from_env_parses_numbers_and_ignores_bad_values(self) -> None:
        env = {
            "AGENT_MAX_ROUNDS": "3",
            "AGENT_MAX_ESTIMATED_TOKENS": "50000",
            "AGENT_MAX_DURATION_MS": "120000",
            "AGENT_MAX_ESTIMATED_COST": "0.05",
        }
        limits = BudgetLimits.from_env(env)
        self.assertEqual(limits.max_rounds, 3)
        self.assertEqual(limits.max_tokens, 50000)
        self.assertEqual(limits.max_duration_ms, 120000)
        self.assertAlmostEqual(limits.max_cost_usd, 0.05)

        bad = BudgetLimits.from_env({"AGENT_MAX_ROUNDS": "abc", "AGENT_MAX_ESTIMATED_COST": ""})
        self.assertEqual(bad.max_rounds, 0)
        self.assertEqual(bad.max_cost_usd, 0.0)


class BudgetTrackerTest(unittest.TestCase):
    def test_unlimited_tracker_never_raises(self) -> None:
        tracker = BudgetTracker().start()
        for _ in range(100):
            tracker.add_round()
            tracker.add_tokens(total_tokens=10_000)
        tracker.check()  # 不应抛异常

    def test_rounds_limit(self) -> None:
        tracker = BudgetTracker(BudgetLimits(max_rounds=2)).start()
        tracker.add_round()
        tracker.add_round()
        tracker.check()
        tracker.add_round()
        with self.assertRaises(BudgetExceeded) as ctx:
            tracker.check()
        self.assertEqual(ctx.exception.limit, "rounds")
        self.assertEqual(ctx.exception.used, 3)
        self.assertEqual(ctx.exception.limit_value, 2)

    def test_tokens_limit(self) -> None:
        tracker = BudgetTracker(BudgetLimits(max_tokens=1000)).start()
        tracker.add_tokens(prompt_tokens=600, completion_tokens=300)
        tracker.check()
        tracker.add_tokens(total_tokens=200)
        with self.assertRaises(BudgetExceeded) as ctx:
            tracker.check()
        self.assertEqual(ctx.exception.limit, "tokens")

    def test_duration_limit_uses_clock(self) -> None:
        clock = FakeClock()
        tracker = BudgetTracker(BudgetLimits(max_duration_ms=1000), clock=clock).start()
        clock.advance_ms(900)
        tracker.check()
        clock.advance_ms(200)
        with self.assertRaises(BudgetExceeded) as ctx:
            tracker.check()
        self.assertEqual(ctx.exception.limit, "duration_ms")

    def test_cost_limit(self) -> None:
        tracker = BudgetTracker(BudgetLimits(max_cost_usd=0.01)).start()
        tracker.add_tokens(prompt_tokens=10, completion_tokens=5, cost_usd=0.004)
        tracker.check()
        tracker.add_cost(0.008)
        with self.assertRaises(BudgetExceeded) as ctx:
            tracker.check()
        self.assertEqual(ctx.exception.limit, "cost_usd")

    def test_negative_values_are_ignored(self) -> None:
        tracker = BudgetTracker(BudgetLimits(max_rounds=1)).start()
        tracker.add_round(-5)
        tracker.add_tokens(total_tokens=-100)
        tracker.add_cost(-1.0)
        self.assertEqual(tracker.rounds, 0)
        self.assertEqual(tracker.tokens, 0)
        self.assertEqual(tracker.cost_usd, 0.0)

    def test_snapshot_shape(self) -> None:
        tracker = BudgetTracker(BudgetLimits(max_rounds=5, max_tokens=100)).start()
        tracker.add_round(2)
        tracker.add_tokens(prompt_tokens=10, completion_tokens=20)
        snap = tracker.snapshot()
        self.assertEqual(snap["rounds"], 2)
        self.assertEqual(snap["tokens"], 30)
        self.assertEqual(snap["limits"]["maxRounds"], 5)
        self.assertIn("elapsedMs", snap)


if __name__ == "__main__":
    unittest.main()
