# -*- coding: utf-8 -*-
"""预算四闸接线单测：token/cost 累加、轮次预检、超限拒答载荷。"""

import unittest

from vidlens.budget import (
    BudgetExceeded,
    BudgetLimits,
    BudgetTracker,
    budget_refusal_payload,
    record_completion_usage,
)


class RecordCompletionUsageTest(unittest.TestCase):
    def test_unlimited_tracker_accumulates_usage(self) -> None:
        tracker = BudgetTracker().start()
        record_completion_usage(
            tracker,
            prompt_tokens=10,
            completion_tokens=5,
            cost_usd=0.001,
        )
        snapshot = tracker.snapshot()
        self.assertEqual(snapshot["tokens"], 15)
        self.assertAlmostEqual(snapshot["costUsd"], 0.001)
        self.assertEqual(snapshot["rounds"], 0)

    def test_token_limit_raises(self) -> None:
        tracker = BudgetTracker(BudgetLimits(max_tokens=10)).start()
        with self.assertRaises(BudgetExceeded) as ctx:
            record_completion_usage(tracker, prompt_tokens=8, completion_tokens=5)
        self.assertEqual(ctx.exception.limit, "tokens")
        self.assertEqual(ctx.exception.used, 13)

    def test_cost_limit_raises(self) -> None:
        tracker = BudgetTracker(BudgetLimits(max_cost_usd=0.01)).start()
        with self.assertRaises(BudgetExceeded) as ctx:
            record_completion_usage(
                tracker,
                prompt_tokens=1,
                completion_tokens=1,
                cost_usd=0.02,
            )
        self.assertEqual(ctx.exception.limit, "cost_usd")

    def test_round_precheck_blocks_before_second_request(self) -> None:
        tracker = BudgetTracker(BudgetLimits(max_rounds=1)).start()
        tracker.check()
        tracker.add_round()
        tracker.check()
        with self.assertRaises(BudgetExceeded) as ctx:
            tracker.check()
            tracker.add_round()
            tracker.check()
        self.assertEqual(ctx.exception.limit, "rounds")
        self.assertEqual(tracker.rounds, 2)


class BudgetRefusalPayloadTest(unittest.TestCase):
    def test_payload_is_explicit_refusal(self) -> None:
        exc = BudgetExceeded("tokens", used=120, limit_value=100)
        payload = budget_refusal_payload(exc)
        self.assertFalse(payload["answerable"])
        self.assertEqual(payload["evidence"], [])
        self.assertIn("无法从视频", payload["finalAnswer"])
        self.assertIn("tokens", payload["conclusions"][0])
        self.assertEqual(payload["title"], "视频分析未完成（预算超限）")

    def test_payload_keeps_limit_name(self) -> None:
        exc = BudgetExceeded("cost_usd", used=0.02, limit_value=0.01)
        payload = budget_refusal_payload(exc)
        self.assertIn("cost_usd", payload["conclusions"][0])


if __name__ == "__main__":
    unittest.main()
