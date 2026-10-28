# -*- coding: utf-8 -*-
"""AgentToolbox.execute 的集成测试：失败分类 + 预算闸门。

只依赖 vidlens.agent（其传递依赖 retrieval 也是纯标准库），
因此可以在没有 fastapi / sqlalchemy 的环境里跑。
"""

import unittest

from vidlens.agent import AgentToolbox
from vidlens.budget import BudgetExceeded, BudgetLimits, BudgetTracker


def _make_toolbox(budget=None) -> AgentToolbox:
    return AgentToolbox(
        metadata={"title": "demo"},
        segments=[],
        normalize_report=lambda report: report,
        evaluate_report=lambda report, segments: {"ok": True},
        budget=budget,
    )


class ToolExecuteFailureTest(unittest.TestCase):
    def test_unknown_tool_is_abort_not_retryable(self) -> None:
        result = _make_toolbox().execute("not_a_tool", {})
        self.assertFalse(result["ok"])
        self.assertEqual(result["errorClass"], "ABORT")
        self.assertFalse(result["retryable"])

    def test_retryable_exception_is_swallowed_with_marker(self) -> None:
        toolbox = _make_toolbox()

        def boom(**kwargs):
            raise TimeoutError("upstream timed out")

        toolbox._get_video_metadata = boom  # type: ignore[assignment]
        result = toolbox.execute("get_video_metadata", {})
        self.assertFalse(result["ok"])
        self.assertEqual(result["errorClass"], "RETRY")
        self.assertTrue(result["retryable"])
        self.assertGreater(result["retryDelayMs"], 0)

    def test_non_retryable_exception_still_raises(self) -> None:
        toolbox = _make_toolbox()

        def boom(**kwargs):
            raise RuntimeError("something odd")

        toolbox._get_video_metadata = boom  # type: ignore[assignment]
        with self.assertRaises(RuntimeError):
            toolbox.execute("get_video_metadata", {})

    def test_trace_records_error_class(self) -> None:
        toolbox = _make_toolbox()
        toolbox.execute("not_a_tool", {})
        trace = toolbox.trace()
        self.assertEqual(len(trace), 1)
        self.assertEqual(trace[0]["errorClass"], "ABORT")
        self.assertFalse(trace[0]["success"])


class ToolExecuteBudgetTest(unittest.TestCase):
    def test_budget_exceeded_blocks_tool_and_marks_trace(self) -> None:
        tracker = BudgetTracker(BudgetLimits(max_rounds=0, max_tokens=0)).start()
        # 手动制造一个已超限的 tracker：直接改 limits
        tracker.limits = BudgetLimits(max_rounds=1)
        tracker.add_round(5)
        toolbox = _make_toolbox(budget=tracker)

        result = toolbox.execute("get_video_metadata", {})
        self.assertFalse(result["ok"])
        self.assertEqual(result["errorClass"], "BUDGET")
        self.assertFalse(result["retryable"])
        trace = toolbox.trace()
        self.assertEqual(trace[-1]["errorClass"], "BUDGET")
        self.assertEqual(trace[-1]["resultPreview"], "预算超限，未执行")

    def test_unlimited_budget_does_not_block(self) -> None:
        tracker = BudgetTracker().start()
        toolbox = _make_toolbox(budget=tracker)
        # metadata 工具本身是轻量实现，正常应返回 ok
        result = toolbox.execute("get_video_metadata", {})
        self.assertTrue(result.get("ok", True))


if __name__ == "__main__":
    unittest.main()
