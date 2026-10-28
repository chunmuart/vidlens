# -*- coding: utf-8 -*-
"""Agent 层指标单测：结构有效率、引用支持率、拒答（误拒/漏拒）、成功率。"""

import unittest

from vidlens.agent_metrics import (
    abstention_metrics,
    citation_support_rate,
    structured_valid_rate,
    summarize,
    task_success_rate,
)


class StructuredValidRateTest(unittest.TestCase):
    def test_empty_returns_none(self) -> None:
        self.assertIsNone(structured_valid_rate([]))

    def test_counts_dicts_and_json_strings(self) -> None:
        payloads = [
            {"answerable": True},
            '{"answerable": false, "conclusions": []}',
            "not json",
            "[1, 2, 3]",          # 列表不算合法报告
            "",                    # 空串不算
        ]
        self.assertEqual(structured_valid_rate(payloads), 0.4)

    def test_bytes_are_decoded(self) -> None:
        self.assertEqual(structured_valid_rate([b'{"a": 1}']), 1.0)


class CitationSupportRateTest(unittest.TestCase):
    def test_empty_returns_none(self) -> None:
        self.assertIsNone(citation_support_rate([]))

    def test_ratio(self) -> None:
        self.assertEqual(citation_support_rate([True, True, False, False]), 0.5)


class AbstentionMetricsTest(unittest.TestCase):
    def test_empty_samples(self) -> None:
        result = abstention_metrics([])
        self.assertEqual(result["samples"], 0)
        self.assertIsNone(result["abstentionPrecision"])
        self.assertIsNone(result["abstentionRecall"])

    def test_mixed_cases(self) -> None:
        # (answerable, abstained)
        samples = [
            (True, False),   # 该答且答了 → answeredOk
            (True, True),    # 该答却拒了 → falseReject
            (False, True),   # 该拒且拒了 → correctAbstain
            (False, False),  # 该拒却答了 → missedAbstain
        ]
        result = abstention_metrics(samples)
        self.assertEqual(result["answeredOk"], 1)
        self.assertEqual(result["falseReject"], 1)
        self.assertEqual(result["correctAbstain"], 1)
        self.assertEqual(result["missedAbstain"], 1)
        self.assertEqual(result["abstentionPrecision"], 0.5)   # 1 / (1+1)
        self.assertEqual(result["abstentionRecall"], 0.5)      # 1 / (1+1)
        self.assertEqual(result["falseRejectRate"], 0.25)      # 1 / 4

    def test_all_correct(self) -> None:
        result = abstention_metrics([(True, False), (False, True)])
        self.assertEqual(result["abstentionPrecision"], 1.0)
        self.assertEqual(result["abstentionRecall"], 1.0)
        self.assertEqual(result["falseRejectRate"], 0.0)
        self.assertEqual(result["missedAbstainRate"], 0.0)


class TaskSuccessRateTest(unittest.TestCase):
    def test_empty_returns_none(self) -> None:
        self.assertIsNone(task_success_rate([]))

    def test_counts_completed_states(self) -> None:
        statuses = ["COMPLETED", "COMPLETED_REUSED", "FAILED", "RETRYING"]
        self.assertEqual(task_success_rate(statuses), 0.5)


class SummarizeTest(unittest.TestCase):
    def test_shape_and_values(self) -> None:
        report = summarize(
            payloads=[{"a": 1}, "bad"],
            citation_supports=[True, False],
            abstention_samples=[(False, True), (True, False)],
            statuses=["COMPLETED", "FAILED"],
        )
        self.assertEqual(report["structuredValidRate"], 0.5)
        self.assertEqual(report["citationSupportRate"], 0.5)
        self.assertEqual(report["taskSuccessRate"], 0.5)
        self.assertEqual(report["abstention"]["abstentionPrecision"], 1.0)

    def test_all_none_when_no_data(self) -> None:
        report = summarize()
        self.assertIsNone(report["structuredValidRate"])
        self.assertIsNone(report["citationSupportRate"])
        self.assertIsNone(report["taskSuccessRate"])
        self.assertEqual(report["abstention"]["samples"], 0)


if __name__ == "__main__":
    unittest.main()
