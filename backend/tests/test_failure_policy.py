# -*- coding: utf-8 -*-
"""工具失败策略单测：分类、退避、重试用尽后降级。"""

import unittest

from vidlens.failure_policy import (
    FailureAction,
    backoff_delay_ms,
    classify_failure,
    decide_failure,
    describe_failure,
)


class ClassifyTest(unittest.TestCase):
    def test_validation_errors_abort(self) -> None:
        self.assertIs(classify_failure(ValueError("bad arg")), FailureAction.ABORT)
        self.assertIs(classify_failure(TypeError("wrong type")), FailureAction.ABORT)
        self.assertIs(classify_failure(KeyError("missing")), FailureAction.ABORT)

    def test_network_errors_retry(self) -> None:
        self.assertIs(classify_failure(TimeoutError("timed out")), FailureAction.RETRY)
        self.assertIs(classify_failure(ConnectionError("reset")), FailureAction.RETRY)

    def test_message_hints(self) -> None:
        self.assertIs(classify_failure(RuntimeError("HTTP 503")), FailureAction.RETRY)
        self.assertIs(classify_failure(RuntimeError("rate limit exceeded")), FailureAction.RETRY)
        self.assertIs(classify_failure(RuntimeError("请求超时")), FailureAction.RETRY)
        self.assertIs(classify_failure(RuntimeError("unknown tool: foo")), FailureAction.ABORT)
        self.assertIs(classify_failure(RuntimeError("参数校验失败")), FailureAction.ABORT)

    def test_unknown_errors_degrade_not_abort(self) -> None:
        self.assertIs(classify_failure(RuntimeError("something odd")), FailureAction.DEGRADE)


class BackoffTest(unittest.TestCase):
    def test_exponential_without_jitter(self) -> None:
        self.assertEqual(backoff_delay_ms(1, base_ms=1000, jitter=0), 1000)
        self.assertEqual(backoff_delay_ms(2, base_ms=1000, jitter=0), 2000)
        self.assertEqual(backoff_delay_ms(3, base_ms=1000, jitter=0), 4000)

    def test_capped(self) -> None:
        self.assertEqual(backoff_delay_ms(10, base_ms=1000, cap_ms=8000, jitter=0), 8000)

    def test_jitter_stays_within_bounds(self) -> None:
        low = backoff_delay_ms(2, base_ms=1000, jitter=0.2, rng=lambda: 0.0)
        high = backoff_delay_ms(2, base_ms=1000, jitter=0.2, rng=lambda: 1.0)
        self.assertEqual(low, 1600)
        self.assertEqual(high, 2400)


class DecideTest(unittest.TestCase):
    def test_retry_with_delay(self) -> None:
        action, delay = decide_failure(TimeoutError("timed out"), attempt=1, max_attempts=3)
        self.assertIs(action, FailureAction.RETRY)
        self.assertGreater(delay, 0)

    def test_last_attempt_degrades_instead_of_retrying(self) -> None:
        action, delay = decide_failure(TimeoutError("timed out"), attempt=3, max_attempts=3)
        self.assertIs(action, FailureAction.DEGRADE)
        self.assertEqual(delay, 0)

    def test_abort_has_no_delay(self) -> None:
        action, delay = decide_failure(ValueError("bad"), attempt=1, max_attempts=3)
        self.assertIs(action, FailureAction.ABORT)
        self.assertEqual(delay, 0)

    def test_describe_failure_payload(self) -> None:
        payload = describe_failure(TimeoutError("timed out"), attempt=1, max_attempts=3)
        self.assertFalse(payload["error"] == "")
        self.assertEqual(payload["errorClass"], "RETRY")
        self.assertTrue(payload["retryable"])
        self.assertEqual(payload["errorType"], "TimeoutError")


if __name__ == "__main__":
    unittest.main()
