# -*- coding: utf-8 -*-
"""零依赖 trace span 树单测。"""

import json
import unittest

from vidlens.trace_spans import attach_trace_spans, build_trace_spans


class BuildTraceSpansTest(unittest.TestCase):
    def test_empty_trace_has_only_root(self) -> None:
        spans = build_trace_spans({})
        self.assertEqual(len(spans), 1)
        root = spans[0]
        self.assertEqual(root["name"], "analysis")
        self.assertIsNone(root["parent"])
        self.assertEqual(root["startOffsetMs"], 0)
        self.assertEqual(root["durationMs"], 0)

    def test_stage_offsets_and_nested_critic(self) -> None:
        spans = build_trace_spans({
            "stageDurationMs": {
                "VIDEO_CONTEXT": 100,
                "PLANNER": 50,
                "EXECUTOR": 400,
                "CRITIC": 80,
            },
        })
        by_name = {item["name"]: item for item in spans}
        self.assertEqual(by_name["analysis"]["durationMs"], 550)
        self.assertEqual(by_name["video_context"]["startOffsetMs"], 0)
        self.assertEqual(by_name["planner"]["startOffsetMs"], 100)
        self.assertEqual(by_name["executor"]["startOffsetMs"], 150)
        self.assertEqual(by_name["critic"]["startOffsetMs"], 470)
        self.assertEqual(by_name["critic"]["durationMs"], 80)
        self.assertEqual(by_name["critic"]["parent"], "executor")
        self.assertTrue(by_name["critic"]["attributes"]["approximated"])

    def test_tool_spans_are_sequential_and_keep_error_class(self) -> None:
        spans = build_trace_spans({
            "stageDurationMs": {"EXECUTOR": 100},
            "toolCalls": [
                {"index": 1, "tool": "search_timeline", "durationMs": 30, "success": True},
                {
                    "index": 2,
                    "tool": "generate_report",
                    "durationMs": 20,
                    "success": False,
                    "errorClass": "BUDGET",
                },
            ],
        })
        tools = [item for item in spans if item["name"].startswith("tool.")]
        self.assertEqual([item["startOffsetMs"] for item in tools], [0, 30])
        self.assertEqual(tools[0]["parent"], "executor")
        self.assertEqual(tools[1]["attributes"]["errorClass"], "BUDGET")
        self.assertFalse(tools[1]["attributes"]["success"])

    def test_provider_group_and_request_attributes(self) -> None:
        spans = build_trace_spans({
            "providerRequests": [
                {
                    "phase": "PLANNER",
                    "model": "m",
                    "latencyMs": 100,
                    "success": True,
                    "promptTokens": 10,
                    "completionTokens": 5,
                    "totalTokens": 15,
                    "estimatedCostUsd": 0.001,
                    "toolCallCount": 1,
                    "attempt": 1,
                },
                {
                    "phase": "WRITER",
                    "model": "m",
                    "latencyMs": 200,
                    "success": True,
                    "promptTokens": 20,
                    "completionTokens": 10,
                    "totalTokens": 30,
                    "estimatedCostUsd": None,
                    "toolCallCount": 1,
                    "attempt": 1,
                },
            ],
        })
        provider = next(item for item in spans if item["name"] == "provider")
        self.assertEqual(provider["durationMs"], 300)
        requests = [item for item in spans if item["name"].startswith("provider.")]
        self.assertEqual([item["startOffsetMs"] for item in requests], [0, 100])
        self.assertEqual(requests[0]["attributes"]["totalTokens"], 15)
        self.assertEqual(requests[0]["parent"], "provider")

    def test_root_duration_covers_provider_and_tool_time(self) -> None:
        spans = build_trace_spans({
            "stageDurationMs": {"EXECUTOR": 10},
            "toolCalls": [{"durationMs": 50}],
            "providerRequests": [{"latencyMs": 80}],
        })
        self.assertEqual(spans[0]["durationMs"], 80)

    def test_failure_trace_without_stage_durations_still_builds(self) -> None:
        spans = build_trace_spans({
            "lastErrorType": "RuntimeError",
            "toolCalls": [{"tool": "search_timeline", "durationMs": 25, "success": False}],
        })
        self.assertEqual(spans[0]["name"], "analysis")
        self.assertEqual(spans[0]["durationMs"], 25)
        self.assertEqual(spans[1]["parent"], "analysis")

    def test_spans_are_json_serializable(self) -> None:
        spans = build_trace_spans({
            "stageDurationMs": {"EXECUTOR": 10},
            "toolCalls": [{"tool": "x", "durationMs": 3, "success": True}],
            "providerRequests": [{"phase": "WRITER", "latencyMs": 4, "success": True}],
        })
        json.dumps(spans, ensure_ascii=False)


class AttachTraceSpansTest(unittest.TestCase):
    def test_attach_only_adds_spans(self) -> None:
        trace = {
            "stageDurationMs": {"EXECUTOR": 10},
            "providerUsage": {"requestCount": 0},
            "providerRequests": [],
        }
        before = set(trace)
        result = attach_trace_spans(trace)
        self.assertIs(result, trace)
        self.assertEqual(set(trace), before | {"spans"})
        self.assertEqual(trace["providerUsage"], {"requestCount": 0})
        self.assertIsInstance(trace["spans"], list)

    def test_build_does_not_mutate_input(self) -> None:
        trace = {"stageDurationMs": {"EXECUTOR": 10}}
        build_trace_spans(trace)
        self.assertEqual(list(trace), ["stageDurationMs"])


if __name__ == "__main__":
    unittest.main()
