# -*- coding: utf-8 -*-
"""追问上下文预算模块单测。"""

import unittest

from vidlens.context_window import ContextPolicy, assemble, compress_for_summary


def history(turns: int, content_size: int = 0) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    for index in range(1, turns + 1):
        suffix = "x" * content_size
        items.append({"role": "user", "content": f"问题 {index}{suffix}"})
        items.append({"role": "assistant", "content": f"回答 {index}{suffix}"})
    return items


class AssembleTest(unittest.TestCase):
    def test_keeps_recent_turns_and_summarizes_older(self) -> None:
        context = assemble(
            history(6),
            policy=ContextPolicy(keep_turns=2, max_chars=6000),
        )
        self.assertEqual(
            [item["content"] for item in context.messages],
            ["问题 5", "回答 5", "问题 6", "回答 6"],
        )
        self.assertIn("早期提问：问题 1", context.summary)
        self.assertIn("当时回答：回答 1", context.summary)
        self.assertFalse(context.truncated)
        self.assertEqual(context.dropped_messages, 0)

    def test_uses_fallback_summary_when_history_is_empty(self) -> None:
        context = assemble([], fallback_summary="旧摘要")
        self.assertEqual(context.messages, [])
        self.assertEqual(context.summary, "旧摘要")
        self.assertFalse(context.truncated)

    def test_char_budget_drops_messages_and_reports_it(self) -> None:
        context = assemble(
            history(6, content_size=400),
            policy=ContextPolicy(
                keep_turns=2,
                max_chars=500,
                older_user_chars=300,
                older_assistant_chars=80,
                max_summary_lines=12,
            ),
        )
        self.assertTrue(context.truncated)
        self.assertGreater(context.dropped_messages, 0)
        self.assertLessEqual(
            sum(len(item["content"]) for item in context.messages) + len(context.summary),
            500,
        )


class CompressForSummaryTest(unittest.TestCase):
    def test_uses_question_answer_prefixes(self) -> None:
        summary = compress_for_summary(
            history(3),
            policy=ContextPolicy(keep_turns=2),
        )
        self.assertIn("问：问题 1", summary)
        self.assertIn("答：回答 1", summary)
        self.assertNotIn("user:", summary)

    def test_empty_history_returns_empty_summary(self) -> None:
        self.assertEqual(compress_for_summary([]), "")


class ContextPolicyTest(unittest.TestCase):
    def test_env_values_are_clamped(self) -> None:
        policy = ContextPolicy.from_env({
            "AGENT_CONTEXT_KEEP_TURNS": "0",
            "AGENT_CONTEXT_MAX_CHARS": "10",
            "AGENT_CONTEXT_OLDER_USER_CHARS": "0",
            "AGENT_CONTEXT_OLDER_ASSISTANT_CHARS": "-5",
            "AGENT_CONTEXT_SUMMARY_LINES": "999",
        })
        self.assertEqual(policy.keep_turns, 1)
        self.assertEqual(policy.max_chars, 500)
        self.assertEqual(policy.older_user_chars, 40)
        self.assertEqual(policy.older_assistant_chars, 0)
        self.assertEqual(policy.max_summary_lines, 50)


if __name__ == "__main__":
    unittest.main()
