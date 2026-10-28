"""多轮追问的上下文治理。

问题：追问是「历史消息 + 摘要 + 本轮证据」一起发给模型。轮次多了会膨胀，
而原来的 `session.summary` 只是「最近 12 条截断拼接」，早期信息被硬切掉，
且没有按预算分配空间。

策略（按损失从小到大）：
1. 最近的若干轮**完整保留**（用户问题 + 助手回答），保证当前话题连贯；
2. 更早的轮次**只保留用户问题**、丢掉助手回答——因为回答长、且可以由证据重建，
   而问题承载了用户的意图变化（改预算、改目标），丢了就答错；
3. 再往前的部分压成一行「早期对话概要」，只留问题开头；
4. 所有内容按字符预算从旧到新截断，保证总长度可控。

这个模块只做「怎么切」，不做 IO、不调模型，便于单测。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class ContextPolicy:
    """上下文预算。"""

    keep_turns: int = 4           # 最近多少轮完整保留（1 轮 = 1 问 + 1 答）
    max_chars: int = 6000         # 历史部分的字符上限
    older_user_chars: int = 300   # 早期轮次：用户问题保留多少字
    older_assistant_chars: int = 80  # 早期轮次：助手回答保留多少字
    max_summary_lines: int = 12   # 概要最多多少行

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "ContextPolicy":
        env = environ if environ is not None else os.environ

        def _int(key: str, default: int, low: int, high: int) -> int:
            try:
                value = int(float(env.get(key, str(default)) or default))
            except (TypeError, ValueError):
                return default
            return max(low, min(high, value))

        return cls(
            keep_turns=_int("AGENT_CONTEXT_KEEP_TURNS", 4, 1, 20),
            max_chars=_int("AGENT_CONTEXT_MAX_CHARS", 6000, 500, 40000),
            older_user_chars=_int("AGENT_CONTEXT_OLDER_USER_CHARS", 300, 40, 2000),
            older_assistant_chars=_int("AGENT_CONTEXT_OLDER_ASSISTANT_CHARS", 80, 0, 2000),
            max_summary_lines=_int("AGENT_CONTEXT_SUMMARY_LINES", 12, 1, 50),
        )


@dataclass
class AssembledContext:
    """装配结果：直接拼进 prompt 的三样东西。"""

    messages: list[dict[str, str]] = field(default_factory=list)
    summary: str = ""
    dropped_messages: int = 0
    truncated: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "messageCount": len(self.messages),
            "summaryChars": len(self.summary),
            "droppedMessages": self.dropped_messages,
            "truncated": self.truncated,
        }


def _clip(text: Any, limit: int) -> str:
    value = str(text or "").strip()
    if limit <= 0:
        return ""
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 1)] + "…"


def _pairs(history: Sequence[Mapping[str, Any]]) -> list[list[dict[str, str]]]:
    """把线性的消息列表切成「轮」。遇到 user 就开新轮。"""
    turns: list[list[dict[str, str]]] = []
    for raw in history:
        role = str(raw.get("role") or "").strip()
        content = str(raw.get("content") or "")
        if role not in {"user", "assistant", "system"}:
            continue
        item = {"role": role, "content": content}
        if role == "user" or not turns:
            turns.append([item])
        else:
            turns[-1].append(item)
    return turns


def assemble(
    history: Sequence[Mapping[str, Any]],
    *,
    policy: ContextPolicy | None = None,
    fallback_summary: str = "",
) -> AssembledContext:
    """把历史装配成「完整轮次 + 概要」。"""
    policy = policy or ContextPolicy()
    turns = _pairs(history)
    if not turns:
        return AssembledContext(summary=_clip(fallback_summary, policy.max_chars))

    recent = turns[-policy.keep_turns :]
    older = turns[: len(turns) - len(recent)]

    messages: list[dict[str, str]] = []
    for turn in recent:
        messages.extend(turn)

    lines: list[str] = []
    for turn in older:
        for item in turn:
            if item["role"] == "user":
                lines.append("早期提问：" + _clip(item["content"], policy.older_user_chars))
            elif item["role"] == "assistant" and policy.older_assistant_chars > 0:
                lines.append("当时回答：" + _clip(item["content"], policy.older_assistant_chars))

    # 概要只保留最近的若干行，且总长不超过预算的一半，给完整轮次留空间
    lines = lines[-policy.max_summary_lines :]
    summary_budget = max(200, policy.max_chars // 2)
    kept: list[str] = []
    used = 0
    truncated = False
    for line in reversed(lines):
        if used + len(line) > summary_budget:
            truncated = True
            break
        kept.append(line)
        used += len(line) + 1
    kept.reverse()
    summary = "\n".join(kept) or _clip(fallback_summary, summary_budget)

    # 完整轮次也受总预算约束：超了就从最旧的轮次开始丢
    dropped_recent = 0
    while messages:
        total = sum(len(m["content"]) for m in messages) + len(summary)
        if total <= policy.max_chars:
            break
        removed = [messages[0]]
        messages = messages[1:]
        if messages and messages[0]["role"] != "user":
            removed.append(messages[0])
            messages = messages[1:]
        dropped_recent += len(removed)
        truncated = True

    dropped_summary_lines = max(0, len(lines) - len(kept))
    return AssembledContext(
        messages=messages,
        summary=summary,
        dropped_messages=dropped_summary_lines + dropped_recent,
        truncated=truncated,
    )


def compress_for_summary(
    history: Sequence[Mapping[str, Any]],
    *,
    policy: ContextPolicy | None = None,
) -> str:
    """生成要写回 `session.summary` 的滚动概要（不是原始拼接）。"""
    policy = policy or ContextPolicy()
    turns = _pairs(history)
    lines: list[str] = []
    for turn in turns[-max(1, policy.keep_turns * 2) :]:
        for item in turn:
            if item["role"] == "user":
                lines.append("问：" + _clip(item["content"], policy.older_user_chars))
            elif item["role"] == "assistant":
                lines.append("答：" + _clip(item["content"], policy.older_assistant_chars))
    return "\n".join(lines)[-policy.max_chars :]
