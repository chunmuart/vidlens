"""Build a zero-dependency span tree from an existing analysis trace.

The trace already records stage durations, tool calls, and provider request
metadata. This module only reshapes that data; it does not measure anything
new and does not change any existing trace field.

The result is a logical tree, not a wall-clock flame graph:
``stageDurationMs`` has no start timestamps, so child offsets are laid out
sequentially and marked with ``offsetApproximation`` where needed.
"""

from __future__ import annotations

from typing import Any, Mapping

ROOT_SPAN = "analysis"


def _int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError, OverflowError):
        return 0


def _span(
    name: str,
    parent: str | None,
    start_offset_ms: int,
    duration_ms: int,
    **attributes: Any,
) -> dict[str, Any]:
    return {
        "name": str(name),
        "parent": parent,
        "startOffsetMs": max(0, int(start_offset_ms)),
        "durationMs": max(0, int(duration_ms)),
        "attributes": attributes,
    }


def build_trace_spans(trace: Mapping[str, Any]) -> list[dict[str, Any]]:
    stages = trace.get("stageDurationMs") or {}
    if not isinstance(stages, Mapping):
        stages = {}
    video_ms = _int(stages.get("VIDEO_CONTEXT"))
    planner_ms = _int(stages.get("PLANNER"))
    executor_ms = _int(stages.get("EXECUTOR"))
    critic_ms = _int(stages.get("CRITIC"))

    raw_tool_calls = trace.get("toolCalls") or []
    tool_calls = raw_tool_calls if isinstance(raw_tool_calls, list) else []
    raw_provider_requests = trace.get("providerRequests") or []
    provider_requests = (
        raw_provider_requests if isinstance(raw_provider_requests, list) else []
    )

    tool_total_ms = sum(
        _int(item.get("durationMs"))
        for item in tool_calls
        if isinstance(item, Mapping)
    )
    provider_total_ms = sum(
        _int(item.get("latencyMs"))
        for item in provider_requests
        if isinstance(item, Mapping)
    )
    stage_total_ms = video_ms + planner_ms + executor_ms
    total_ms = max(stage_total_ms, tool_total_ms, provider_total_ms)

    spans: list[dict[str, Any]] = [
        _span(
            ROOT_SPAN,
            None,
            0,
            total_ms,
            agentMode=trace.get("agentMode"),
            provider=trace.get("provider"),
            attempt=_int(trace.get("attempt")),
            evidenceSegmentCount=_int(trace.get("evidenceSegmentCount")),
            toolCallCount=len(tool_calls),
            providerRequestCount=len(provider_requests),
            layout="logical-tree",
        )
    ]

    planner_start = video_ms
    executor_start = video_ms + planner_ms
    if video_ms:
        spans.append(_span("video_context", ROOT_SPAN, 0, video_ms))
    if planner_ms:
        spans.append(_span("planner", ROOT_SPAN, planner_start, planner_ms))
    if executor_ms:
        spans.append(
            _span(
                "executor",
                ROOT_SPAN,
                executor_start,
                executor_ms,
                toolCallCount=len(tool_calls),
            )
        )
    if critic_ms:
        # CRITIC is measured inside EXECUTOR; place it at the end of that span.
        spans.append(
            _span(
                "critic",
                "executor" if executor_ms else ROOT_SPAN,
                executor_start + max(0, executor_ms - critic_ms),
                min(critic_ms, executor_ms) if executor_ms else critic_ms,
                approximated=True,
            )
        )

    tool_offset = executor_start
    for index, item in enumerate(tool_calls, start=1):
        if not isinstance(item, Mapping):
            continue
        duration_ms = _int(item.get("durationMs"))
        spans.append(
            _span(
                f"tool.{item.get('tool') or 'unknown'}#{index}",
                "executor" if executor_ms else ROOT_SPAN,
                tool_offset,
                duration_ms,
                tool=item.get("tool"),
                index=_int(item.get("index")) or index,
                success=bool(item.get("success")),
                errorClass=item.get("errorClass"),
                offsetApproximation=True,
            )
        )
        tool_offset += duration_ms

    if provider_requests:
        spans.append(
            _span(
                "provider",
                ROOT_SPAN,
                0,
                provider_total_ms,
                requestCount=len(provider_requests),
                layout="sequential-approximation",
            )
        )
        request_offset = 0
        for index, item in enumerate(provider_requests, start=1):
            if not isinstance(item, Mapping):
                continue
            duration_ms = _int(item.get("latencyMs"))
            spans.append(
                _span(
                    f"provider.{item.get('phase') or 'UNKNOWN'}#{index}",
                    "provider",
                    request_offset,
                    duration_ms,
                    phase=item.get("phase"),
                    model=item.get("model"),
                    success=bool(item.get("success")),
                    statusCode=_int(item.get("statusCode")),
                    promptTokens=_int(item.get("promptTokens")),
                    completionTokens=_int(item.get("completionTokens")),
                    totalTokens=_int(item.get("totalTokens")),
                    estimatedCostUsd=item.get("estimatedCostUsd"),
                    toolCallCount=_int(item.get("toolCallCount")),
                    attempt=_int(item.get("attempt")),
                    offsetApproximation=True,
                )
            )
            request_offset += duration_ms

    return spans


def attach_trace_spans(trace: dict[str, Any]) -> dict[str, Any]:
    """Add ``trace["spans"]`` only; every existing key is left untouched."""
    trace["spans"] = build_trace_spans(trace)
    return trace
