"""Agent 层评测指标。

检索层已有 Recall / MRR，但检索好不等于回答好。这里补四个 Agent 层指标：

1. 结构化输出有效率（structured_valid_rate）—— 报告能不能被解析成预期结构
2. 引用支持率（citation_support_rate）—— 结论有没有证据支撑
3. 拒答指标（abstention）—— 分「误拒」和「漏拒」两个方向算，因为代价不对称
4. 端到端成功率（task_success_rate）—— 有多少任务产出了可用报告

设计取舍：这些函数只吃「已采集好的数据」，不做 IO，便于单测和在离线评测里复用。
"""

from __future__ import annotations

import json
import math
from typing import Any, Iterable, Mapping, Sequence


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(numerator / denominator, 4)


def _is_structured(payload: Any) -> bool:
    """验证规范化报告的必要字段和引用结构，不代替答案语义验证。"""
    if isinstance(payload, (bytes, bytearray)):
        try:
            payload = payload.decode("utf-8")
        except UnicodeDecodeError:
            return False
    if isinstance(payload, str):
        text = payload.strip()
        if not text:
            return False
        try:
            payload = json.loads(text)
        except (TypeError, ValueError):
            return False
    if not isinstance(payload, Mapping) or not isinstance(payload.get("answerable"), bool):
        return False
    for key in ("finalAnswer", "title"):
        if not isinstance(payload.get(key), str) or not payload[key].strip():
            return False
    for key in ("conclusions", "suggestions"):
        items = payload.get(key)
        if not isinstance(items, list) or any(not isinstance(item, str) or not item.strip() for item in items):
            return False
    if not payload["conclusions"]:
        return False
    evidence = payload.get("evidence")
    if not isinstance(evidence, list):
        return False
    if payload["answerable"] != bool(evidence):
        return False
    for item in evidence:
        if not isinstance(item, Mapping):
            return False
        if not isinstance(item.get("source"), str) or item["source"] not in {"ASR", "OCR"}:
            return False
        if not isinstance(item.get("content"), str) or not item["content"].strip():
            return False
        start = item.get("timestampMs", item.get("startMs"))
        end = item.get("endMs", start)
        for value in (start, end):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return False
            try:
                finite = math.isfinite(value)
            except OverflowError:
                return False
            if not finite or value < 0:
                return False
        if end < start:
            return False
    return True


def structured_valid_rate(payloads: Iterable[Any]) -> float | None:
    items = list(payloads)
    if not items:
        return None
    return _ratio(sum(1 for item in items if _is_structured(item)), len(items))


def citation_support_rate(supports: Iterable[bool]) -> float | None:
    flags = [bool(x) for x in supports]
    if not flags:
        return None
    return _ratio(sum(1 for f in flags if f), len(flags))


def abstention_metrics(samples: Iterable[Sequence[Any]]) -> dict[str, Any]:
    """拒答指标。

    samples 的每一项是 (answerable, abstained)：
        answerable=True  表示这道题视频里其实有答案（应该回答）
        abstained=True   表示系统选择了拒答

    返回：
        correctAbstain  该拒且拒了
        falseReject     不该拒却拒了（用户最烦这种）
        missedAbstain   该拒却没拒（系统开始编）
        precision       拒答的准确率 = correctAbstain / 所有拒答
        recall          该拒的覆盖率 = correctAbstain / 所有不可回答题
        falseRejectRate 误拒率 = falseReject / 所有可回答题
    """
    correct = false_reject = missed = answered_ok = 0
    total = 0
    for sample in samples:
        try:
            answerable, abstained = bool(sample[0]), bool(sample[1])
        except (IndexError, TypeError):
            continue
        total += 1
        if answerable and not abstained:
            answered_ok += 1
        elif answerable and abstained:
            false_reject += 1
        elif not answerable and abstained:
            correct += 1
        else:  # 不可回答却回答了
            missed += 1

    abstained_total = correct + false_reject
    unanswerable_total = correct + missed
    answerable_total = answered_ok + false_reject
    return {
        "samples": total,
        "answerableCaseCount": answerable_total,
        "unanswerableCaseCount": unanswerable_total,
        "correctAbstain": correct,
        "falseReject": false_reject,
        "missedAbstain": missed,
        "answeredOk": answered_ok,
        "abstentionPrecision": _ratio(correct, abstained_total),
        "abstentionRecall": _ratio(correct, unanswerable_total),
        "falseRejectRate": _ratio(false_reject, answerable_total),
        "missedAbstainRate": _ratio(missed, unanswerable_total),
    }


def task_success_rate(statuses: Iterable[str]) -> float | None:
    items = [str(s or "").upper() for s in statuses]
    if not items:
        return None
    ok = sum(1 for s in items if s in {"COMPLETED", "COMPLETED_REUSED", "SUCCESS", "OK"})
    return _ratio(ok, len(items))


def summarize(
    *,
    payloads: Iterable[Any] | None = None,
    citation_supports: Iterable[bool] | None = None,
    abstention_samples: Iterable[Sequence[Any]] | None = None,
    statuses: Iterable[str] | None = None,
) -> dict[str, Any]:
    """把四类指标合成一份可以直接写进任务记录/评测报告的结构。"""
    report: dict[str, Any] = {
        "structuredValidRate": structured_valid_rate(payloads or []),
        "citationSupportRate": citation_support_rate(citation_supports or []),
        "taskSuccessRate": task_success_rate(statuses or []),
    }
    report["abstention"] = abstention_metrics(abstention_samples or [])
    return report
