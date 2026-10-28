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
from typing import Any, Iterable, Mapping, Sequence


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(numerator / denominator, 4)


def _is_structured(payload: Any) -> bool:
    """判断一份报告是不是结构化的。

    接受两种输入：dict，或能被 json.loads 解析成 dict 的字符串。
    列表和其它类型一律视为不合法——报告必须是一个对象。
    """
    if isinstance(payload, Mapping):
        return True
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
            return isinstance(json.loads(text), dict)
        except (TypeError, ValueError):
            return False
    return False


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
    return {
        "samples": total,
        "correctAbstain": correct,
        "falseReject": false_reject,
        "missedAbstain": missed,
        "answeredOk": answered_ok,
        "abstentionPrecision": _ratio(correct, abstained_total),
        "abstentionRecall": _ratio(correct, unanswerable_total),
        "falseRejectRate": _ratio(false_reject, total),
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
