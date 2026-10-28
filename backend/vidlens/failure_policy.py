"""工具失败策略：把异常分成「重试 / 降级 / 放弃」三类。

为什么需要它：
同一条 Agent 链路上，不同工具抛的异常含义不一样——网络超时重试有意义，
参数写错重试一万次也没用。如果没有统一分类，代码里就会到处是
「有的地方吞掉、有的地方直接抛、有的地方重试三次」，排查时无从下手。

三条处置：
    RETRY    可重试：网络抖动、超时、限流、503 —— 退避后重试
    DEGRADE  降级：重试用尽或未知错误 —— 走备用路径，返回部分结果
    ABORT    放弃：参数/校验错误 —— 重试无意义，直接失败

用法::

    action, delay_ms = decide_failure(exc, attempt=1, max_attempts=3)
"""

from __future__ import annotations

import random
from enum import Enum
from typing import Any, Callable


class FailureAction(str, Enum):
    RETRY = "RETRY"
    DEGRADE = "DEGRADE"
    ABORT = "ABORT"


# 参数/校验类错误：重试无意义
_ABORT_TYPES = (ValueError, TypeError, KeyError, AttributeError, AssertionError)
# 网络/IO 类错误：值得重试
_RETRY_TYPES = (TimeoutError, ConnectionError, OSError)

# 兜底判据：异常消息里的关键词
_ABORT_HINTS = (
    "invalid",
    "not found",
    "unknown",
    "missing",
    "schema",
    "validation",
    "参数",
    "未知",
    "不存在",
    "校验",
)
_RETRY_HINTS = (
    "timeout",
    "timed out",
    "time out",
    "connection reset",
    "connection aborted",
    "connection refused",
    "temporarily unavailable",
    "rate limit",
    "too many requests",
    "429",
    "502",
    "503",
    "504",
    "超时",
    "限流",
)


def classify_failure(exc: BaseException) -> FailureAction:
    """把异常归到三类之一。

    判定顺序：先看异常类型（最可靠），再看消息关键词，最后默认 DEGRADE。
    默认不是 ABORT，是因为抛出去的代价比降级大——降级至少还能给用户一个部分结果。
    """
    if isinstance(exc, _ABORT_TYPES):
        return FailureAction.ABORT
    if isinstance(exc, _RETRY_TYPES):
        return FailureAction.RETRY

    message = str(exc).lower()
    if any(hint in message for hint in _ABORT_HINTS):
        return FailureAction.ABORT
    if any(hint in message for hint in _RETRY_HINTS):
        return FailureAction.RETRY
    return FailureAction.DEGRADE


def backoff_delay_ms(
    attempt: int,
    *,
    base_ms: int = 1000,
    factor: float = 2.0,
    cap_ms: int = 30000,
    jitter: float = 0.2,
    rng: Callable[[], float] | None = None,
) -> int:
    """指数退避 + 抖动。

    attempt 从 1 开始：第 1 次重试等 base_ms，第 2 次 base*factor……
    抖动避免多个任务同时失败后齐步重试（惊群）。
    """
    attempt = max(1, int(attempt))
    raw = base_ms * (factor ** (attempt - 1))
    capped = min(float(cap_ms), raw)
    if jitter <= 0:
        return int(capped)
    random_value = (rng or random.random)()
    # 在 [1-jitter, 1+jitter] 区间抖动
    return int(max(0.0, capped * (1.0 - jitter + 2.0 * jitter * random_value)))


def decide_failure(
    exc: BaseException,
    *,
    attempt: int = 1,
    max_attempts: int = 3,
) -> tuple[FailureAction, int]:
    """返回「该怎么做」和「如果要重试，等多久（毫秒）」。

    重试次数用尽后不再 RETRY，改为 DEGRADE——给用户部分结果，
    而不是把异常一路抛到最外层变成「任务失败」。
    """
    action = classify_failure(exc)
    if action is FailureAction.RETRY and attempt >= max(1, int(max_attempts)):
        return FailureAction.DEGRADE, 0
    if action is FailureAction.RETRY:
        return action, backoff_delay_ms(attempt)
    return action, 0


def describe_failure(exc: BaseException, *, attempt: int = 1, max_attempts: int = 3) -> dict[str, Any]:
    """给工具返回值用的结构化描述，前端/日志可以直接展示。"""
    action, delay_ms = decide_failure(exc, attempt=attempt, max_attempts=max_attempts)
    return {
        "error": str(exc) or exc.__class__.__name__,
        "errorType": exc.__class__.__name__,
        "errorClass": action.value,
        "retryable": action is FailureAction.RETRY,
        "retryDelayMs": delay_ms,
    }
