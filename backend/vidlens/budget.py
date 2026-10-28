"""Agent 执行预算：轮次、token、耗时、成本四道闸。

设计原则：
1. **默认不限制**——所有上限为 0 时 `active` 为 False，行为与未引入本模块完全一致，
   避免上线即改变既有链路。
2. **只报第一个触发的限制**——`check()` 抛 `BudgetExceeded` 时带上具体是哪一项，
   便于任务记录里写清「为什么停」。
3. **不依赖任何第三方库**，方便单测和复用。

典型用法::

    tracker = BudgetTracker(BudgetLimits.from_env()).start()
    tracker.add_round()
    tracker.add_tokens(prompt_tokens=1200, completion_tokens=300, cost_usd=0.0012)
    tracker.check()          # 超限抛 BudgetExceeded

对应环境变量（与 DoVideoAI 的命名保持一致，便于对比讲解）：
    AGENT_MAX_ROUNDS              最大轮次
    AGENT_MAX_ESTIMATED_TOKENS    最大 token
    AGENT_MAX_DURATION_MS         最大耗时（毫秒）
    AGENT_MAX_ESTIMATED_COST      最大估算成本（美元）
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping


class BudgetExceeded(RuntimeError):
    """预算超限。携带具体是哪一项、用了多少、上限多少。"""

    def __init__(self, limit: str, *, used: float, limit_value: float) -> None:
        self.limit = limit
        self.used = used
        self.limit_value = limit_value
        super().__init__(f"预算超限：{limit}（used={used}, limit={limit_value}）")


@dataclass(frozen=True)
class BudgetLimits:
    """四项上限，0 表示不限制该项。"""

    max_rounds: int = 0
    max_tokens: int = 0
    max_duration_ms: int = 0
    max_cost_usd: float = 0.0

    @property
    def active(self) -> bool:
        return any(
            (
                self.max_rounds > 0,
                self.max_tokens > 0,
                self.max_duration_ms > 0,
                self.max_cost_usd > 0,
            )
        )

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "BudgetLimits":
        env = environ if environ is not None else os.environ
        return cls(
            max_rounds=_env_int(env, "AGENT_MAX_ROUNDS"),
            max_tokens=_env_int(env, "AGENT_MAX_ESTIMATED_TOKENS"),
            max_duration_ms=_env_int(env, "AGENT_MAX_DURATION_MS"),
            max_cost_usd=_env_float(env, "AGENT_MAX_ESTIMATED_COST"),
        )


class BudgetTracker:
    """累计用量并在 `check()` 时判定是否超限。"""

    def __init__(
        self,
        limits: BudgetLimits | None = None,
        *,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.limits = limits or BudgetLimits()
        self.rounds = 0
        self.tokens = 0
        self.cost_usd = 0.0
        self._clock = clock or time.monotonic
        self._started: float | None = None

    def start(self) -> "BudgetTracker":
        self._started = self._clock()
        return self

    @property
    def elapsed_ms(self) -> int:
        if self._started is None:
            return 0
        return max(0, int((self._clock() - self._started) * 1000))

    def add_round(self, count: int = 1) -> "BudgetTracker":
        self.rounds += max(0, int(count))
        return self

    def add_tokens(
        self,
        *,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int | None = None,
        cost_usd: float | None = None,
    ) -> "BudgetTracker":
        if total_tokens is None:
            total_tokens = max(0, int(prompt_tokens)) + max(0, int(completion_tokens))
        self.tokens += max(0, int(total_tokens))
        if cost_usd is not None:
            self.cost_usd += max(0.0, float(cost_usd))
        return self

    def add_cost(self, usd: float) -> "BudgetTracker":
        self.cost_usd += max(0.0, float(usd))
        return self

    def check(self) -> None:
        """任一项超限抛 `BudgetExceeded`；全部未超则静默返回。"""
        limits = self.limits
        # 先查轮次：最便宜，也最容易先撞线
        if limits.max_rounds > 0 and self.rounds > limits.max_rounds:
            raise BudgetExceeded("rounds", used=self.rounds, limit_value=limits.max_rounds)
        if limits.max_duration_ms > 0 and self.elapsed_ms > limits.max_duration_ms:
            raise BudgetExceeded(
                "duration_ms", used=self.elapsed_ms, limit_value=limits.max_duration_ms
            )
        if limits.max_tokens > 0 and self.tokens > limits.max_tokens:
            raise BudgetExceeded("tokens", used=self.tokens, limit_value=limits.max_tokens)
        if limits.max_cost_usd > 0 and self.cost_usd > limits.max_cost_usd:
            raise BudgetExceeded("cost_usd", used=self.cost_usd, limit_value=limits.max_cost_usd)

    def snapshot(self) -> dict[str, Any]:
        """写入任务记录用的快照，便于事后分析「是不是被预算截断的」。"""
        return {
            "rounds": self.rounds,
            "tokens": self.tokens,
            "costUsd": round(self.cost_usd, 8),
            "elapsedMs": self.elapsed_ms,
            "limits": {
                "maxRounds": self.limits.max_rounds,
                "maxTokens": self.limits.max_tokens,
                "maxDurationMs": self.limits.max_duration_ms,
                "maxCostUsd": self.limits.max_cost_usd,
            },
        }


def record_completion_usage(
    tracker: BudgetTracker,
    *,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    total_tokens: int | None = None,
    cost_usd: float | None = None,
) -> None:
    """Record one successful model call and enforce the token/cost gates."""
    tracker.add_tokens(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        cost_usd=cost_usd,
    )
    tracker.check()


def budget_refusal_payload(exc: BudgetExceeded) -> dict[str, Any]:
    """A safe, explicitly non-answerable report used when a budget stops the run."""
    return {
        "answerable": False,
        "finalAnswer": "本次分析因预算超限提前结束，无法从视频确定答案。",
        "title": "视频分析未完成（预算超限）",
        "conclusions": [f"预算超限：{exc.limit}"],
        "evidence": [],
        "suggestions": ["可在配置中提高 AGENT_MAX_* 预算后重试。"],
    }


def _env_int(env: Mapping[str, str], key: str) -> int:
    raw = str(env.get(key, "0") or "0").strip()
    try:
        return max(0, int(float(raw)))
    except (TypeError, ValueError):
        return 0


def _env_float(env: Mapping[str, str], key: str) -> float:
    raw = str(env.get(key, "0") or "0").strip()
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return 0.0
