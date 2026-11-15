"""可选的 Agent 阶段 checkpoint。

只持久化 Planner 和 Retriever 两个阶段，在目标、prompt 版本和证据快照
仍兼容时复用已完成产物。Planner 的模型输出不保证确定性。Verifier/Writer/Critic
的产物故意不存：半份已评审的报告不能在未重新验证的情况下复用。

`CHECKPOINT_ENABLED` 默认是 `0`。关闭时 `load()` 返回 `None`、`save()` 是空操作，
不会发生任何数据库读写。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Callable, Mapping

log = logging.getLogger("vidlens.checkpoint")

# v1 的公共元数据可能使旧 Retriever 被错误重新标记为有效，保守地整体失效。
CHECKPOINT_VERSION = 2
SUPPORTED_STAGES = ("planner", "retriever")

PersistCallback = Callable[[dict[str, Any]], None]


def checkpoint_enabled(environ: Mapping[str, str] | None = None) -> bool:
    env = environ if environ is not None else os.environ
    raw = str(env.get("CHECKPOINT_ENABLED", "0") or "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def evidence_fingerprint(segments: list[dict[str, Any]]) -> str:
    """证据内容的稳定哈希，不依赖数据库行 ID。

    `replace_evidence` 会删除后重新插入证据、生成新的数字 ID，但内容不变；
    指纹排除 `segmentId`，就是为了让 checkpoint 在这种重建后仍然可用。
    """
    canonical = sorted(
        (
            {
                "source": str(item.get("source", "")).upper(),
                "startMs": max(0, int(item.get("startMs", 0))),
                "endMs": max(0, int(item.get("endMs", item.get("startMs", 0)))),
                "content": str(item.get("content", "")),
            }
            for item in segments
        ),
        key=lambda item: (
            item["startMs"],
            item["endMs"],
            item["source"],
            item["content"],
        ),
    )
    rendered = json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


class AgentCheckpoint:
    """一个任务的 checkpoint 快照，加载时做完整性校验。"""

    def __init__(
        self,
        *,
        enabled: bool,
        goal: str,
        prompt_version: str,
        evidence_fingerprint: str,
        attempt: int = 0,
        data: Mapping[str, Any] | None = None,
        persist: PersistCallback | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.goal = " ".join(str(goal).split())
        self.prompt_version = str(prompt_version)
        self.evidence_fingerprint = str(evidence_fingerprint)
        self.attempt = max(0, int(attempt))
        self._data = dict(data) if isinstance(data, Mapping) else {}
        self._persist = persist
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self.restored_stages: list[str] = []
        self.saved_stages: list[str] = []

    @classmethod
    def disabled(cls) -> "AgentCheckpoint":
        return cls(enabled=False, goal="", prompt_version="", evidence_fingerprint="")

    def _valid(self) -> bool:
        if not self.enabled:
            return False
        try:
            version = int(self._data.get("version") or 0)
        except (TypeError, ValueError):
            return False
        return (
            version == CHECKPOINT_VERSION
            and str(self._data.get("goal") or "") == self.goal
            and str(self._data.get("promptVersion") or "") == self.prompt_version
        )

    def load(self, stage: str) -> dict[str, Any] | None:
        if not self._valid() or stage not in SUPPORTED_STAGES:
            return None
        payload = self._data.get(stage)
        if not isinstance(payload, dict):
            return None
        if (
            stage == "retriever"
            and str(self._data.get("evidenceFingerprint") or "") != self.evidence_fingerprint
        ):
            log.info("agent_checkpoint_evidence_changed goal=%s", self.goal[:80])
            return None
        if stage not in self.restored_stages:
            self.restored_stages.append(stage)
        return json.loads(json.dumps(payload, ensure_ascii=False, default=str))

    def save(self, stage: str, payload: dict[str, Any]) -> None:
        if not self.enabled or stage not in SUPPORTED_STAGES:
            return
        # 先清理旧阶段，再写新元数据。不能用新 goal/prompt/fingerprint
        # 给旧阶段「换标签」，否则写完 Planner 后崩溃会恢复到旧 Retriever。
        if not self._valid():
            self._data = {}
        elif str(self._data.get("evidenceFingerprint") or "") != self.evidence_fingerprint:
            self._data.pop("retriever", None)
        if stage == "planner":
            self._data.pop("retriever", None)
        self._data = {
            **self._data,
            "version": CHECKPOINT_VERSION,
            "goal": self.goal,
            "promptVersion": self.prompt_version,
            "evidenceFingerprint": self.evidence_fingerprint,
            "attempt": self.attempt,
            "updatedAt": self._clock().isoformat(),
            stage: json.loads(json.dumps(payload, ensure_ascii=False, default=str)),
        }
        if stage not in self.saved_stages:
            self.saved_stages.append(stage)
        if self._persist is None:
            return
        try:
            self._persist(self.snapshot())
        except Exception:  # noqa: BLE001 —— checkpoint 写失败不能拖垮主任务
            log.exception("agent_checkpoint_persist_failed stage=%s", stage)

    def snapshot(self) -> dict[str, Any]:
        return json.loads(json.dumps(self._data, ensure_ascii=False, default=str))

    def summary(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "version": CHECKPOINT_VERSION,
            "restoredStages": list(self.restored_stages),
            "savedStages": list(self.saved_stages),
            "promptVersion": self.prompt_version,
        }
