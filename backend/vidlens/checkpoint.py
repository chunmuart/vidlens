"""Opt-in Agent stage checkpoints.

Only the Planner and Retriever stages are persisted. Both are deterministic
given ``(goal, prompt version, evidence snapshot)``, so replaying them after a
transient failure is safe. Verifier/Writer/Critic output is deliberately not
checkpointed: a half-reviewed report must not be reused without a fresh
verification pass.

``CHECKPOINT_ENABLED`` defaults to ``0``. When disabled, ``load()`` returns
``None`` and ``save()`` is a no-op, so no database read or write happens.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Callable, Mapping

log = logging.getLogger("vidlens.checkpoint")

CHECKPOINT_VERSION = 1
SUPPORTED_STAGES = ("planner", "retriever")

PersistCallback = Callable[[dict[str, Any]], None]


def checkpoint_enabled(environ: Mapping[str, str] | None = None) -> bool:
    env = environ if environ is not None else os.environ
    raw = str(env.get("CHECKPOINT_ENABLED", "0") or "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def evidence_fingerprint(segments: list[dict[str, Any]]) -> str:
    """Stable hash of the evidence content, independent of database row IDs.

    ``replace_evidence`` re-inserts rows with new numeric IDs while keeping the
    same content; excluding ``segmentId`` lets a checkpoint survive that.
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
    """One task's checkpoint snapshot, with validation on load."""

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
        except Exception:  # noqa: BLE001 -- a checkpoint failure must not fail the task
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
