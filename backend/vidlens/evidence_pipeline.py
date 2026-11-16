"""Two bounded evidence branches with independently saved, versioned artifacts.

Only plain dictionaries cross threads; database sessions stay in the caller.
Files live on the existing shared upload volume. This is a local recovery cache,
not a cross-process exactly-once execution protocol.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from .evidence_api import EvidenceAPIError


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # A content-hash destination can already approach Windows' path limit.
    temporary = path.with_name('.tmp-' + uuid.uuid4().hex[:16])
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')
        # Windows readers/antivirus can briefly deny replacement even though
        # the file is writable. Retain the complete temporary file while
        # retrying only these transient sharing/access errors.
        for attempt in range(6):
            try:
                os.replace(temporary, path)
                break
            except PermissionError as exc:
                if os.name != 'nt' or getattr(exc, 'winerror', None) not in (5, 32, 33) or attempt == 5:
                    raise
                time.sleep(0.025 * (2 ** attempt))
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


class EvidenceArtifacts:
    def __init__(self, root: Path, source: Path, configuration: dict[str, Any]):
        stat = source.stat()
        identity = str(source.resolve())
        self.directory = root / hashlib.sha256(identity.encode()).hexdigest()
        self._source = {'path': identity, 'size': stat.st_size, 'mtimeNs': stat.st_mtime_ns}
        self._configuration = configuration

    def fingerprint(self, branch: str) -> str:
        payload = {'version': 2, 'source': self._source, 'config': self._configuration[branch]}
        return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    def load(self, branch: str) -> dict[str, Any]:
        saved = read_json(self.directory / f'{branch.lower()}.json')
        return saved if saved.get('fingerprint') == self.fingerprint(branch) else {}

    def save(self, branch: str, payload: dict[str, Any]) -> None:
        atomic_json(self.directory / f'{branch.lower()}.json', {
            **payload, 'fingerprint': self.fingerprint(branch), 'updatedAt': time.time(),
        })

    def pending_task(self) -> str | None:
        pending = self.load('ASR')
        return str(pending['taskId']) if pending.get('status') == 'PENDING' and pending.get('taskId') else None

    def submitted(self, task_id: str) -> None:
        self.save('ASR', {'status': 'PENDING', 'taskId': task_id})


class EvidenceBranchError(RuntimeError):
    def __init__(self, branches: list[str]):
        self.branches = branches
        super().__init__('证据提取失败：' + '、'.join(branches) + '；成功分路已保存，重试会复用。')


def _failure_metadata(exc: Exception) -> dict[str, Any]:
    """Keep known billable work without persisting exception bodies or text."""
    metadata: dict[str, Any] = {}
    if isinstance(exc, EvidenceAPIError):
        metadata.update({
            'errorCode': exc.code,
            'taskId': exc.task_id,
            'retryable': exc.retryable,
            # Missing provider usage remains unknown; never invent a zero fee.
            'usage': dict(exc.usage),
        })
    completed = getattr(exc, 'completed_request_events', None)
    if isinstance(completed, list):
        allowed = {'provider', 'model', 'status', 'usage', 'reused', 'taskId'}
        metadata['requestEvents'] = [
            {key: value for key, value in event.items() if key in allowed}
            for event in completed if isinstance(event, dict)
        ]
    return metadata


def run_evidence_branches(
    jobs: dict[str, Callable[[], dict[str, Any]]],
    *,
    artifacts: EvidenceArtifacts,
    parallel: bool,
    on_status: Callable[[str, dict[str, Any]], None] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    failures: list[str] = []
    status_lock = threading.Lock()

    def execute(branch: str) -> dict[str, Any]:
        saved = artifacts.load(branch)
        if saved.get('status') in {'SUCCESS', 'NO_TEXT', 'SKIPPED'}:
            return {**saved, 'reused': True, 'durationMs': 0}
        started = time.perf_counter()
        try:
            value = jobs[branch]()
            segments = list(value.get('segments') or [])
            status = value.get('status') or ('SUCCESS' if segments else 'NO_TEXT')
            if status not in {'SUCCESS', 'NO_TEXT', 'SKIPPED'}:
                raise ValueError('Evidence branch did not finish successfully')
            result = {**value, 'segments': segments, 'status': status,
                      'durationMs': int((time.perf_counter() - started) * 1000), 'reused': False}
            artifacts.save(branch, result)
            return result
        except Exception as exc:
            # Keep a submitted remote task so a timeout can resume polling it.
            pending = artifacts.load(branch)
            result = {'status': 'FAILED', 'errorType': type(exc).__name__,
                      'durationMs': int((time.perf_counter() - started) * 1000), 'reused': False,
                      **_failure_metadata(exc)}
            if pending.get('status') != 'PENDING' or not getattr(exc, 'retryable', False):
                artifacts.save(branch, result)
            return result

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=2 if parallel else 1, thread_name_prefix='evidence') as pool:
        futures = {pool.submit(execute, branch): branch for branch in jobs}
        for future in as_completed(futures):
            branch = futures[future]
            result = future.result()
            results[branch] = result
            if result['status'] == 'FAILED':
                failures.append(branch)
            if on_status is not None:
                with status_lock:
                    on_status(branch, result)
    manifest = {'parallel': parallel, 'durationMs': int((time.perf_counter() - started) * 1000),
                'branches': {key: {k: v for k, v in item.items() if k != 'segments'}
                             for key, item in results.items()}}
    atomic_json(artifacts.directory / 'latest.json', manifest)
    if failures:
        raise EvidenceBranchError(sorted(failures))
    segments = [segment for branch in sorted(results) for segment in results[branch].get('segments', [])]
    return sorted(segments, key=lambda item: (item['startMs'], item['source'])), manifest
