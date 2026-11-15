from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from vidlens import evidence_pipeline
from vidlens.evidence_api import EvidenceAPIError
from vidlens.evidence_pipeline import EvidenceArtifacts, EvidenceBranchError, atomic_json, read_json, run_evidence_branches


def artifacts_for(tmp_path: Path, *, asr_model: str = "asr-v1", ocr_model: str = "ocr-v1") -> EvidenceArtifacts:
    source = tmp_path / "video.mp4"
    if not source.exists():
        source.write_bytes(b"immutable-test-video")
    return EvidenceArtifacts(tmp_path / "artifacts", source, {
        "ASR": {"model": asr_model}, "OCR": {"model": ocr_model},
    })


def segment(source: str) -> dict:
    return {"source": source, "startMs": 100, "endMs": 200, "content": source + " text"}


def test_successful_asr_is_saved_before_ocr_failure_and_only_ocr_reruns(tmp_path: Path) -> None:
    artifacts = artifacts_for(tmp_path)
    calls = {"ASR": 0, "OCR": 0}

    def asr() -> dict:
        calls["ASR"] += 1
        return {"segments": [segment("ASR")]}

    def ocr() -> dict:
        calls["OCR"] += 1
        if calls["OCR"] == 1:
            raise TimeoutError("OCR unavailable")
        return {"segments": [segment("OCR")]}

    with pytest.raises(EvidenceBranchError) as failure:
        run_evidence_branches({"ASR": asr, "OCR": ocr}, artifacts=artifacts, parallel=True)
    assert failure.value.branches == ["OCR"]
    assert artifacts.load("ASR")["status"] == "SUCCESS"
    assert artifacts.load("OCR")["status"] == "FAILED"
    manifest = read_json(artifacts.directory / "latest.json")
    assert manifest["branches"]["OCR"]["errorType"] == "TimeoutError"
    assert "segments" not in manifest["branches"]["ASR"]

    segments, manifest = run_evidence_branches({"ASR": asr, "OCR": ocr}, artifacts=artifacts, parallel=True)
    assert calls == {"ASR": 1, "OCR": 2}
    assert {item["source"] for item in segments} == {"ASR", "OCR"}
    assert manifest["branches"]["ASR"]["reused"] is True
    assert manifest["branches"]["OCR"]["reused"] is False


def test_no_text_and_skipped_are_terminal_cached_results(tmp_path: Path) -> None:
    artifacts = artifacts_for(tmp_path)
    jobs = {"ASR": lambda: {"segments": []}, "OCR": lambda: {"status": "SKIPPED", "segments": []}}
    segments, first = run_evidence_branches(jobs, artifacts=artifacts, parallel=False)
    assert segments == []
    assert first["branches"]["ASR"]["status"] == "NO_TEXT"
    assert first["branches"]["OCR"]["status"] == "SKIPPED"

    def must_not_run() -> dict:
        raise AssertionError("terminal branch should be reused")

    segments, second = run_evidence_branches(
        {"ASR": must_not_run, "OCR": must_not_run}, artifacts=artifacts, parallel=True,
    )
    assert segments == []
    assert all(item["reused"] for item in second["branches"].values())


def test_configuration_change_only_invalidates_affected_branch(tmp_path: Path) -> None:
    calls = {"ASR": 0, "OCR": 0}

    def job(branch: str) -> dict:
        calls[branch] += 1
        return {"segments": [segment(branch)]}

    jobs = {"ASR": lambda: job("ASR"), "OCR": lambda: job("OCR")}
    run_evidence_branches(jobs, artifacts=artifacts_for(tmp_path), parallel=True)
    run_evidence_branches(jobs, artifacts=artifacts_for(tmp_path, asr_model="asr-v2"), parallel=True)
    assert calls == {"ASR": 2, "OCR": 1}
    run_evidence_branches(
        jobs, artifacts=artifacts_for(tmp_path, asr_model="asr-v2", ocr_model="ocr-v2"), parallel=True,
    )
    assert calls == {"ASR": 2, "OCR": 2}


def test_parallel_jobs_overlap_and_asr_is_persisted_while_ocr_still_running(tmp_path: Path) -> None:
    artifacts = artifacts_for(tmp_path)
    started_together = threading.Barrier(2, timeout=3)
    asr_persisted = threading.Event()
    timings: dict[str, tuple[float, float]] = {}

    def asr() -> dict:
        start = time.perf_counter_ns()
        started_together.wait()
        timings["ASR"] = (start, time.perf_counter_ns())
        return {"segments": [segment("ASR")]}

    def ocr() -> dict:
        start = time.perf_counter_ns()
        started_together.wait()
        assert asr_persisted.wait(timeout=3)
        assert artifacts.load("ASR")["status"] == "SUCCESS"
        assert artifacts.load("OCR") == {}
        timings["OCR"] = (start, time.perf_counter_ns())
        return {"segments": [segment("OCR")]}

    def on_status(branch: str, result: dict) -> None:
        if branch == "ASR" and result["status"] == "SUCCESS":
            asr_persisted.set()

    run_evidence_branches({"ASR": asr, "OCR": ocr}, artifacts=artifacts, parallel=True, on_status=on_status)
    assert max(item[0] for item in timings.values()) < min(item[1] for item in timings.values())


def test_pending_asr_timeout_keeps_remote_task_and_resumes_polling_on_retry(tmp_path: Path) -> None:
    artifacts = artifacts_for(tmp_path)
    submissions = 0
    polls = []
    ocr_calls = 0

    def asr() -> dict:
        nonlocal submissions
        task_id = artifacts.pending_task()
        if task_id is None:
            submissions += 1
            task_id = "already-submitted-task"
            artifacts.submitted(task_id)
        polls.append(task_id)
        if len(polls) == 1:
            raise EvidenceAPIError(
                "ASR_POLL_TIMEOUT", task_id=task_id, retryable=True, usage={"duration": 12.5},
            )
        return {"segments": [segment("ASR")]}

    def ocr() -> dict:
        nonlocal ocr_calls
        ocr_calls += 1
        return {"segments": [segment("OCR")]}

    jobs = {"ASR": asr, "OCR": ocr}
    with pytest.raises(EvidenceBranchError):
        run_evidence_branches(jobs, artifacts=artifacts, parallel=True)
    assert artifacts.pending_task() == "already-submitted-task"
    assert artifacts.load("ASR")["status"] == "PENDING"
    failed = read_json(artifacts.directory / "latest.json")["branches"]["ASR"]
    assert failed["errorCode"] == "ASR_POLL_TIMEOUT"
    assert failed["taskId"] == "already-submitted-task"
    assert failed["retryable"] is True
    assert failed["usage"] == {"duration": 12.5}
    # 重新构造缓存对象，模拟进程重启后只读取持久文件。
    recovered = artifacts_for(tmp_path)
    _, manifest = run_evidence_branches(jobs, artifacts=recovered, parallel=True)
    assert submissions == 1
    assert polls == ["already-submitted-task", "already-submitted-task"]
    assert ocr_calls == 1
    assert recovered.pending_task() is None
    assert manifest["branches"]["OCR"]["reused"] is True


def test_permanent_remote_failure_clears_pending_task(tmp_path: Path) -> None:
    artifacts = artifacts_for(tmp_path)
    artifacts.submitted("failed-remote-task")

    def permanently_failed() -> dict:
        raise EvidenceAPIError("ASR_TASK_FAILED", task_id="failed-remote-task", retryable=False)

    with pytest.raises(EvidenceBranchError):
        run_evidence_branches({"ASR": permanently_failed}, artifacts=artifacts, parallel=True)
    assert artifacts.pending_task() is None
    assert artifacts.load("ASR")["status"] == "FAILED"


def test_failed_api_request_retains_usage_and_completed_frame_events(tmp_path: Path) -> None:
    artifacts = artifacts_for(tmp_path)
    failure = EvidenceAPIError(
        "OCR_OUTPUT_TRUNCATED", task_id="known-task", usage={"total_tokens": 123},
    )
    failure.args = ("private-provider-body-and-key",)
    failure.completed_request_events = [{
        "provider": "openai-compatible", "model": "ocr-v1", "status": "SUCCESS",
        "usage": {"prompt_tokens": 10, "completion_tokens": 2}, "reused": False,
        "text": "private-recognized-text", "response": "private-provider-body-and-key",
    }]

    def fail() -> dict:
        raise failure

    with pytest.raises(EvidenceBranchError):
        run_evidence_branches({"OCR": fail}, artifacts=artifacts, parallel=False)
    saved = artifacts.load("OCR")
    manifest = read_json(artifacts.directory / "latest.json")["branches"]["OCR"]
    for record in (saved, manifest):
        assert record["errorCode"] == "OCR_OUTPUT_TRUNCATED"
        assert record["taskId"] == "known-task"
        assert record["retryable"] is False
        assert record["usage"] == {"total_tokens": 123}
        assert record["requestEvents"] == [{
            "provider": "openai-compatible", "model": "ocr-v1", "status": "SUCCESS",
            "usage": {"prompt_tokens": 10, "completion_tokens": 2}, "reused": False,
        }]
        assert "private-" not in json.dumps(record)


@pytest.mark.parametrize("api_error", [False, True])
def test_unreported_failure_usage_stays_unknown_and_error_text_is_not_saved(
    tmp_path: Path, api_error: bool,
) -> None:
    artifacts = artifacts_for(tmp_path)
    failure = EvidenceAPIError("API_NETWORK_ERROR") if api_error else RuntimeError("private-api-key")

    def fail() -> dict:
        raise failure

    with pytest.raises(EvidenceBranchError):
        run_evidence_branches({"OCR": fail}, artifacts=artifacts, parallel=False)
    for record in (artifacts.load("OCR"), read_json(artifacts.directory / "latest.json")["branches"]["OCR"]):
        assert record.get("usage", {}) == {}
        assert "estimatedCostUsd" not in record
        assert "totalTokens" not in record
        assert "private-api-key" not in json.dumps(record)


def test_generic_frame_failure_keeps_usage_of_already_completed_requests(tmp_path: Path) -> None:
    artifacts = artifacts_for(tmp_path)
    failure = RuntimeError("private-frame-path")
    failure.completed_request_events = [{"usage": {"total_tokens": 21}, "reused": True}]

    def fail() -> dict:
        raise failure

    with pytest.raises(EvidenceBranchError):
        run_evidence_branches({"OCR": fail}, artifacts=artifacts, parallel=False)
    saved = artifacts.load("OCR")
    assert saved["requestEvents"] == [{"usage": {"total_tokens": 21}, "reused": True}]
    assert "usage" not in saved  # The failing request itself did not report usage.
    assert "errorCode" not in saved


def test_atomic_json_uses_short_independent_temporary_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / ("a" * 64 + ".json")
    original_replace = evidence_pipeline.os.replace
    temporary_paths = []

    def replace(source: Path, target: Path) -> None:
        temporary_paths.append(source)
        assert source.parent == destination.parent
        assert len(source.name) == 21
        assert source.name.startswith(".tmp-")
        assert target == destination
        original_replace(source, target)

    monkeypatch.setattr(evidence_pipeline.os, "replace", replace)
    atomic_json(destination, {"usage": {"total_tokens": 42}})
    assert read_json(destination) == {"usage": {"total_tokens": 42}}
    assert len(temporary_paths) == 1
    assert not temporary_paths[0].exists()
