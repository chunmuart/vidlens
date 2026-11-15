"""OCR pipeline contracts, with no real media tooling or paid API requests."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

# Standalone collection must never create a connection to the developer's DB.
# The main module has already been imported into a test DB in the full suite.
with patch.dict(os.environ, {
    "DATABASE_URL": "sqlite:///:memory:",
    "UPLOAD_ROOT": str(Path(tempfile.gettempdir()) / "vidlens-ocr-tests-import"),
    "JWT_SECRET": "ocr-test-secret", "APP_ENV": "test",
    "ROCKETMQ_NAMESERVER": "", "REDIS_URL": "",
}):
    import vidlens.main as main

from vidlens.evidence_api import EvidenceAPIError
from vidlens.evidence_pipeline import EvidenceBranchError, run_evidence_branches


@pytest.fixture
def video(monkeypatch: pytest.MonkeyPatch):
    # Keep normal Windows paths short without bypassing path handling with \\?\.
    test_root = Path(__file__).resolve().parents[1] / ".test-runtime" / "ocr"
    test_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="o-", dir=test_root) as directory:
        upload_root = Path(directory) / "u"
        upload_root.mkdir()
        monkeypatch.setattr(main, "UPLOAD_ROOT", upload_root)
        settings = {
            "ASR_PROVIDER": "disabled", "LOCAL_ASR_ENABLED": "false",
            "OCR_ENABLED": "true", "OCR_PROVIDER": "dashscope",
            "OCR_BASE_URL": "https://ocr.example.com/v1", "OCR_API_KEY": "test-key",
            "OCR_MODEL": "qwen3.5-ocr", "OCR_INTERVAL_SECONDS": "15",
            "PADDLEOCR_MAX_FRAMES": "4", "PADDLEOCR_FRAME_MAX_WIDTH": "960",
            "PADDLEOCR_DEDUP_THRESHOLD": "1.0", "OCR_MAX_TOKENS": "4096",
        }
        for name, value in settings.items():
            monkeypatch.setenv(name, value)
        path = upload_root / "video.mp4"
        path.write_bytes(b"immutable video fixture")
        yield path


def mock_media_tools(
    monkeypatch: pytest.MonkeyPatch,
    *,
    duration: float,
    pts: list[float],
    frame_count: int | None = None,
) -> list[list[str]]:
    calls: list[list[str]] = []

    def run(command, **kwargs):
        calls.append(command)
        if command[0] == "ffprobe":
            return SimpleNamespace(returncode=0, stdout=f"{duration}\n", stderr="")
        assert command[0] == "ffmpeg", "no external executable may run in this test"
        pattern = command[-1]
        for index in range(len(pts) if frame_count is None else frame_count):
            frame = Path(pattern.replace("%06d", f"{index + 1:06d}"))
            frame.write_bytes(f"deterministic frame {index}".encode())
        stderr = "\n".join(
            f"[Parsed_showinfo_2 @ test] n: {index} pts: 1234 pts_time:{value} fmt:rgb24"
            for index, value in enumerate(pts)
        )
        return SimpleNamespace(returncode=0, stdout="", stderr=stderr)

    monkeypatch.setattr(main.subprocess, "run", run)
    return calls


def ocr_result(text: str) -> dict:
    return {
        "text": text, "status": "SUCCESS" if text else "NO_TEXT",
        "provider": "openai-compatible", "model": "qwen3.5-ocr",
        "usage": {"prompt_tokens": 10, "completion_tokens": 2},
    }


def test_frame_budget_is_spread_across_full_duration_and_actual_pts_are_used(
    video: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands = mock_media_tools(monkeypatch, duration=200, pts=[0.12, 50.04, 100.24, 150.4])
    requested = []

    def recognize(frame, **kwargs):
        requested.append(frame.name)
        return ocr_result("Content " + frame.name)

    monkeypatch.setattr(main, "request_ocr_image", recognize)
    result = main.extract_ocr_result(video)
    ffmpeg = next(command for command in commands if command[0] == "ffmpeg")
    assert "gte(t-prev_selected_t,50)" in ffmpeg[ffmpeg.index("-vf") + 1]
    assert "showinfo" in ffmpeg[ffmpeg.index("-vf") + 1]
    assert ffmpeg[ffmpeg.index("-frames:v") + 1] == "4"
    assert len(requested) == 4
    assert result["samplingIntervalSeconds"] == 50
    assert result["frameCount"] == 4
    assert [item["startMs"] for item in result["segments"]] == [120, 50040, 100240, 150400]
    assert [item["endMs"] for item in result["segments"]] == [50040, 100240, 150400, 200000]


def test_configured_sampling_interval_is_preserved_for_short_video(
    video: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands = mock_media_tools(monkeypatch, duration=20, pts=[0, 15.1])
    monkeypatch.setattr(main, "request_ocr_image", lambda frame, **_: ocr_result(frame.name))
    result = main.extract_ocr_result(video)
    ffmpeg = next(command for command in commands if command[0] == "ffmpeg")
    assert "gte(t-prev_selected_t,15)" in ffmpeg[ffmpeg.index("-vf") + 1]
    assert result["samplingIntervalSeconds"] == 15


def test_completed_cloud_frames_are_reused_after_partial_failure_but_branch_is_not_successful(
    video: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_media_tools(monkeypatch, duration=45, pts=[0.1, 15.04, 30.12])
    calls: list[str] = []
    fail_second_once = [True]

    def recognize(frame, **kwargs):
        calls.append(frame.name)
        if frame.name == "frame-000002.png" and fail_second_once[0]:
            fail_second_once[0] = False
            raise EvidenceAPIError("API_HTTP_503")
        return ocr_result("Content " + frame.name)

    monkeypatch.setattr(main, "request_ocr_image", recognize)
    artifacts = main.evidence_artifacts(video)
    jobs = {"OCR": lambda: main.extract_ocr_result(video)}
    with pytest.raises(EvidenceBranchError) as error:
        run_evidence_branches(jobs, artifacts=artifacts, parallel=False)
    assert error.value.branches == ["OCR"]
    assert artifacts.load("OCR")["status"] == "FAILED"
    assert calls == ["frame-000001.png", "frame-000002.png"]

    segments, manifest = run_evidence_branches(jobs, artifacts=artifacts, parallel=False)
    assert calls == ["frame-000001.png", "frame-000002.png", "frame-000002.png", "frame-000003.png"]
    assert len(segments) == 3
    assert manifest["branches"]["OCR"]["status"] == "SUCCESS"
    events = manifest["branches"]["OCR"]["requestEvents"]
    assert events[0]["reused"] is True
    assert events[0]["usage"] == {"prompt_tokens": 10, "completion_tokens": 2}
    assert all(not item.get("reused", False) for item in events[1:])


def test_model_change_invalidates_cloud_frame_cache(
    video: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_media_tools(monkeypatch, duration=20, pts=[0.1, 15.04])
    models = []

    def recognize(frame, **kwargs):
        models.append(kwargs["model"])
        return {**ocr_result("Content " + frame.name), "model": kwargs["model"]}

    monkeypatch.setattr(main, "request_ocr_image", recognize)
    main.extract_ocr_result(video)
    reused = main.extract_ocr_result(video)
    assert models == ["qwen3.5-ocr", "qwen3.5-ocr"]
    assert all(item["reused"] for item in reused["requestEvents"])
    monkeypatch.setenv("OCR_MODEL", "qwen-vl-ocr")
    changed = main.extract_ocr_result(video)
    assert models == ["qwen3.5-ocr", "qwen3.5-ocr", "qwen-vl-ocr", "qwen-vl-ocr"]
    assert changed["model"] == "qwen-vl-ocr"
    assert all(not item.get("reused", False) for item in changed["requestEvents"])


def test_output_token_limit_change_invalidates_cloud_frame_cache(
    video: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_media_tools(monkeypatch, duration=20, pts=[0.1, 15.04])
    token_limits = []

    def recognize(frame, **kwargs):
        token_limits.append(kwargs["max_tokens"])
        return ocr_result("Content " + frame.name)

    monkeypatch.setattr(main, "request_ocr_image", recognize)
    main.extract_ocr_result(video)
    main.extract_ocr_result(video)
    assert token_limits == [4096, 4096]
    monkeypatch.setenv("OCR_MAX_TOKENS", "8192")
    changed = main.extract_ocr_result(video)
    assert token_limits == [4096, 4096, 8192, 8192]
    assert all(not item.get("reused", False) for item in changed["requestEvents"])


def test_partial_cloud_failure_retains_completed_and_failed_request_usage(
    video: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_media_tools(monkeypatch, duration=20, pts=[0.1, 15.04])
    failed_usage = {"prompt_tokens": 20, "completion_tokens": 4096}

    def recognize(frame, **kwargs):
        if frame.name == "frame-000002.png":
            raise EvidenceAPIError("OCR_TRUNCATED_RESPONSE", usage=failed_usage)
        return ocr_result("Content " + frame.name)

    monkeypatch.setattr(main, "request_ocr_image", recognize)
    with pytest.raises(EvidenceAPIError) as error:
        main.extract_ocr_result(video)
    assert error.value.usage == failed_usage
    completed = error.value.completed_request_events
    assert len(completed) == 1
    assert completed[0]["usage"] == {"prompt_tokens": 10, "completion_tokens": 2}
    assert completed[0]["status"] == "SUCCESS"


def test_blank_frames_are_no_text_and_successfully_cached(
    video: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_media_tools(monkeypatch, duration=20, pts=[0, 15])
    calls = []

    def recognize(frame, **kwargs):
        calls.append(frame.name)
        return ocr_result("")

    monkeypatch.setattr(main, "request_ocr_image", recognize)
    first = main.extract_ocr_result(video)
    second = main.extract_ocr_result(video)
    assert first["status"] == "NO_TEXT"
    assert first["segments"] == []
    assert second["status"] == "NO_TEXT"
    assert calls == ["frame-000001.png", "frame-000002.png"]
    assert all(item["reused"] for item in second["requestEvents"])


def test_disabled_ocr_is_skipped_before_probe_or_api_call(
    video: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OCR_ENABLED", "false")

    def must_not_run(*args, **kwargs):
        raise AssertionError("disabled OCR must perform no work")

    monkeypatch.setattr(main.subprocess, "run", must_not_run)
    monkeypatch.setattr(main, "request_ocr_image", must_not_run)
    assert main.extract_ocr_result(video) == {"segments": [], "status": "SKIPPED", "provider": "disabled"}


def test_missing_frame_pts_fails_before_api_calls(
    video: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_media_tools(monkeypatch, duration=45, pts=[0.1, 15.04], frame_count=3)

    def must_not_run(*args, **kwargs):
        raise AssertionError("must not invent OCR timing")

    monkeypatch.setattr(main, "request_ocr_image", must_not_run)
    with pytest.raises(ValueError, match="真实时间戳"):
        main.extract_ocr_result(video)


def test_empty_frame_output_is_no_text_without_api_calls(
    video: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_media_tools(monkeypatch, duration=45, pts=[])

    def must_not_run(*args, **kwargs):
        raise AssertionError("no frames to recognize")

    monkeypatch.setattr(main, "request_ocr_image", must_not_run)
    result = main.extract_ocr_result(video)
    assert result["status"] == "NO_TEXT"
    assert result["frameCount"] == 0
    assert result["segments"] == []


def test_local_partial_frame_errors_do_not_become_complete_evidence(
    video: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OCR_PROVIDER", "local")
    monkeypatch.setenv("LOCAL_ASR_RELEASE_BEFORE_OCR", "false")
    mock_media_tools(monkeypatch, duration=20, pts=[0, 15])
    monkeypatch.setattr(main, "run_paddle_ocr_frames", lambda _: {
        "frameCount": 2, "results": [{"index": 0, "content": "partial text"}],
        "errors": [{"frame": "frame-000002.png", "error": "failed"}],
    })
    with pytest.raises(RuntimeError, match="仅完成部分画面"):
        main.extract_ocr_result(video)
