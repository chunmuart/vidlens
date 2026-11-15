"""Recover the cache/DB commit window without using media tools or paid APIs."""
from __future__ import annotations

import importlib
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from vidlens.evidence_pipeline import read_json


@pytest.fixture
def runtime(monkeypatch: pytest.MonkeyPatch):
    # Use our own engine even when test_flow has already imported main. This
    # also isolates this file from simultaneous test processes on Windows.
    test_root = Path(__file__).resolve().parents[2] / ".test-runtime" / "cache"
    test_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="c-", dir=test_root) as directory:
        root = Path(directory)
        database_url = "sqlite:///" + (root / "test.db").as_posix()
        settings = {
            "VIDLENS_TEST_ROOT": str(root), "DATABASE_URL": database_url,
            "UPLOAD_ROOT": str(root / "u"), "JWT_SECRET": "cache-test-secret",
            "APP_ENV": "test", "ROCKETMQ_NAMESERVER": "", "REDIS_URL": "",
            "ASR_PROVIDER": "openai", "ASR_BASE_URL": "https://asr.example/v1",
            "ASR_MODEL": "new-asr-model", "ASR_API_KEY": "test-key",
            "LOCAL_ASR_ENABLED": "false", "ASR_LOCAL_FALLBACK_ENABLED": "false",
            "OCR_ENABLED": "true", "OCR_PROVIDER": "dashscope",
            "OCR_MODEL": "new-ocr-model", "EVIDENCE_PARALLEL_ENABLED": "true",
        }
        for name, value in settings.items():
            monkeypatch.setenv(name, value)
        main = importlib.import_module("vidlens.main")
        engine = create_engine(database_url, connect_args={"check_same_thread": False})
        sessions = sessionmaker(bind=engine, expire_on_commit=False, class_=Session)
        monkeypatch.setattr(main, "engine", engine)
        monkeypatch.setattr(main, "SessionLocal", sessions)
        monkeypatch.setattr(main, "UPLOAD_ROOT", root / "u")
        monkeypatch.setattr(main, "build_lock_heartbeat", lambda task: None)
        monkeypatch.setattr(main, "release_active_lock", lambda *args: None)
        main.Base.metadata.create_all(engine)
        main.UPLOAD_ROOT.mkdir()
        video = main.UPLOAD_ROOT / "video.mp4"
        video.write_bytes(b"immutable-video-content")
        try:
            yield SimpleNamespace(main=main, video=video)
        finally:
            engine.dispose()


def segment(content: str) -> dict:
    return {"source": "ASR", "startMs": 1000, "endMs": 2000, "content": content}


def completed_artifacts(runtime):
    artifacts = runtime.main.evidence_artifacts(runtime.video)
    artifacts.save("ASR", {"status": "SUCCESS", "segments": [segment("new model evidence")]})
    artifacts.save("OCR", {"status": "NO_TEXT", "segments": []})
    return artifacts


def create_media(runtime) -> int:
    main = runtime.main
    with main.SessionLocal() as db:
        user = main.User(username="cache-user", password_hash="test-hash")
        db.add(user)
        db.flush()
        media = main.Media(
            user_id=user.id, filename=runtime.video.name, file_path=str(runtime.video),
            content_hash="a" * 64, transcript_text="old model evidence", status="PROCESSING",
        )
        db.add(media)
        db.commit()
        return media.id


def test_completed_artifacts_do_not_match_old_database_content(runtime) -> None:
    artifacts = completed_artifacts(runtime)
    old_database = [{**segment("old model evidence"), "segmentId": 7}]
    assert runtime.main.evidence_cache_matches(old_database, artifacts) is False
    # Database IDs are local metadata and do not invalidate equal evidence.
    new_database = [{**segment("new model evidence"), "segmentId": 8}]
    assert runtime.main.evidence_cache_matches(new_database, artifacts) is True


def test_cache_written_before_database_commit_rebuilds_rows_without_api_calls(
    runtime, monkeypatch: pytest.MonkeyPatch,
) -> None:
    main = runtime.main
    media_id = create_media(runtime)
    with main.SessionLocal() as db:
        main.replace_evidence(db, media_id, [segment("old model evidence")])
        db.commit()
    # Simulate extraction finishing and saving both artifacts, followed by a
    # crash before the replacement database transaction was committed.
    artifacts = completed_artifacts(runtime)

    def must_not_request(*args, **kwargs):
        raise AssertionError("completed evidence must be recovered without another API request")

    for name in ("request_openai_asr", "request_dashscope_asr", "request_local_asr",
                 "extract_audio_file", "extract_ocr_result"):
        monkeypatch.setattr(main, name, must_not_request)
    with main.SessionLocal() as db:
        media = db.get(main.Media, media_id)
        restored = main.ensure_media_evidence(db, media)
        assert [item["content"] for item in restored] == ["new model evidence"]
        assert media.transcript_text == "new model evidence"
        db.commit()
    with main.SessionLocal() as db:
        saved = main.stored_evidence(db, media_id)
        assert [item["content"] for item in saved] == ["new model evidence"]
        assert main.evidence_cache_matches(saved, main.evidence_artifacts(runtime.video)) is True
    manifest = read_json(artifacts.directory / "latest.json")
    assert all(branch["reused"] for branch in manifest["branches"].values())


def test_whitespace_uppercase_local_ocr_and_local_asr_run_serially(
    runtime, monkeypatch: pytest.MonkeyPatch,
) -> None:
    main = runtime.main
    monkeypatch.setenv("ASR_PROVIDER", "local")
    monkeypatch.setenv("LOCAL_ASR_ENABLED", "true")
    monkeypatch.setenv("OCR_PROVIDER", " LOCAL ")
    calls = []

    def local_asr(*args, **kwargs):
        calls.append("ASR")
        return [segment("local speech")]

    def local_ocr(*args, **kwargs):
        calls.append("OCR")
        return {"status": "NO_TEXT", "segments": []}

    monkeypatch.setattr(main, "request_local_asr", local_asr)
    monkeypatch.setattr(main, "extract_ocr_result", local_ocr)
    assert main.collect_evidence(runtime.video)[0]["content"] == "local speech"
    manifest = main.evidence_manifest(runtime.video)
    assert manifest["parallel"] is False
    assert calls == ["ASR", "OCR"]


@pytest.mark.parametrize("max_attempts, expected", [(1, "FAILED"), (2, "RETRYING")])
def test_invalid_asr_provider_persists_failure_state_instead_of_processing(
    runtime, monkeypatch: pytest.MonkeyPatch, max_attempts: int, expected: str,
) -> None:
    main = runtime.main
    media_id = create_media(runtime)
    task_id = "invalid-asr-provider"
    with main.SessionLocal() as db:
        media = db.get(main.Media, media_id)
        db.add(main.AnalysisTask(
            id=task_id, media_id=media_id, user_id=media.user_id, goal="summarize",
            goal_hash="b" * 64, state="QUEUED", max_attempts=max_attempts,
        ))
        db.commit()
    monkeypatch.setenv("ASR_PROVIDER", "invalid-provider")

    assert main.process_analysis(task_id) == expected
    with main.SessionLocal() as db:
        task = db.get(main.AnalysisTask, task_id)
        assert task.state == expected
        assert task.stage == expected
        assert task.attempt_count == 1
        assert "ASR_PROVIDER" in task.error
        trace = json.loads(task.trace_json)
        assert trace["lastErrorType"] == "ValueError"
        assert trace["evidenceExtraction"] == {}
