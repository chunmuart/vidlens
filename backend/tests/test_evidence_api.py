from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from vidlens.evidence_api import (
    ASRTimestampError,
    EvidenceAPIError,
    request_dashscope_asr,
    request_ocr_image,
    request_openai_asr,
)


@pytest.fixture
def audio(tmp_path: Path) -> Path:
    path = tmp_path / "audio.mp3"
    path.write_bytes(b"test audio")
    return path


@pytest.fixture
def image(tmp_path: Path) -> Path:
    path = tmp_path / "frame.png"
    path.write_bytes(b"test image")
    return path


def response(data: object, *, status: int = 200) -> httpx.Response:
    return httpx.Response(status, json=data)


def test_dashscope_upload_submit_poll_and_parse_timestamped_sentences(audio: Path) -> None:
    saved = []
    seen = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path == "/api/v1/uploads":
            assert dict(request.url.params) == {"action": "getPolicy", "model": "fun-asr"}
            assert request.headers["authorization"] == "Bearer test-key"
            return response({"data": {
                "upload_host": "https://uploads.example.com/",
                "upload_dir": "temporary/test", "oss_access_key_id": "temporary-access",
                "policy": "temporary-policy", "signature": "temporary-signature",
                "x_oss_object_acl": "private", "x_oss_forbid_overwrite": "true",
            }})
        if request.url.host == "uploads.example.com":
            assert "authorization" not in request.headers
            assert b'temporary-policy' in request.content
            assert b'name="file"; filename="audio.mp3"' in request.content
            return httpx.Response(200)
        if request.url.path.endswith("/transcription"):
            body = json.loads(request.content)
            assert body["model"] == "fun-asr"
            assert body["input"]["file_urls"][0].startswith("oss://temporary/test/")
            assert request.headers["X-DashScope-Async"] == "enable"
            assert request.headers["X-DashScope-OssResourceResolve"] == "enable"
            return response({"output": {"task_id": "task-123", "task_status": "PENDING"}})
        if request.url.path.endswith("/task-123"):
            assert saved == ["task-123"], "persist ID before polling"
            return response({
                "output": {"task_status": "SUCCEEDED", "results": [{
                    "subtask_status": "SUCCEEDED", "transcription_url": "https://results.example.com/result.json",
                }]}, "usage": {"duration": 6.5},
            })
        if request.url.host == "results.example.com":
            assert "authorization" not in request.headers
            return response({"transcripts": [{"sentences": [
                {"begin_time": 1250, "end_time": 2600, "text": "第一段文字。"},
                {"begin_time": 3000, "end_time": 4000, "text": "第二段文字。"},
            ]}]})
        raise AssertionError(request.url)

    with httpx.Client(
        transport=httpx.MockTransport(handle),
        headers={"Authorization": "Bearer unrelated-default"},
        auth=httpx.BasicAuth("unrelated-user", "unrelated-secret"),
    ) as client:
        result = request_dashscope_asr(audio, api_key="test-key", client=client, on_submitted=saved.append)
    assert len(seen) == 5
    assert result == {
        "segments": [
            {"source": "ASR", "startMs": 1250, "endMs": 2600, "content": "第一段文字。"},
            {"source": "ASR", "startMs": 3000, "endMs": 4000, "content": "第二段文字。"},
        ],
        "status": "SUCCESS", "provider": "dashscope", "model": "fun-asr",
        "usage": {"duration": 6.5}, "taskId": "task-123",
    }


def test_dashscope_resume_does_not_upload_or_resubmit(audio: Path) -> None:
    calls = []

    def handle(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        assert request.method == "GET"
        if request.url.host == "results.example.com":
            return response({"transcripts": [{"text": "", "sentences": []}]})
        return response({"output": {"task_status": "SUCCEEDED", "results": [{
            "transcription_url": "https://results.example.com/empty.json",
        }]}})

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        result = request_dashscope_asr(audio, api_key="test-key", client=client, resume_task_id="existing-task")
    assert len(calls) == 2
    assert result["status"] == "NO_TEXT"
    assert result["taskId"] == "existing-task"


def test_dashscope_running_timeout_exposes_resume_id_without_resubmission(audio: Path) -> None:
    elapsed = [0.0]
    calls = []

    def sleep(seconds: float) -> None:
        elapsed[0] += seconds

    def handle(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        assert request.method == "GET"
        return response({"output": {"task_status": "RUNNING"}})

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(EvidenceAPIError, match="ASR_POLL_TIMEOUT") as error:
            request_dashscope_asr(
                audio, api_key="test-key", client=client, resume_task_id="existing-task",
                poll_timeout=3, poll_interval=2, clock=lambda: elapsed[0], sleep=sleep,
            )
    assert error.value.task_id == "existing-task"
    assert error.value.retryable
    assert len(calls) == 2
    assert calls[-1].extensions["timeout"]["read"] == 1


@pytest.mark.parametrize("status,code", [
    ("FAILED", "ASR_TASK_FAILED"), ("CANCELED", "ASR_TASK_CANCELED"),
    ("SOMETHING_NEW", "ASR_TASK_STATUS_INVALID"),
])
def test_dashscope_terminal_failure_is_explicit(audio: Path, status: str, code: str) -> None:
    with httpx.Client(transport=httpx.MockTransport(lambda _: response({
        "output": {"task_status": status, "message": "sensitive provider text"},
        "usage": {"duration": 8},
    }))) as client:
        with pytest.raises(EvidenceAPIError, match=code) as error:
            request_dashscope_asr(audio, api_key="test-key", resume_task_id="task-1", client=client)
    assert error.value.task_id == "task-1"
    assert error.value.usage == {"duration": 8}
    assert "sensitive" not in str(error.value)


def test_dashscope_rejects_transcript_without_sentence_timestamps(audio: Path) -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.host == "results.example.com":
            return response({"transcripts": [{"text": "文字", "sentences": [{"text": "文字"}]}]})
        return response({"output": {"task_status": "SUCCEEDED", "results": [{
            "transcription_url": "https://results.example.com/text.json",
        }]}})

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(ASRTimestampError) as error:
            request_dashscope_asr(audio, api_key="test-key", resume_task_id="task-1", client=client)
    assert error.value.task_id == "task-1"


def test_openai_asr_requests_verbose_segments_and_preserves_usage(audio: Path) -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        assert request.url == "https://api.example.com/v1/audio/transcriptions"
        assert request.method == "POST"
        assert b'timestamp_granularities[]' in request.content
        assert b'verbose_json' in request.content
        return response({"text": "你好", "segments": [{"start": 1.25, "end": 2.5, "text": "你好"}], "usage": {"seconds": 3}})

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        result = request_openai_asr(audio, base_url="https://api.example.com/v1", api_key="key", client=client)
    assert result["segments"] == [{"source": "ASR", "startMs": 1250, "endMs": 2500, "content": "你好"}]
    assert result["usage"] == {"seconds": 3}


@pytest.mark.parametrize("payload", [
    {"text": "只有整段文字"},
    {"text": "只有整段文字", "segments": []},
    {"segments": [{"text": "没有时间"}]},
    {"segments": [{"start": 0, "end": 0, "text": "无效时间"}]},
    {"segments": [{"start": -1, "end": 1, "text": "负数"}]},
    {"segments": [{"start": "nan", "end": 1, "text": "非数"}]},
])
def test_openai_asr_rejects_missing_or_invalid_timestamps(audio: Path, payload: dict) -> None:
    with httpx.Client(transport=httpx.MockTransport(lambda _: response(payload))) as client:
        with pytest.raises(ASRTimestampError):
            request_openai_asr(audio, base_url="https://api.example.com/v1", api_key="key", client=client)


def test_openai_asr_empty_audio_is_no_text(audio: Path) -> None:
    with httpx.Client(transport=httpx.MockTransport(lambda _: response({"text": "", "segments": []}))) as client:
        result = request_openai_asr(audio, base_url="https://api.example.com/v1", api_key="key", client=client)
    assert result["status"] == "NO_TEXT"
    assert result["segments"] == []


def test_ocr_uses_inline_image_and_preserves_text_usage(image: Path) -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert request.url == "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
        assert body["model"] == "qwen3.5-ocr"
        assert body["messages"][0]["content"][0]["image_url"]["url"].startswith("data:image/png;base64,")
        return response({"choices": [{"finish_reason": "stop", "message": {"content": " 第一行\n第二行 "}}], "usage": {"prompt_tokens": 10, "completion_tokens": 5}})

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        result = request_ocr_image(image, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", api_key="key", client=client)
    assert result["text"] == "第一行\n第二行"
    assert result["usage"] == {"prompt_tokens": 10, "completion_tokens": 5}
    assert result["status"] == "SUCCESS"


def test_ocr_empty_text_is_distinct_from_error(image: Path) -> None:
    with httpx.Client(transport=httpx.MockTransport(lambda _: response({
        "choices": [{"finish_reason": "stop", "message": {"content": ""}}],
    }))) as client:
        result = request_ocr_image(image, base_url="https://api.example.com/v1", api_key="key", client=client)
    assert result["status"] == "NO_TEXT"


def test_ocr_truncation_is_not_saved_as_complete(image: Path) -> None:
    with httpx.Client(transport=httpx.MockTransport(lambda _: response({
        "choices": [{"finish_reason": "length", "message": {"content": "一半的文字"}}], "usage": {"total_tokens": 4096},
    }))) as client:
        with pytest.raises(EvidenceAPIError, match="OCR_OUTPUT_TRUNCATED") as error:
            request_ocr_image(image, base_url="https://api.example.com/v1", api_key="key", client=client)
    assert error.value.usage == {"total_tokens": 4096}


def test_billable_post_is_not_automatically_retried_and_error_is_redacted(image: Path) -> None:
    seen = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(503, text="provider details including secret-key")

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(EvidenceAPIError, match="API_HTTP_503") as error:
            request_ocr_image(image, base_url="https://api.example.com/v1", api_key="secret-key", client=client)
    assert len(seen) == 1
    assert "secret-key" not in str(error.value)


def test_missing_key_is_rejected_before_requests(image: Path) -> None:
    with pytest.raises(EvidenceAPIError, match="API_KEY_REQUIRED"):
        request_ocr_image(image, base_url="https://api.example.com/v1", api_key="")


@pytest.mark.parametrize("failure_kind", ["network", "http503", "invalid_json"])
@pytest.mark.parametrize("at_result_download", [False, True])
def test_known_task_transient_error_keeps_task_id_for_resume(
    audio: Path, failure_kind: str, at_result_download: bool,
) -> None:
    calls = []

    def handle(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        assert request.method == "GET"
        if at_result_download and request.url.host != "results.example.com":
            return response({"output": {"task_status": "SUCCEEDED", "results": [{
                "transcription_url": "https://results.example.com/result.json",
            }]}})
        if failure_kind == "network":
            raise httpx.ReadTimeout("sensitive network detail", request=request)
        if failure_kind == "http503":
            return httpx.Response(503, text="sensitive provider detail")
        return httpx.Response(200, text="invalid JSON sensitive content")

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(EvidenceAPIError) as error:
            request_dashscope_asr(audio, api_key="secret-key", resume_task_id="known-task", client=client)
    assert error.value.task_id == "known-task"
    assert error.value.retryable
    assert "sensitive" not in str(error.value)
    assert len(calls) == (2 if at_result_download else 1)


def test_dashscope_invalid_resume_id_does_not_issue_request(audio: Path) -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        raise AssertionError("must not request invalid task path")

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(EvidenceAPIError, match="ASR_TASK_ID_INVALID"):
            request_dashscope_asr(audio, api_key="key", resume_task_id="../other-endpoint", client=client)


@pytest.mark.parametrize("submitted_output", [[], "unexpected", None])
def test_dashscope_malformed_submit_response_is_explicit(audio: Path, submitted_output: object) -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/uploads":
            return response({"data": {
                "upload_host": "https://uploads.example.com/", "upload_dir": "temporary/test",
                "oss_access_key_id": "a", "policy": "p", "signature": "s",
                "x_oss_object_acl": "private", "x_oss_forbid_overwrite": "true",
            }})
        if request.url.host == "uploads.example.com":
            return httpx.Response(200)
        assert request.url.path.endswith("/transcription")
        return response({"output": submitted_output})

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(EvidenceAPIError, match="ASR_TASK_RESPONSE_INVALID") as error:
            request_dashscope_asr(audio, api_key="key", client=client)
    assert not error.value.retryable, "unknown submission outcome must not be blindly re-posted"


def test_ocr_explicit_model_remains_configurable(image: Path) -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["model"] == "qwen-vl-ocr"
        return response({"choices": [{"finish_reason": "stop", "message": {"content": ""}}]})

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        result = request_ocr_image(image, base_url="https://api.example.com/v1", api_key="key", model="qwen-vl-ocr", client=client)
    assert result["model"] == "qwen-vl-ocr"
