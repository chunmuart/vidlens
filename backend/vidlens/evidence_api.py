"""Small synchronous ASR/OCR adapters, with no provider SDK dependency.

Only ASR responses with real segment timestamps become timeline evidence.
DashScope task IDs can be persisted by the caller and resumed after a timeout.
GET requests are safe to retry; billable POST requests are never retried here.
"""

from __future__ import annotations

import base64
import math
import mimetypes
import re
import time
import uuid
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

import httpx


class EvidenceAPIError(RuntimeError):
    """A redacted error safe for task records; never includes keys or bodies."""

    def __init__(
        self,
        code: str,
        *,
        task_id: str | None = None,
        retryable: bool = False,
        usage: dict[str, Any] | None = None,
    ) -> None:
        self.code = code
        self.task_id = task_id
        self.retryable = retryable
        self.usage = dict(usage or {})
        super().__init__(code)


class ASRTimestampError(EvidenceAPIError):
    """Text without valid timing cannot be presented as timestamped evidence."""


def _usage(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _check_key(api_key: str) -> None:
    if not str(api_key or "").strip():
        raise EvidenceAPIError("API_KEY_REQUIRED")


def _request(
    client: httpx.Client,
    method: str,
    url: str,
    *,
    timeout: float,
    task_id: str | None = None,
    retries: int = 0,
    external: bool = False,
    sleep: Callable[[float], None] = time.sleep,
    **kwargs: Any,
) -> httpx.Response:
    """Retry GET only, and keep provider response bodies out of exceptions."""
    attempts = max(0, retries) + 1 if method == "GET" else 1
    for attempt in range(attempts):
        try:
            request = client.build_request(method, url, timeout=max(0.01, timeout), **kwargs)
            if external:
                # OSS uploads/result downloads must not inherit a client's API token.
                request.headers.pop("authorization", None)
            # Explicit request headers carry authentication. Do not let an
            # injected client's default auth re-add a token on OSS/result hosts.
            response = client.send(request, auth=None)
        except httpx.HTTPError:
            if attempt + 1 < attempts:
                sleep(min(0.25 * (attempt + 1), 1.0))
                continue
            raise EvidenceAPIError(
                "API_NETWORK_ERROR", task_id=task_id,
                retryable=method == "GET" or task_id is not None,
            ) from None
        if response.is_success:
            return response
        transient = response.status_code == 429 or response.status_code >= 500
        if transient and attempt + 1 < attempts:
            sleep(min(0.25 * (attempt + 1), 1.0))
            continue
        raise EvidenceAPIError(
            f"API_HTTP_{response.status_code}", task_id=task_id,
            retryable=transient and (method == "GET" or task_id is not None),
        )
    raise AssertionError("unreachable")


def _json(response: httpx.Response, *, task_id: str | None = None) -> dict[str, Any]:
    try:
        payload = response.json()
    except (ValueError, UnicodeDecodeError):
        raise EvidenceAPIError("API_INVALID_JSON", task_id=task_id, retryable=task_id is not None) from None
    if not isinstance(payload, dict):
        raise EvidenceAPIError("API_INVALID_RESPONSE", task_id=task_id, retryable=task_id is not None)
    return payload


def _milliseconds(value: Any, *, seconds: bool = False) -> int:
    if isinstance(value, bool) or value is None:
        raise ASRTimestampError("ASR_TIMESTAMP_REQUIRED")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ASRTimestampError("ASR_TIMESTAMP_REQUIRED") from None
    if not math.isfinite(number) or number < 0:
        raise ASRTimestampError("ASR_TIMESTAMP_INVALID")
    return round(number * 1000) if seconds else round(number)


def _segment(text: Any, start: Any, end: Any, *, seconds: bool = False) -> dict[str, Any] | None:
    content = re.sub(r"\s+", " ", str(text or "")).strip()
    if not content:
        return None
    start_ms = _milliseconds(start, seconds=seconds)
    end_ms = _milliseconds(end, seconds=seconds)
    if end_ms <= start_ms:
        raise ASRTimestampError("ASR_TIMESTAMP_INVALID")
    return {"source": "ASR", "startMs": start_ms, "endMs": end_ms, "content": content}


def _result(
    *, segments: list[dict[str, Any]], provider: str, model: str,
    usage: Any = None, task_id: str | None = None,
) -> dict[str, Any]:
    return {
        "segments": segments,
        "status": "SUCCESS" if segments else "NO_TEXT",
        "provider": provider,
        "model": model,
        "usage": _usage(usage),
        "taskId": task_id,
    }


def request_openai_asr(
    audio_path: Path,
    *,
    base_url: str,
    api_key: str,
    model: str = "whisper-1",
    timeout: float = 180,
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    """OpenAI-compatible multipart transcription with segment timestamps."""
    _check_key(api_key)
    with nullcontext(client) if client is not None else httpx.Client() as http:
        with Path(audio_path).open("rb") as audio:
            response = _request(
                http, "POST", base_url.rstrip("/") + "/audio/transcriptions",
                headers={"Authorization": f"Bearer {api_key}"},
                data={
                    "model": model, "response_format": "verbose_json",
                    "timestamp_granularities[]": "segment",
                },
                files={"file": (Path(audio_path).name, audio, mimetypes.guess_type(str(audio_path))[0] or "audio/mpeg")},
                timeout=timeout,
            )
        payload = _json(response)
    raw = payload.get("segments")
    if raw is not None and not isinstance(raw, list):
        raise ASRTimestampError("ASR_TIMESTAMP_REQUIRED", usage=_usage(payload.get("usage")))
    segments: list[dict[str, Any]] = []
    for item in raw or []:
        if not isinstance(item, dict):
            raise ASRTimestampError("ASR_TIMESTAMP_REQUIRED", usage=_usage(payload.get("usage")))
        try:
            segment = _segment(item.get("text", item.get("content")), item.get("start"), item.get("end"), seconds=True)
        except ASRTimestampError as exc:
            exc.usage = _usage(payload.get("usage"))
            raise
        if segment:
            segments.append(segment)
    if not segments and str(payload.get("text") or "").strip():
        raise ASRTimestampError("ASR_TIMESTAMP_REQUIRED", usage=_usage(payload.get("usage")))
    return _result(segments=segments, provider="openai", model=model, usage=payload.get("usage"))


def _upload_dashscope_audio(
    http: httpx.Client,
    audio_path: Path,
    *,
    base_url: str,
    api_key: str,
    model: str,
    timeout: float,
    sleep: Callable[[float], None],
) -> str:
    policy = _json(_request(
        http, "GET", base_url.rstrip("/") + "/api/v1/uploads",
        params={"action": "getPolicy", "model": model},
        headers={"Authorization": f"Bearer {api_key}"}, timeout=timeout,
        retries=2, sleep=sleep,
    )).get("data")
    required = ("upload_host", "upload_dir", "oss_access_key_id", "policy", "signature", "x_oss_object_acl", "x_oss_forbid_overwrite")
    if not isinstance(policy, dict) or any(not str(policy.get(key) or "") for key in required):
        raise EvidenceAPIError("ASR_UPLOAD_POLICY_INVALID")
    upload_url = str(policy["upload_host"])
    if urlparse(upload_url).scheme != "https":
        raise EvidenceAPIError("ASR_UPLOAD_URL_INVALID")
    object_key = str(policy["upload_dir"]).rstrip("/") + "/" + uuid.uuid4().hex + "-" + audio_path.name
    data = {
        "OSSAccessKeyId": str(policy["oss_access_key_id"]),
        "Signature": str(policy["signature"]),
        "policy": str(policy["policy"]),
        "x-oss-object-acl": str(policy["x_oss_object_acl"]),
        "x-oss-forbid-overwrite": str(policy["x_oss_forbid_overwrite"]),
        "key": object_key,
        "success_action_status": "200",
    }
    with audio_path.open("rb") as audio:
        _request(
            http, "POST", upload_url, timeout=timeout, external=True,
            data=data, files={"file": (audio_path.name, audio, "audio/mpeg")},
        )
    return "oss://" + object_key


def request_dashscope_asr(
    audio_path: Path,
    *,
    base_url: str = "https://dashscope.aliyuncs.com",
    api_key: str,
    model: str = "fun-asr",
    request_timeout: float = 60,
    poll_timeout: float = 600,
    poll_interval: float = 2,
    resume_task_id: str | None = None,
    on_submitted: Callable[[str], None] | None = None,
    client: httpx.Client | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Upload, submit once and poll a Fun-ASR task, or resume its saved ID.

    ``on_submitted`` must durably save the ID before polling. A timeout/error
    carries ``task_id`` so a retry can continue polling rather than submit again.
    """
    _check_key(api_key)
    task_id = str(resume_task_id or "").strip() or None
    headers = {"Authorization": f"Bearer {api_key}"}
    with nullcontext(client) if client is not None else httpx.Client() as http:
        if task_id is None:
            file_url = _upload_dashscope_audio(
                http, Path(audio_path), base_url=base_url, api_key=api_key,
                model=model, timeout=request_timeout, sleep=sleep,
            )
            submitted = _json(_request(
                http, "POST", base_url.rstrip("/") + "/api/v1/services/audio/asr/transcription",
                headers={**headers, "X-DashScope-Async": "enable", "X-DashScope-OssResourceResolve": "enable"},
                json={"model": model, "input": {"file_urls": [file_url]}, "parameters": {}},
                timeout=request_timeout,
            ))
            submitted_output = submitted.get("output")
            if not isinstance(submitted_output, dict):
                raise EvidenceAPIError("ASR_TASK_RESPONSE_INVALID", usage=_usage(submitted.get("usage")))
            task_id = str(submitted_output.get("task_id") or "").strip()
            if not task_id:
                raise EvidenceAPIError("ASR_TASK_ID_MISSING", usage=_usage(submitted.get("usage")))
            if on_submitted is not None:
                try:
                    on_submitted(task_id)
                except Exception:
                    raise EvidenceAPIError("ASR_TASK_SAVE_FAILED", task_id=task_id, retryable=True) from None

        # Provider task IDs are path segments; forbid accidentally treating URLs
        # or supplied path traversal as another API endpoint.
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,200}", task_id):
            raise EvidenceAPIError("ASR_TASK_ID_INVALID")
        deadline = clock() + max(0.01, poll_timeout)
        usage: dict[str, Any] = {}
        while True:
            remaining = deadline - clock()
            if remaining <= 0:
                raise EvidenceAPIError("ASR_POLL_TIMEOUT", task_id=task_id, retryable=True, usage=usage)
            payload = _json(_request(
                http, "GET", base_url.rstrip("/") + "/api/v1/tasks/" + task_id,
                headers=headers, timeout=min(request_timeout, remaining),
                task_id=task_id, sleep=sleep,
                # A failed poll is resumable. The loop's total deadline takes
                # precedence over hidden request-level retries.
                retries=0,
            ), task_id=task_id)
            usage = _usage(payload.get("usage")) or usage
            output = payload.get("output")
            if not isinstance(output, dict):
                raise EvidenceAPIError("ASR_TASK_RESPONSE_INVALID", task_id=task_id, usage=usage)
            state = str(output.get("task_status") or "").upper()
            if state in {"PENDING", "RUNNING", "UNKNOWN"}:
                sleep(min(max(0.01, poll_interval), max(0.0, deadline - clock())))
                continue
            if state in {"FAILED", "CANCELED", "CANCELLED"}:
                raise EvidenceAPIError("ASR_TASK_" + state, task_id=task_id, usage=usage)
            if state != "SUCCEEDED":
                raise EvidenceAPIError("ASR_TASK_STATUS_INVALID", task_id=task_id, usage=usage)
            results = output.get("results")
            if not isinstance(results, list) or len(results) != 1 or not isinstance(results[0], dict):
                raise EvidenceAPIError("ASR_RESULT_MISSING", task_id=task_id, usage=usage)
            result = results[0]
            if str(result.get("subtask_status") or "SUCCEEDED").upper() != "SUCCEEDED":
                raise EvidenceAPIError("ASR_SUBTASK_FAILED", task_id=task_id, usage=usage)
            result_url = str(result.get("transcription_url") or "")
            if urlparse(result_url).scheme not in {"https", "http"}:
                raise EvidenceAPIError("ASR_RESULT_URL_INVALID", task_id=task_id, usage=usage)
            remaining = deadline - clock()
            if remaining <= 0:
                raise EvidenceAPIError("ASR_POLL_TIMEOUT", task_id=task_id, retryable=True, usage=usage)
            transcript = _json(_request(
                http, "GET", result_url, timeout=min(request_timeout, remaining),
                task_id=task_id, external=True,
            ), task_id=task_id)
            transcripts = transcript.get("transcripts")
            if not isinstance(transcripts, list):
                raise ASRTimestampError("ASR_TIMESTAMP_REQUIRED", task_id=task_id, usage=usage)
            segments: list[dict[str, Any]] = []
            for channel in transcripts:
                if not isinstance(channel, dict) or not isinstance(channel.get("sentences"), list):
                    raise ASRTimestampError("ASR_TIMESTAMP_REQUIRED", task_id=task_id, usage=usage)
                channel_segments: list[dict[str, Any]] = []
                for sentence in channel["sentences"]:
                    if not isinstance(sentence, dict):
                        raise ASRTimestampError("ASR_TIMESTAMP_REQUIRED", task_id=task_id, usage=usage)
                    try:
                        segment = _segment(sentence.get("text"), sentence.get("begin_time"), sentence.get("end_time"))
                    except ASRTimestampError as exc:
                        exc.task_id = task_id
                        exc.usage = usage
                        raise
                    if segment:
                        channel_segments.append(segment)
                if not channel_segments and str(channel.get("text") or "").strip():
                    raise ASRTimestampError("ASR_TIMESTAMP_REQUIRED", task_id=task_id, usage=usage)
                segments.extend(channel_segments)
            if not segments and str(transcript.get("text") or "").strip():
                raise ASRTimestampError("ASR_TIMESTAMP_REQUIRED", task_id=task_id, usage=usage)
            return _result(
                segments=sorted(segments, key=lambda item: (item["startMs"], item["endMs"])),
                provider="dashscope", model=model, usage=usage, task_id=task_id,
            )


def request_ocr_image(
    image_path: Path,
    *,
    base_url: str,
    api_key: str,
    model: str = "qwen3.5-ocr",
    timeout: float = 60,
    max_tokens: int = 4096,
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    """Extract visible text via an OpenAI-compatible vision completion."""
    _check_key(api_key)
    path = Path(image_path)
    mime = mimetypes.guess_type(str(path))[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    with nullcontext(client) if client is not None else httpx.Client() as http:
        payload = _json(_request(
            http, "POST", base_url.rstrip("/") + "/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"}, timeout=timeout,
            json={
                "model": model,
                "messages": [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}},
                    {"type": "text", "text": "仅逐字转录图片中实际可见的文字，保留换行。不要解释、总结、推测或补全。没有文字时返回空字符串。"},
                ]}],
                "max_tokens": max(1, int(max_tokens)),
            },
        ))
    usage = _usage(payload.get("usage"))
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise EvidenceAPIError("OCR_RESPONSE_INVALID", usage=usage)
    choice = choices[0]
    finish_reason = choice.get("finish_reason")
    if finish_reason != "stop":
        code = "OCR_OUTPUT_TRUNCATED" if finish_reason == "length" else "OCR_OUTPUT_INCOMPLETE"
        raise EvidenceAPIError(code, usage=usage)
    message = choice.get("message")
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        raise EvidenceAPIError("OCR_TEXT_MISSING", usage=usage)
    text = message["content"].strip()
    return {
        "text": text, "status": "SUCCESS" if text else "NO_TEXT",
        "provider": "openai-compatible", "model": model, "usage": usage,
    }
