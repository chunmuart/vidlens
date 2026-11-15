"""Integration regressions for cumulative budgets and observable follow-up failures.

All model responses are local HTTP fixtures; these tests never call a paid API.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from test_flow import main, register, reset_database, upload  # noqa: F401
from vidlens.budget import BudgetExceeded, BudgetLimits, BudgetTracker


@pytest.fixture(autouse=True)
def isolated_runtime(monkeypatch: pytest.MonkeyPatch, reset_database: None) -> None:
    """Keep task execution synchronous and independent of local service settings."""
    monkeypatch.setattr(main, "publish_analysis", lambda _task_id: None)
    monkeypatch.setattr(main, "redis_client", lambda: None)
    monkeypatch.setenv("CHECKPOINT_ENABLED", "0")
    monkeypatch.setenv("AI_REQUEST_TIMEOUT_SECONDS", "180")
    monkeypatch.setenv("AI_INPUT_COST_PER_MILLION_TOKENS", "2")
    monkeypatch.setenv("AI_OUTPUT_COST_PER_MILLION_TOKENS", "4")
    monkeypatch.setenv("AI_CACHE_HIT_INPUT_COST_PER_MILLION_TOKENS", "0")


def response(message: dict[str, Any], *, usage: dict[str, Any] | None = None) -> httpx.Response:
    return httpx.Response(
        200,
        request=httpx.Request("POST", "https://unit.test/chat/completions"),
        json={"choices": [{"message": message}], "usage": usage or {
            "prompt_tokens": 30, "completion_tokens": 20, "total_tokens": 50,
        }},
    )


def tool_message(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {"content": None, "tool_calls": [{
        "id": "unit-call", "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
    }]}


def seeded_media(client: TestClient, headers: dict[str, str], *, report: bool = False) -> int:
    media_id = upload(client, headers, b"optimization-flow-video")
    with main.SessionLocal() as db:
        media = db.get(main.Media, media_id)
        segment = {"source": "ASR", "startMs": 1000, "endMs": 4000,
                   "content": "先建立数据库索引，再用 EXPLAIN 检查查询执行计划。"}
        Path(media.file_path + ".segments.json").write_text(
            json.dumps({"segments": [segment]}, ensure_ascii=False), encoding="utf-8",
        )
        main.replace_evidence(db, media_id, [segment])
        media.transcript_text = segment["content"]
        if report:
            media.ai_summary = "视频介绍数据库索引和查询执行计划。"
        db.commit()
    return media_id


@pytest.mark.parametrize(
    ("variable", "limit", "expected_limit", "request_count", "expected_tokens", "expected_cost"),
    [
        ("AGENT_MAX_ROUNDS", "1", "rounds", 1, 50, 0.00014),
        ("AGENT_MAX_ESTIMATED_TOKENS", "75", "tokens", 2, 100, 0.00028),
        ("AGENT_MAX_ESTIMATED_COST", "0.0002", "cost_usd", 2, 100, 0.00028),
    ],
)
def test_analysis_retry_keeps_prior_rounds_tokens_and_cost(
    monkeypatch: pytest.MonkeyPatch,
    variable: str,
    limit: str,
    expected_limit: str,
    request_count: int,
    expected_tokens: int,
    expected_cost: float,
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_post(_url: str, **kwargs: Any) -> httpx.Response:
        calls.append(kwargs)
        return response({"content": "模型已完成本轮调用"})

    class InterruptedProvider(main.OpenAICompatibleProvider):
        def run_agent(self, toolbox: Any, goal: str) -> dict[str, Any]:
            self._completion([{"role": "user", "content": goal}])
            raise RuntimeError("temporary interruption after a successful completion")

    monkeypatch.setenv(variable, limit)
    monkeypatch.setattr(main.httpx, "post", fake_post)
    monkeypatch.setattr(main, "provider", lambda: InterruptedProvider(
        "https://unit.test", "unit-key", "unit-model",
    ))

    with TestClient(main.app) as client:
        headers = register(client)
        media_id = seeded_media(client, headers)
        created = client.post("/analysis/ai", params={"id": media_id, "goal": "介绍数据库索引"}, headers=headers)
        assert created.status_code == 202
        task_id = created.json()["taskId"]
        with main.SessionLocal() as db:
            task = db.get(main.AnalysisTask, task_id)
            task.max_attempts = 4
            db.commit()

        assert main.process_analysis(task_id) == "RETRYING"
        first = client.get("/analysis/task-status", params={"taskId": task_id}, headers=headers).json()
        assert first["trace"]["budget"]["rounds"] == 1
        assert first["trace"]["budget"]["tokens"] == 50
        assert first["trace"]["budget"]["costUsd"] == pytest.approx(0.00014)
        assert first["trace"]["providerUsage"]["requestCount"] == 1

        assert main.process_analysis(task_id) == "COMPLETED"
        final = client.get("/analysis/task-status", params={"taskId": task_id}, headers=headers).json()
        assert len(calls) == request_count
        assert final["answerable"] is False
        assert "预算" in final["finalAnswer"]
        assert final["trace"]["attempt"] == 2
        assert final["trace"]["budgetExceeded"] is True
        assert final["evaluation"]["budgetLimit"] == expected_limit
        assert final["trace"]["budget"]["rounds"] == 2
        assert final["trace"]["budget"]["tokens"] == expected_tokens
        assert final["trace"]["budget"]["costUsd"] == pytest.approx(expected_cost)
        assert final["trace"]["providerUsage"]["requestCount"] == request_count
        assert [item["attempt"] for item in final["trace"]["providerRequests"]] == list(range(1, request_count + 1))
        assert main.process_analysis(task_id) == "SKIPPED"
        assert len(calls) == request_count


def test_completion_timeout_is_capped_by_remaining_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    now = [10.0]
    budget = BudgetTracker(BudgetLimits(max_duration_ms=1000), clock=lambda: now[0]).start()
    now[0] += 0.75
    captured: dict[str, Any] = {}

    def fake_post(_url: str, **kwargs: Any) -> httpx.Response:
        captured.update(kwargs)
        return response({"content": "ok"})

    monkeypatch.setattr(main.httpx, "post", fake_post)
    provider = main.OpenAICompatibleProvider("https://unit.test", "unit-key", "unit-model")
    provider.budget = budget
    assert provider._completion([{"role": "user", "content": "问题"}])["content"] == "ok"
    assert captured["timeout"] == pytest.approx(0.25)
    assert 0 < captured["timeout"] <= (budget.limits.max_duration_ms - budget.elapsed_ms) / 1000


def test_expired_duration_budget_prevents_network_request(monkeypatch: pytest.MonkeyPatch) -> None:
    now = [10.0]
    budget = BudgetTracker(BudgetLimits(max_duration_ms=1000), clock=lambda: now[0]).start()
    now[0] += 2.0

    def unexpected_post(*_args: Any, **_kwargs: Any) -> httpx.Response:
        pytest.fail("An expired duration budget must stop before an HTTP request")

    monkeypatch.setattr(main.httpx, "post", unexpected_post)
    provider = main.OpenAICompatibleProvider("https://unit.test", "unit-key", "unit-model")
    provider.budget = budget
    with pytest.raises(BudgetExceeded) as error:
        provider._completion([{"role": "user", "content": "问题"}])
    assert error.value.limit == "duration_ms"
    assert provider.provider_usage_summary()["requestCount"] == 0


def test_completion_uses_input_plus_output_when_total_tokens_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main.httpx, "post", lambda *_args, **_kwargs: response(
        {"content": "ok"}, usage={"prompt_tokens": 12, "completion_tokens": 7},
    ))
    provider = main.OpenAICompatibleProvider("https://unit.test", "unit-key", "unit-model")
    provider.budget = BudgetTracker().start()
    provider._completion([{"role": "user", "content": "问题"}])
    assert provider.budget.tokens == 19
    assert provider.provider_usage_summary()["totalTokens"] == 19
    assert provider.provider_usage_events()[0]["totalTokens"] == 19


def test_follow_up_budget_stop_is_visible_to_owner_in_memory_trace(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    def fake_post(_url: str, **kwargs: Any) -> httpx.Response:
        calls.append(kwargs)
        assert kwargs["json"]["tool_choice"]["function"]["name"] == "submit_follow_up_answer"
        return response(tool_message("submit_follow_up_answer", {
            "answerBasis": "EXPLICIT", "answer": "先建立数据库索引，再检查查询执行计划。", "evidenceIds": ["E001"],
        }))

    monkeypatch.setenv("AGENT_MAX_ROUNDS", "1")
    monkeypatch.setattr(main.httpx, "post", fake_post)
    monkeypatch.setattr(main, "provider", lambda: main.OpenAICompatibleProvider(
        "https://unit.test", "unit-key", "unit-model",
    ))
    with TestClient(main.app) as client:
        owner, other = register(client), register(client)
        media_id = seeded_media(client, owner, report=True)
        result = client.post("/analysis/follow-up", params={"id": media_id, "question": "视频如何介绍数据库索引？"}, headers=owner)
        assert result.status_code == 200
        assert "预算" in result.json()
        memory = client.get("/analysis/agent-memory", params={"id": media_id}, headers=owner).json()
        trace = memory["latestFollowUpTrace"]
        assert memory["sessionCount"] == 1
        assert memory["messageCount"] == 2
        assert trace == memory["sessions"][0]["latestFollowUpTrace"]
        assert trace["review"] == {"accepted": False, "reason": "BUDGET_EXCEEDED", "limit": "rounds"}
        assert trace["budget"]["rounds"] == 2
        assert trace["providerUsage"]["requestCount"] == 1
        assert trace["providerRequests"][0]["success"] is True
        assert client.get("/analysis/agent-memory", params={"id": media_id}, headers=other).status_code == 404
        assert client.post("/analysis/follow-up", params={"id": media_id, "question": "数据库索引？"}, headers=other).status_code == 404
        assert len(calls) == 1


def test_failed_follow_up_persists_new_session_and_failure_trace(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    def failing_post(_url: str, **kwargs: Any) -> httpx.Response:
        calls.append(kwargs)
        return httpx.Response(503, request=httpx.Request("POST", "https://unit.test/chat/completions"))

    monkeypatch.setattr(main.httpx, "post", failing_post)
    monkeypatch.setattr(main, "provider", lambda: main.OpenAICompatibleProvider(
        "https://unit.test", "unit-key", "unit-model",
    ))
    with TestClient(main.app) as client:
        owner, other = register(client), register(client)
        media_id = seeded_media(client, owner, report=True)
        assert client.get("/analysis/agent-memory", params={"id": media_id}, headers=owner).json()["sessionCount"] == 0
        result = client.post("/analysis/follow-up", params={"id": media_id, "question": "视频如何介绍数据库索引？"}, headers=owner)
        assert result.status_code == 502
        assert "失败" in result.json()["detail"]
        memory = client.get("/analysis/agent-memory", params={"id": media_id}, headers=owner).json()
        trace = memory["latestFollowUpTrace"]
        assert memory["sessionCount"] == 1
        assert memory["messageCount"] == 0
        assert trace == memory["sessions"][0]["latestFollowUpTrace"]
        assert trace["review"] == {"accepted": False, "reason": "SERVICE_ERROR", "errorType": "HTTPStatusError"}
        assert trace["providerUsage"]["failureCount"] == 1
        assert trace["providerRequests"][0]["statusCode"] == 503
        assert trace["providerRequests"][0]["success"] is False
        assert trace["budget"]["rounds"] == 1
        assert client.get("/analysis/agent-memory", params={"id": media_id}, headers=other).status_code == 404
        assert client.post("/analysis/follow-up", params={"id": media_id, "question": "数据库索引？"}, headers=other).status_code == 404
        assert len(calls) == 1
