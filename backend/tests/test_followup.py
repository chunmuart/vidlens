from __future__ import annotations

import json
from typing import Any

import pytest

from vidlens.followup import DISCLAIMER, REFUSAL, answer_with_evidence, rewrite_question


def response(name: str, arguments: Any) -> dict[str, Any]:
    return {"tool_calls": [{"function": {
        "name": name,
        "arguments": arguments if isinstance(arguments, str) else json.dumps(arguments, ensure_ascii=False),
    }}]}


class ScriptedProvider:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def _completion(self, messages, tools=None, tool_choice=None):
        self.calls.append({
            "name": tool_choice["function"]["name"],
            "phase": getattr(self, "_agent_phase", None),
            "payload": json.loads(messages[-1]["content"]),
        })
        assert self.responses, "Unexpected extra model call"
        return self.responses.pop(0)


TIMELINE = (
    "[120000-130000ms][ASR] 普通方案的高质量结果概率更高。\n"
    "[140000-145000ms][OCR] 紧急方案的执行速度更快。"
)
PASS = {"supported": True, "complete": True, "contradictionFree": True}


def answer_response(**changes: Any) -> dict[str, Any]:
    return response("submit_follow_up_answer", {
        "answerBasis": "EXPLICIT", "answer": "普通方案的高质量结果概率更高。",
        "evidenceIds": ["E001"], **changes,
    })


def test_complete_question_skips_rewriting_even_when_history_exists() -> None:
    provider = ScriptedProvider([])
    question = "Python与Java分别适合哪些场景？"

    result = rewrite_question(provider, question, [{"role": "user", "content": "之前的问题"}], "旧摘要")

    assert result == {"query": question, "rewritten": False, "needsClarification": False}
    assert provider.calls == []


@pytest.mark.parametrize("question", ["这个视频主要讲了什么？", "What does this video explain?", "Discuss iteration in Python."])
def test_current_video_or_complete_english_words_do_not_need_history(question: str) -> None:
    provider = ScriptedProvider([])
    result = rewrite_question(provider, question, [], "")
    assert result == {"query": question, "rewritten": False, "needsClarification": False}
    assert provider.calls == []


def test_reference_without_history_requests_clarification_without_a_model_call() -> None:
    provider = ScriptedProvider([])

    result = rewrite_question(provider, "它有什么缺点？", [], "")

    assert result == {"query": "它有什么缺点？", "rewritten": False, "needsClarification": True}
    assert provider.calls == []


def test_reference_is_rewritten_to_a_standalone_query_using_recent_history() -> None:
    provider = ScriptedProvider([response("submit_follow_up_query", {
        "query": "Java企业后端方案有什么缺点？", "needsClarification": False,
    })])
    history = [{"role": "user", "content": f"历史{index}"} for index in range(12)]

    result = rewrite_question(provider, "第二个方案有什么缺点？", history, "Java企业后端方案")

    assert result["query"] == "Java企业后端方案有什么缺点？"
    assert result["rewritten"] is True
    assert result["needsClarification"] is False
    assert provider.calls[0]["phase"] == "FOLLOWUP_REWRITE"
    assert provider.calls[0]["payload"]["history"] == history[-8:]
    assert provider.calls[0]["payload"]["question"] == "第二个方案有什么缺点？"


def test_memory_summary_can_resolve_a_reference_without_recent_messages() -> None:
    provider = ScriptedProvider([response("submit_follow_up_query", {
        "query": "普通方案的收益是什么？", "needsClarification": False,
    })])

    result = rewrite_question(provider, "它的收益是什么？", [], "刚讨论普通方案")

    assert result["needsClarification"] is False
    assert provider.calls[0]["payload"]["summary"] == "刚讨论普通方案"


@pytest.mark.parametrize("arguments", [
    {"query": "两个方案都有可能", "needsClarification": True},
    {"query": "", "needsClarification": False},
    {"query": "x" * 1001, "needsClarification": False},
    {"query": "擅自给出主题"},
    {"query": "擅自给出主题", "needsClarification": "false"},
    [],
])
def test_ambiguous_or_invalid_rewrite_keeps_the_original_and_requests_clarification(arguments) -> None:
    provider = ScriptedProvider([response("submit_follow_up_query", arguments)])

    result = rewrite_question(provider, "它有什么问题？", [{"role": "user", "content": "两个不同方案"}], "")

    assert result == {"query": "它有什么问题？", "rewritten": False, "needsClarification": True}


@pytest.mark.parametrize("timeline", ["", "历史摘要不是当前视频证据", "[02:00][ASR] 无法解析的时间轴"])
def test_no_current_evidence_refuses_without_a_model_call(timeline: str) -> None:
    provider = ScriptedProvider([])

    answer, review = answer_with_evidence(provider, "负责人是谁？", timeline)

    assert answer == REFUSAL
    assert review == {"accepted": False, "reason": "NO_EVIDENCE", "evidenceIds": []}
    assert provider.calls == []


@pytest.mark.parametrize("changes", [
    {"evidenceIds": ["E999"]},
    {"evidenceIds": []},
    {"evidenceIds": "E001"},
    {"evidenceIds": ["E001", 1]},
    {"evidenceIds": None},
    {"answer": ""},
    {"answerBasis": "UNANSWERABLE"},
    {"answerBasis": "EXTERNAL_KNOWLEDGE"},
])
def test_invalid_answer_or_citations_are_refused_before_semantic_verification(changes) -> None:
    provider = ScriptedProvider([answer_response(**changes)])

    answer, review = answer_with_evidence(provider, "普通方案的优势是什么？", TIMELINE)

    assert answer == REFUSAL
    assert review["accepted"] is False
    assert review["reason"] == "REFUSED_OR_INVALID_CITATIONS"
    assert [item["name"] for item in provider.calls] == ["submit_follow_up_answer"]


@pytest.mark.parametrize("raw", [
    None,
    "plain answer without a tool call",
    {},
    {"tool_calls": "invalid"},
    {"tool_calls": [None]},
    {"tool_calls": [{"function": "invalid"}]},
    response("wrong_tool", {"answer": "伪造答案"}),
    response("submit_follow_up_answer", "{not json"),
    response("submit_follow_up_answer", []),
    response("submit_follow_up_answer", None),
])
def test_malformed_answer_tool_output_fails_closed(raw) -> None:
    provider = ScriptedProvider([raw])

    answer, review = answer_with_evidence(provider, "普通方案的优势是什么？", TIMELINE)

    assert answer == REFUSAL
    assert review["accepted"] is False
    assert len(provider.calls) == 1


@pytest.mark.parametrize("key,value", [
    ("supported", False), ("complete", False), ("contradictionFree", False),
    ("supported", "true"), ("supported", 1), ("complete", None),
])
def test_semantic_rejection_or_invalid_boolean_cannot_be_overridden_by_valid_ids(key: str, value: Any) -> None:
    provider = ScriptedProvider([
        answer_response(), response("verify_follow_up_answer", {**PASS, key: value}),
    ])

    answer, review = answer_with_evidence(provider, "普通方案的优势是什么？", TIMELINE)

    assert answer == REFUSAL
    assert review == {"accepted": False, "reason": "SUPPORT_CHECK_FAILED", "evidenceIds": ["E001"]}
    assert len(provider.calls) == 2


def test_malformed_semantic_review_refuses_instead_of_accepting_existing_ids() -> None:
    provider = ScriptedProvider([answer_response(), response("verify_follow_up_answer", "{bad json")])

    answer, review = answer_with_evidence(provider, "普通方案的优势是什么？", TIMELINE)

    assert answer == REFUSAL
    assert review["reason"] == "SUPPORT_CHECK_FAILED"


@pytest.mark.parametrize("counterevidence", [
    "但该方案只在小数据量下更快。",
    "在大数据量下，该方案反而更慢。",
])
def test_follow_up_review_receives_uncited_limits_and_contradictions(counterevidence: str) -> None:
    timeline = (
        "[1000-2000ms][ASR] 该方案速度更快。\n"
        f"[2000-3000ms][OCR] {counterevidence}"
    )
    provider = ScriptedProvider([
        answer_response(answer="该方案在所有数据量下都更快。"),
        response("verify_follow_up_answer", {**PASS, "contradictionFree": False}),
    ])

    answer, review = answer_with_evidence(provider, "所有数据量下都更快吗？", timeline)

    payload = provider.calls[1]["payload"]
    assert [item["evidenceId"] for item in payload["evidence"]] == ["E001"]
    assert [item["evidenceId"] for item in payload["candidateEvidence"]] == ["E001", "E002"]
    assert payload["candidateEvidence"][1]["content"] == counterevidence
    assert answer == REFUSAL
    assert review == {"accepted": False, "reason": "SUPPORT_CHECK_FAILED", "evidenceIds": ["E001"]}
    assert len(provider.calls) == 2


def test_grounded_inference_adds_disclaimer_and_unique_references_after_semantic_approval() -> None:
    provider = ScriptedProvider([
        answer_response(answerBasis="GROUNDED_INFERENCE", answer="重视质量时可以优先普通方案。", evidenceIds=["E001", "E002", "E001"]),
        response("verify_follow_up_answer", PASS),
    ])

    answer, review = answer_with_evidence(provider, "重视质量时应该选择哪个方案？", TIMELINE)

    assert answer.startswith("重视质量时可以优先普通方案。")
    assert answer.count(DISCLAIMER) == 1
    assert answer.endswith("依据：[02:00] ASR E001；[02:20] OCR E002")
    assert review == {"accepted": True, "answerBasis": "GROUNDED_INFERENCE", "evidenceIds": ["E001", "E002"], "reason": "VERIFIED"}
    assert [item["phase"] for item in provider.calls] == ["FOLLOWUP_ANSWER", "FOLLOWUP_VERIFY"]
    verified = provider.calls[1]["payload"]["evidence"]
    assert [item["evidenceId"] for item in verified] == ["E001", "E002"]
    assert verified[1]["content"] == "紧急方案的执行速度更快。"


def test_inference_does_not_duplicate_an_existing_disclaimer() -> None:
    provider = ScriptedProvider([
        answer_response(answerBasis="GROUNDED_INFERENCE", answer="可以优先普通方案。" + DISCLAIMER),
        response("verify_follow_up_answer", PASS),
    ])

    answer, review = answer_with_evidence(provider, "选择哪个方案？", TIMELINE)

    assert review["accepted"] is True
    assert answer.count(DISCLAIMER) == 1


@pytest.mark.parametrize("basis", ["EXPLICIT", "SYNTHESIS"])
def test_verified_explicit_or_synthesis_answer_keeps_citations_without_inference_disclaimer(basis: str) -> None:
    provider = ScriptedProvider([answer_response(answerBasis=basis), response("verify_follow_up_answer", PASS)])

    answer, review = answer_with_evidence(provider, "普通方案的优势是什么？", TIMELINE)

    assert review["accepted"] is True
    assert DISCLAIMER not in answer
    assert answer.endswith("依据：[02:00] ASR E001")
