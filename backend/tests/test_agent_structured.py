from __future__ import annotations

import json
from typing import Any

import pytest

from vidlens.agent import AgentToolbox
from vidlens.agent_structured import (
    _critic_node,
    _normalize_evidence_plan,
    _normalize_verification,
    _retrieve_node,
    _verification_needs_deictic_audit,
    _write_node,
    run_structured_evidence_agent,
)


def tool_response(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "content": None,
        "tool_calls": [{
            "id": f"call-{name}",
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(arguments, ensure_ascii=False),
            },
        }],
    }


def toolbox(segments: list[dict[str, Any]]) -> AgentToolbox:
    return AgentToolbox(
        metadata={"mediaId": 1, "filename": "comparison.mp4", "status": "COMPLETED"},
        segments=segments,
        normalize_report=lambda value: value,
        evaluate_report=lambda report, _: {
            "structuredValid": bool(report.get("finalAnswer")),
            "evidenceSupportRate": 1.0,
            "criticPassed": bool(report.get("finalAnswer")),
            "citationCount": len(report.get("evidence", [])),
            "supportedCitationCount": len(report.get("evidence", [])),
        },
    )


class ComparisonProvider:
    def __init__(self, *, reject_first_critic: bool = False) -> None:
        self.reject_first_critic = reject_first_critic
        self.critic_calls = 0
        self.prompt_lengths: list[int] = []

    def _completion(self, messages, tools=None, tool_choice=None):
        self.prompt_lengths.append(sum(len(str(item.get("content") or "")) for item in messages))
        name = tool_choice["function"]["name"]
        if name == "submit_evidence_plan":
            return tool_response(name, {
                "answerMode": "COMPARISON",
                "requirements": [
                    {
                        "requirementId": "python-side",
                        "subQuestion": "Python 适合什么场景？",
                        "retrievalQuery": "Python 数据处理 场景",
                        "evidenceRole": "COMPARE_LEFT",
                        "completionPolicy": "COMPARE_SIDE",
                        "expectedSources": ["ASR"],
                        "required": True,
                    },
                    {
                        "requirementId": "java-side",
                        "subQuestion": "Java 适合什么场景？",
                        "retrievalQuery": "Java 企业后端 场景",
                        "evidenceRole": "COMPARE_RIGHT",
                        "completionPolicy": "COMPARE_SIDE",
                        "expectedSources": ["ASR"],
                        "required": True,
                    },
                ],
            })
        payload = json.loads(messages[-1]["content"])
        if name == "submit_evidence_verification":
            evidence = payload["ledger"]["evidence"]
            python_id = next(item["evidenceId"] for item in evidence if "Python" in item["content"])
            java_id = next(item["evidenceId"] for item in evidence if "Java" in item["content"])
            return tool_response(name, {
                "requirements": [
                    {
                        "requirementId": "python-side",
                        "supported": True,
                        "complete": True,
                        "evidenceIds": [python_id],
                        "missingInformation": "",
                        "contradictionEvidenceIds": [],
                    },
                    {
                        "requirementId": "java-side",
                        "supported": True,
                        "complete": True,
                        "evidenceIds": [java_id],
                        "missingInformation": "",
                        "contradictionEvidenceIds": [],
                    },
                ],
                "overallSufficient": True,
                "shouldRefuse": False,
                "refusalReason": "",
            })
        if name == "submit_grounded_report":
            verification = payload["verification"]
            evidence_ids = [
                evidence_id
                for item in verification["requirements"]
                for evidence_id in item["evidenceIds"]
            ]
            revised = "criticReview" in payload
            return tool_response(name, {
                "answerable": True,
                "finalAnswer": (
                    "Python 适合数据处理，Java 适合企业后端。"
                    if revised or not self.reject_first_critic
                    else "两者都适合所有场景。"
                ),
                "title": "语言场景比较",
                "conclusions": ["分别覆盖 Python 和 Java。"],
                "evidenceIds": evidence_ids,
                "suggestions": [],
            })
        if name == "submit_critic_review":
            self.critic_calls += 1
            rejected = self.reject_first_critic and self.critic_calls == 1
            return tool_response(name, {
                "approved": not rejected,
                "answerabilityCorrect": True,
                "allRequiredSlotsCovered": not rejected,
                "contradictionFree": not rejected,
                "externalKnowledgeFree": not rejected,
                "citationIdsValid": True,
                "missingRequirementIds": ["python-side", "java-side"] if rejected else [],
                "unsupportedClaims": ["所有场景"] if rejected else [],
                "contradictions": [],
                "revisionInstruction": "分别回答两个槽位，不要扩大为所有场景。" if rejected else "",
            })
        raise AssertionError(name)


def comparison_segments() -> list[dict[str, Any]]:
    return [
        {
            "segmentId": "python",
            "source": "ASR",
            "startMs": 10000,
            "endMs": 15000,
            "content": "Python 更适合数据处理和快速脚本。",
        },
        {
            "segmentId": "java",
            "source": "ASR",
            "startMs": 30000,
            "endMs": 35000,
            "content": "Java 更适合大型企业后端服务。",
        },
    ]


def test_enumeration_retrieval_expands_a_wider_context_window() -> None:
    current_toolbox = toolbox([
        {
            "segmentId": "definition-1",
            "source": "ASR",
            "startMs": 10000,
            "endMs": 15000,
            "content": "第一种是参数量大。",
        },
        {
            "segmentId": "summary",
            "source": "ASR",
            "startMs": 60000,
            "endMs": 65000,
            "content": "这三种大都具备。",
        },
    ])

    result = _retrieve_node(current_toolbox, {
        "goal": "三种大分别是什么？",
        "evidence_plan": {
            "answerMode": "EXHAUSTIVE_LIST",
            "requirements": [{
                "requirementId": "R1",
                "subQuestion": "三种大分别是什么？",
                "retrievalQuery": "三种 大",
                "evidenceRole": "ENUMERATION",
                "completionPolicy": "ALL_RELEVANT",
                "expectedSources": ["ASR"],
                "required": True,
            }],
        },
    })

    window = next(item for item in current_toolbox.trace() if item["tool"] == "get_evidence_window")
    assert '"before_ms":60000' in window["arguments"]
    assert '"after_ms":30000' in window["arguments"]
    assert result["evidence_ledger"]["statistics"]["compactEvidenceCount"] == 2


def test_direct_deictic_citation_requires_a_second_verifier_audit() -> None:
    plan = {
        "requirements": [{
            "requirementId": "R1",
            "completionPolicy": "DIRECT",
            "candidateEvidenceIds": ["E001", "E002"],
        }],
    }
    ledger = {
        "evidence": [
            {"evidenceId": "E001", "content": "K3 使用了这篇论文中的技术。"},
            {"evidenceId": "E002", "content": "这个技术叫注意力残差。"},
        ],
    }
    verification = {
        "requirements": [{
            "requirementId": "R1",
            "complete": True,
            "evidenceIds": ["E001"],
        }],
    }

    assert _verification_needs_deictic_audit(plan, ledger, verification) is True


def test_structured_agent_covers_each_comparison_slot_with_evidence_ids() -> None:
    provider = ComparisonProvider()

    result = run_structured_evidence_agent(
        provider,
        toolbox(comparison_segments()),
        "Python 与 Java 分别适合什么场景？",
    )

    graph = result["agentGraph"]
    assert result["accepted"] is True
    assert graph["promptVersion"] == "video-evidence-agent-v5.4-full-ledger-audit"
    assert graph["evidencePlan"]["answerMode"] == "COMPARISON"
    assert graph["evidenceVerification"]["fullyCovered"] is True
    assert len(graph["evidenceVerification"]["completeRequirementIds"]) == 2
    assert graph["modelCalls"] == 4
    assert graph["revisionCount"] == 0
    assert {item["evidenceId"] for item in result["report"]["evidence"]} == {"E001", "E002"}
    assert max(provider.prompt_lengths) < 20000


def test_structured_agent_allows_one_critic_driven_revision() -> None:
    provider = ComparisonProvider(reject_first_critic=True)

    result = run_structured_evidence_agent(
        provider,
        toolbox(comparison_segments()),
        "Python 与 Java 分别适合什么场景？",
    )

    assert result["accepted"] is True
    assert result["agentGraph"]["revisionCount"] == 1
    assert result["agentGraph"]["modelCalls"] == 6
    assert provider.critic_calls == 2
    assert result["report"]["finalAnswer"] == "Python 适合数据处理，Java 适合企业后端。"


class AlwaysRejectCriticProvider(ComparisonProvider):
    def _completion(self, messages, tools=None, tool_choice=None):
        name = tool_choice["function"]["name"]
        if name != "submit_critic_review":
            return super()._completion(messages, tools, tool_choice)
        self.critic_calls += 1
        return tool_response(name, {
            "approved": False,
            "answerabilityCorrect": True,
            "allRequiredSlotsCovered": True,
            "contradictionFree": True,
            "externalKnowledgeFree": True,
            "citationIdsValid": True,
            "missingRequirementIds": [],
            "unsupportedClaims": [],
            "contradictions": [],
            "revisionInstruction": "再次确认措辞。",
        })


def test_structured_agent_refuses_when_critic_still_rejects_after_one_revision() -> None:
    provider = AlwaysRejectCriticProvider()

    result = run_structured_evidence_agent(
        provider,
        toolbox(comparison_segments()),
        "Python 与 Java 分别适合什么场景？",
    )

    assert result["accepted"] is True
    assert result["agentGraph"]["revisionCount"] == 1
    assert result["agentGraph"]["modelCalls"] == 6
    assert result["agentGraph"]["acceptanceMode"] == "SAFE_REFUSAL_AFTER_CRITIC_REVISION"
    assert result["agentGraph"]["repairCount"] == 1
    assert result["agentGraph"]["criticReview"]["modelGatePassed"] is False
    assert result["report"]["answerable"] is False
    assert result["report"]["evidence"] == []


def test_single_direct_plan_cannot_be_upgraded_to_all_relevant() -> None:
    plan = _normalize_evidence_plan({
        "answerMode": "SINGLE",
        "requirements": [{
            "requirementId": "R1",
            "subQuestion": "大模型的大字是什么意思？",
            "retrievalQuery": "大模型 大字 含义",
            "evidenceRole": "DIRECT",
            "completionPolicy": "ALL_RELEVANT",
            "expectedSources": ["ASR", "OCR"],
            "required": True,
        }],
    }, "大模型的大字是什么意思？")

    assert plan is not None
    assert plan["requirements"][0]["completionPolicy"] == "DIRECT"
    assert plan["requirements"][0]["subQuestion"] == "大模型的大字是什么意思？"


def test_single_target_contrast_is_not_expanded_into_comparison() -> None:
    plan = _normalize_evidence_plan({
        "answerMode": "COMPARISON",
        "requirements": [
            {
                "requirementId": "R1",
                "subQuestion": "其他模型使用什么路径？",
                "retrievalQuery": "其他模型 残差 路径",
                "evidenceRole": "COMPARE_LEFT",
                "completionPolicy": "COMPARE_SIDE",
                "expectedSources": ["ASR"],
                "required": True,
            },
            {
                "requirementId": "R2",
                "subQuestion": "Kimi 选择的另一条路径是什么？",
                "retrievalQuery": "Kimi 另一条路径",
                "evidenceRole": "COMPARE_RIGHT",
                "completionPolicy": "COMPARE_SIDE",
                "expectedSources": ["ASR"],
                "required": True,
            },
        ],
    }, "Kimi 没有与其他模型一样，而选择的另一条路径是什么？")

    assert plan is not None
    assert plan["answerMode"] == "SINGLE"
    assert len(plan["requirements"]) == 1
    assert plan["requirements"][0]["completionPolicy"] == "DIRECT"
    assert plan["requirements"][0]["subQuestion"] == "Kimi 没有与其他模型一样，而选择的另一条路径是什么？"


def test_single_target_sequence_is_not_expanded_into_full_enumeration() -> None:
    plan = _normalize_evidence_plan({
        "answerMode": "SEQUENCE",
        "requirements": [
            {
                "requirementId": "R1",
                "subQuestion": "四个任务是什么？",
                "retrievalQuery": "四个任务",
                "evidenceRole": "ENUMERATION",
                "completionPolicy": "ALL_RELEVANT",
                "expectedSources": ["ASR", "OCR"],
                "required": True,
            },
            {
                "requirementId": "R2",
                "subQuestion": "工作流之后的下一个任务是什么？",
                "retrievalQuery": "工作流 之后 下一个任务",
                "evidenceRole": "SEQUENCE_STEP",
                "completionPolicy": "ORDERED_STEP",
                "expectedSources": ["ASR", "OCR"],
                "required": True,
            },
        ],
    }, "“文生图生视频的工作流”之后的任务是什么？")

    assert plan is not None
    assert plan["answerMode"] == "SINGLE"
    assert len(plan["requirements"]) == 1
    assert plan["requirements"][0]["completionPolicy"] == "DIRECT"


@pytest.mark.parametrize("explanation", [
    "E001 说‘我老婆这么问我’，这直接回答了问题，因此证据直接支持了答案。",
    "E001 只是提到了主题，不能认为证据直接支持答案。",
    "E001 并不意味着这直接回答了问题。",
    "不能依据 E001 就声称答案正确、证据充分。",
])
def test_verifier_explanation_cannot_override_an_explicit_rejection(explanation: str) -> None:
    plan = {
        "requirements": [{
            "requirementId": "R1",
            "required": True,
        }],
    }
    ledger = {
        "evidence": [{"evidenceId": "E001", "content": "前几天我老婆这么问我。"}],
    }
    raw = {
        "requirements": [{
            "requirementId": "R1",
            "supported": False,
            "complete": False,
            "evidenceIds": [],
            "missingInformation": explanation,
            "contradictionEvidenceIds": [],
        }],
        "overallSufficient": False,
        "shouldRefuse": True,
        "refusalReason": "自相矛盾",
    }

    verification = _normalize_verification(raw, plan, ledger)

    assert verification["overallSufficient"] is False
    assert verification["shouldRefuse"] is True
    assert verification["requirements"][0]["supported"] is False
    assert verification["requirements"][0]["complete"] is False
    assert verification["requirements"][0]["evidenceIds"] == []
    assert verification["requirements"][0]["missingInformation"] == explanation


@pytest.mark.parametrize("field,value", [
    ("supported", "false"), ("supported", "true"), ("supported", 1),
    ("complete", "false"), ("complete", "true"), ("complete", 1),
])
def test_verifier_requires_actual_boolean_verdicts(field: str, value: Any) -> None:
    raw = {"requirementId": "R1", "supported": True, "complete": True,
           "evidenceIds": ["E001"], field: value}
    result = _normalize_verification(
        {"requirements": [raw]},
        {"requirements": [{"requirementId": "R1", "required": True}]},
        {"evidence": [{"evidenceId": "E001", "content": "直接答案。"}]},
    )
    assert result["overallSufficient"] is False
    assert result["shouldRefuse"] is True


@pytest.mark.parametrize("counterevidence", [
    "但该方案只在小数据量下更快。",
    "在大数据量下，该方案反而更慢。",
])
def test_critic_receives_uncited_limits_and_contradictions(counterevidence: str) -> None:
    ledger = {"evidence": [
        {"evidenceId": "E001", "source": "ASR", "startMs": 1000,
         "content": "该方案速度更快。"},
        {"evidenceId": "E002", "source": "OCR", "startMs": 2000,
         "content": counterevidence},
    ]}
    payloads: list[dict[str, Any]] = []

    class CounterevidenceCritic:
        def _completion(self, messages, tools=None, tool_choice=None):
            assert tool_choice["function"]["name"] == "submit_critic_review"
            payload = json.loads(messages[-1]["content"])
            payloads.append(payload)
            # Simulate a semantic reviewer rejecting the visible qualification.
            assert payload["ledger"]["evidence"][1]["content"] == counterevidence
            return tool_response("submit_critic_review", {
                "approved": False, "answerabilityCorrect": True,
                "allRequiredSlotsCovered": True, "contradictionFree": False,
                "externalKnowledgeFree": True, "citationIdsValid": True,
                "missingRequirementIds": [], "unsupportedClaims": [],
                "contradictions": [counterevidence],
                "revisionInstruction": "补充数据量限制，避免无条件声称更快。",
            })

    current_toolbox = toolbox([])
    result = _critic_node(CounterevidenceCritic(), current_toolbox, {
        "goal": "该方案在所有数据量下都更快吗？",
        "evidence_plan": {"requirements": [{"requirementId": "R1", "required": True}]},
        "evidence_ledger": ledger,
        "verification": {
            "overallSufficient": True, "requiredRequirementIds": ["R1"],
            "requirements": [{"requirementId": "R1", "evidenceIds": ["E001"]}],
        },
        "draft_report": {
            "answerable": True, "finalAnswer": "该方案在所有数据量下都更快。",
            "evidenceIds": ["E001"],
        },
    })

    assert len(payloads) == 1
    assert [item["evidenceId"] for item in payloads[0]["citedEvidence"]] == ["E001"]
    assert [item["evidenceId"] for item in payloads[0]["ledger"]["evidence"]] == ["E001", "E002"]
    assert result["critic_review"]["deterministicGatePassed"] is True
    assert result["critic_review"]["modelGatePassed"] is False
    assert result["accepted"] is False
    assert result["last_report"] is None
    assert not any(item["tool"] == "generate_report" for item in current_toolbox.trace())


class RefusalProvider:
    def _completion(self, messages, tools=None, tool_choice=None):
        name = tool_choice["function"]["name"]
        if name == "submit_evidence_plan":
            return tool_response(name, {
                "answerMode": "SINGLE",
                "requirements": [{
                    "requirementId": "owner",
                    "subQuestion": "负责人是谁？",
                    "retrievalQuery": "日志系统 负责人",
                    "evidenceRole": "DIRECT",
                    "completionPolicy": "DIRECT",
                    "expectedSources": ["ASR", "OCR"],
                    "required": True,
                }],
            })
        if name == "submit_evidence_verification":
            return tool_response(name, {
                "requirements": [{
                    "requirementId": "owner",
                    "supported": False,
                    "complete": False,
                    "evidenceIds": [],
                    "missingInformation": "标题只说明主题，没有负责人姓名。",
                    "contradictionEvidenceIds": [],
                }],
                "overallSufficient": False,
                "shouldRefuse": True,
                "refusalReason": "视频未说明负责人。",
            })
        if name == "submit_grounded_report":
            return tool_response(name, {
                "answerable": False,
                "finalAnswer": "根据现有证据，无法确定日志系统负责人。",
                "title": "视频证据不足",
                "conclusions": ["缺少负责人姓名证据。"],
                "evidenceIds": [],
                "suggestions": [],
            })
        if name == "submit_critic_review":
            return tool_response(name, {
                "approved": True,
                "answerabilityCorrect": True,
                "allRequiredSlotsCovered": False,
                "contradictionFree": True,
                "externalKnowledgeFree": True,
                "citationIdsValid": True,
                "missingRequirementIds": ["owner"],
                "unsupportedClaims": [],
                "contradictions": [],
                "revisionInstruction": "",
            })
        raise AssertionError(name)


class VerifierFalseNegativeProvider(ComparisonProvider):
    def __init__(self) -> None:
        super().__init__()
        self.verifier_calls = 0

    def _completion(self, messages, tools=None, tool_choice=None):
        name = tool_choice["function"]["name"]
        if name != "submit_evidence_verification":
            return super()._completion(messages, tools, tool_choice)
        self.verifier_calls += 1
        if self.verifier_calls > 1:
            return super()._completion(messages, tools, tool_choice)
        return tool_response(name, {
            "requirements": [
                {
                    "requirementId": "python-side",
                    "supported": False,
                    "complete": False,
                    "evidenceIds": [],
                    "missingInformation": "首次误判",
                    "contradictionEvidenceIds": [],
                },
                {
                    "requirementId": "java-side",
                    "supported": False,
                    "complete": False,
                    "evidenceIds": [],
                    "missingInformation": "首次误判",
                    "contradictionEvidenceIds": [],
                },
            ],
            "overallSufficient": False,
            "shouldRefuse": True,
            "refusalReason": "首次误判",
        })


def test_structured_agent_audits_a_verifier_false_negative_once() -> None:
    provider = VerifierFalseNegativeProvider()

    result = run_structured_evidence_agent(
        provider,
        toolbox(comparison_segments()),
        "Python 与 Java 分别适合什么场景？",
    )

    assert result["accepted"] is True
    assert result["agentGraph"]["verificationPasses"] == 2
    assert result["agentGraph"]["modelCalls"] == 5
    assert result["agentGraph"]["evidenceVerification"]["overallSufficient"] is True
    assert provider.verifier_calls == 2


def test_structured_agent_refuses_when_title_is_related_but_answer_is_missing() -> None:
    related_title = [{
        "segmentId": "title",
        "source": "OCR",
        "startMs": 0,
        "endMs": 15000,
        "content": "日志系统事故复盘",
    }]

    result = run_structured_evidence_agent(
        RefusalProvider(),
        toolbox(related_title),
        "视频是否说明日志系统负责人是谁？",
    )

    assert result["accepted"] is True
    assert result["report"]["answerable"] is False
    assert result["report"]["evidence"] == []
    assert result["report"]["finalAnswer"] == "视频未提供足够证据，无法从视频确定答案。"
    assert "无法从视频确定答案" in result["report"]["finalAnswer"]
    assert result["agentGraph"]["evidenceVerification"]["shouldRefuse"] is True


class SummaryProvider:
    def __init__(self) -> None:
        self.phase_names: list[str] = []

    def _completion(self, messages, tools=None, tool_choice=None):
        name = tool_choice["function"]["name"]
        self.phase_names.append(name)
        payload = json.loads(messages[-1]["content"])
        if name == "submit_evidence_verification":
            evidence_ids = [item["evidenceId"] for item in payload["ledger"]["evidence"][:6]]
            return tool_response(name, {
                "requirements": [{
                    "requirementId": "summary",
                    "supported": True,
                    "complete": True,
                    "supportLevel": "SYNTHESIS",
                    "evidenceIds": evidence_ids,
                    "missingInformation": "",
                    "contradictionEvidenceIds": [],
                }],
                "overallSufficient": True,
                "shouldRefuse": False,
                "refusalReason": "",
            })
        if name == "submit_grounded_report":
            evidence_ids = payload["verification"]["requirements"][0]["evidenceIds"]
            return tool_response(name, {
                "answerable": True,
                "finalAnswer": "视频围绕游戏机制展开，依次分析机制表现、实战影响和作者结论。",
                "title": "视频主要内容",
                "conclusions": ["视频讨论了游戏机制及其实战影响。"],
                "evidenceIds": evidence_ids,
                "suggestions": [],
            })
        if name == "submit_critic_review":
            return tool_response(name, {
                "approved": True,
                "answerabilityCorrect": True,
                "allRequiredSlotsCovered": True,
                "contradictionFree": True,
                "externalKnowledgeFree": True,
                "citationIdsValid": True,
                "missingRequirementIds": [],
                "unsupportedClaims": [],
                "contradictions": [],
                "revisionInstruction": "",
            })
        raise AssertionError(name)


def test_video_summary_uses_representative_timeline_without_planner_over_decomposition() -> None:
    segments = [
        {
            "segmentId": f"asr-{index}",
            "source": "ASR",
            "startMs": index * 10000,
            "endMs": index * 10000 + 5000,
            "content": f"第 {index} 段游戏机制讲解",
        }
        for index in range(12)
    ] + [
        {
            "segmentId": f"ocr-{index}",
            "source": "OCR",
            "startMs": index * 20000,
            "endMs": index * 20000 + 1000,
            "content": f"机制参数画面 {index}",
        }
        for index in range(6)
    ]
    provider = SummaryProvider()
    current_toolbox = toolbox(segments)

    result = run_structured_evidence_agent(provider, current_toolbox, "这个视频主要讲了什么？")

    graph = result["agentGraph"]
    assert result["report"]["answerable"] is True
    assert graph["plannerMode"] == "DETERMINISTIC_SUMMARY"
    assert graph["evidencePlan"]["answerMode"] == "SYNTHESIS"
    assert graph["evidencePlan"]["requirements"][0]["completionPolicy"] == "REPRESENTATIVE"
    assert graph["evidenceLedger"]["statistics"]["compactEvidenceCount"] == 18
    assert graph["evidenceVerification"]["refusalReason"] == ""
    assert "submit_evidence_plan" not in provider.phase_names
    assert current_toolbox.trace()[0]["tool"] == "get_timeline_overview"


def test_grounded_inference_writer_appends_explicit_uncertainty_notice() -> None:
    class InferenceWriter:
        def _completion(self, messages, tools=None, tool_choice=None):
            return tool_response("submit_grounded_report", {
                "answerable": True,
                "finalAnswer": "作者可能认为这个机制的收益低于风险。",
                "title": "机制判断",
                "conclusions": ["收益可能低于风险。"],
                "evidenceIds": ["E001"],
                "suggestions": [],
            })

    state = {
        "goal": "作者是否推荐使用这个机制？",
        "evidence_plan": {"answerMode": "SINGLE", "requirements": []},
        "verification": {
            "overallSufficient": True,
            "shouldRefuse": False,
            "requirements": [{
                "requirementId": "R1",
                "complete": True,
                "supportLevel": "GROUNDED_INFERENCE",
                "evidenceIds": ["E001"],
            }],
        },
        "evidence_ledger": {
            "evidence": [{
                "evidenceId": "E001",
                "segmentId": "asr-1",
                "source": "ASR",
                "startMs": 10000,
                "endMs": 15000,
                "content": "这个机制虽然有收益，但失败后的代价很高。",
            }],
        },
        "model_calls": 0,
        "context_chars": {},
    }

    written = _write_node(InferenceWriter(), state)

    assert written["draft_report"]["answerable"] is True
    assert written["draft_report"]["finalAnswer"].endswith(
        "此答案为基于视频证据的推测，视频没有明确说明。"
    )


class InvalidCitationProvider(ComparisonProvider):
    def _completion(self, messages, tools=None, tool_choice=None):
        name = tool_choice["function"]["name"]
        if name == "submit_grounded_report":
            return tool_response(name, {
                "answerable": True,
                "finalAnswer": "引用不存在。",
                "title": "无效引用",
                "conclusions": ["无效"],
                "evidenceIds": ["E999"],
                "suggestions": [],
            })
        return super()._completion(messages, tools, tool_choice)


def test_structured_agent_rejects_unknown_evidence_ids_even_when_model_critic_approves() -> None:
    result = run_structured_evidence_agent(
        InvalidCitationProvider(),
        toolbox(comparison_segments()),
        "Python 与 Java 分别适合什么场景？",
    )

    assert result["accepted"] is True
    assert result["report"]["answerable"] is False
    assert result["report"]["evidence"] == []
    assert result["agentGraph"]["acceptanceMode"] == "SAFE_REFUSAL_AFTER_CRITIC_REVISION"


def evidence(segment_id: str, content: str = "证据", *, source: str = "ASR", at: int = 0) -> dict[str, Any]:
    return {
        "segmentId": segment_id, "source": source,
        "startMs": at, "endMs": at + 1000, "content": content,
    }


def requirement(requirement_id: str, *, query: str = "", sources: list[str] | None = None) -> dict[str, Any]:
    return {
        "requirementId": requirement_id, "subQuestion": query or requirement_id,
        "retrievalQuery": query or requirement_id, "evidenceRole": "DIRECT",
        "completionPolicy": "DIRECT", "expectedSources": sources or ["ASR", "OCR"],
        "required": True,
    }


class FixedCandidates:
    def __init__(self, matches: dict[str, list[dict[str, Any]]], windows: dict[int, list[dict[str, Any]]]) -> None:
        self.matches = matches
        self.windows = windows

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name == "search_timeline":
            return {"matches": self.matches.get(arguments["query"], [])}
        if name == "get_evidence_window":
            return {"segments": self.windows.get(arguments["timestamp_ms"], [])}
        raise AssertionError(name)


def test_direct_candidates_are_shared_fairly_before_any_slot_adds_neighbors() -> None:
    requirements = [requirement(f"R{index}") for index in range(6)]
    matches = {
        f"R{index}": [evidence(f"R{index}-{rank}", at=index * 100000) for rank in range(8)]
        for index in range(6)
    }
    windows = {
        index * 100000: [evidence(f"neighbor-{index}-{rank}") for rank in range(16)]
        for index in range(6)
    }

    ledger = _retrieve_node(FixedCandidates(matches, windows), {
        "goal": "分别解释六个问题",
        "evidence_plan": {"answerMode": "MULTI_PART", "requirements": requirements},
    })["evidence_ledger"]

    assert len(ledger["evidence"]) == 18
    assert [item["segmentId"] for item in ledger["evidence"][:6]] == [f"R{index}-0" for index in range(6)]
    assert all(len(item["candidateEvidenceIds"]) == 3 for item in ledger["requirements"])
    assert not any(item["segmentId"].startswith("neighbor-") for item in ledger["evidence"])


def test_shared_evidence_uses_one_id_and_remains_linked_to_both_requirements() -> None:
    shared = evidence("shared")
    current_toolbox = FixedCandidates({"R1": [shared], "R2": [shared]}, {})

    ledger = _retrieve_node(current_toolbox, {
        "goal": "两个问题",
        "evidence_plan": {"answerMode": "MULTI_PART", "requirements": [requirement("R1"), requirement("R2")]},
    })["evidence_ledger"]

    assert len(ledger["evidence"]) == 1
    assert ledger["evidence"][0]["requirementIds"] == ["R1", "R2"]
    assert [item["candidateEvidenceIds"] for item in ledger["requirements"]] == [["E001"], ["E001"]]


def test_time_and_source_candidates_get_space_even_after_a_large_first_requirement() -> None:
    first = requirement("R1")
    timed = {**requirement("R2", query="00:30画面的数字", sources=["OCR"]), "completionPolicy": "TEMPORAL_WINDOW"}
    current_toolbox = FixedCandidates(
        {"R1": [evidence(f"asr-{index}") for index in range(8)], "00:30画面的数字": []},
        {0: [evidence(f"extra-{index}") for index in range(16)], 30000: [
            evidence("wrong-source", source="ASR", at=30000),
            evidence("actual-screen", "42", source="OCR", at=30000),
        ]},
    )

    ledger = _retrieve_node(current_toolbox, {
        "goal": "分别解释原理和00:30的画面数字",
        "evidence_plan": {"answerMode": "MULTI_PART", "requirements": [first, timed]},
    })["evidence_ledger"]

    by_id = {item["evidenceId"]: item for item in ledger["evidence"]}
    selected = [by_id[item] for item in ledger["requirements"][1]["candidateEvidenceIds"]]
    assert [item["segmentId"] for item in selected] == ["actual-screen"]
    assert selected[0]["startMs"] == 30000


class RepairOwnerProvider(RefusalProvider):
    def __init__(self, *, reject_answer: bool = False) -> None:
        self.reject_answer = reject_answer
        self.ledgers: list[dict[str, Any]] = []
        self.critic_calls = 0

    def _completion(self, messages, tools=None, tool_choice=None):
        name = tool_choice["function"]["name"]
        if name == "submit_evidence_verification":
            ledger = json.loads(messages[-1]["content"])["ledger"]
            self.ledgers.append(ledger)
            owner = next((item for item in ledger["evidence"] if "负责人是林" in item["content"]), None)
            return tool_response(name, {
                "requirements": [{
                    "requirementId": "owner", "supported": owner is not None, "complete": owner is not None,
                    "supportLevel": "DIRECT" if owner else "NONE",
                    "evidenceIds": [owner["evidenceId"]] if owner else [],
                    "missingInformation": "" if owner else "需要负责人姓名",
                    "contradictionEvidenceIds": [],
                }],
                "overallSufficient": owner is not None, "shouldRefuse": owner is None,
                "refusalReason": "" if owner else "视频未说明负责人",
            })
        if name == "submit_grounded_report":
            verification = json.loads(messages[-1]["content"])["verification"]
            if verification["overallSufficient"]:
                return tool_response(name, {
                    "answerable": True, "finalAnswer": "日志系统负责人是林。", "title": "负责人",
                    "conclusions": ["负责人是林。"],
                    "evidenceIds": verification["requirements"][0]["evidenceIds"], "suggestions": [],
                })
        if name == "submit_critic_review":
            self.critic_calls += 1
            rejected = self.reject_answer
            return tool_response(name, {
                "approved": not rejected, "answerabilityCorrect": True, "allRequiredSlotsCovered": True,
                "contradictionFree": True, "externalKnowledgeFree": not rejected, "citationIdsValid": True,
                "missingRequirementIds": [], "unsupportedClaims": ["姓名无依据"] if rejected else [],
                "contradictions": [], "revisionInstruction": "删除无依据姓名" if rejected else "",
            })
        return super()._completion(messages, tools, tool_choice)


class SecondSearchFindsOwner:
    def __init__(self, *, answer_exists: bool = True) -> None:
        self.answer_exists = answer_exists
        self.queries: list[str] = []

    def search(self, query, top_k=8, sources=None):
        self.queries.append(query)
        if len(self.queries) > 1 and self.answer_exists:
            return {"matches": [evidence("owner-name", "日志系统负责人是林。", at=600000)]}
        return {"matches": [evidence("title", "日志系统事故复盘", source="OCR")]}


def owner_toolbox(retriever: SecondSearchFindsOwner) -> AgentToolbox:
    current_toolbox = toolbox([
        evidence("title", "日志系统事故复盘", source="OCR"),
        *([evidence("owner-name", "日志系统负责人是林。", at=600000)] if retriever.answer_exists else []),
    ])
    current_toolbox.retriever = retriever
    return current_toolbox


def test_missing_evidence_is_searched_once_and_existing_evidence_ids_stay_stable() -> None:
    provider = RepairOwnerProvider()
    retriever = SecondSearchFindsOwner()

    result = run_structured_evidence_agent(provider, owner_toolbox(retriever), "日志系统负责人是谁？")

    graph = result["agentGraph"]
    assert result["report"]["answerable"] is True
    assert graph["repairCount"] == graph["retrievalRepairCount"] == 1
    assert graph["repairDetails"]["requirementIds"] == ["owner"]
    assert graph["repairDetails"]["newEvidenceCount"] == 1
    assert graph["modelCalls"] == 5
    assert len(retriever.queries) == 2
    assert "需要负责人姓名" in retriever.queries[1]
    first_ids = {item["segmentId"]: item["evidenceId"] for item in provider.ledgers[0]["evidence"]}
    second_ids = {item["segmentId"]: item["evidenceId"] for item in provider.ledgers[1]["evidence"]}
    assert first_ids["title"] == second_ids["title"]
    assert second_ids["owner-name"] != first_ids["title"]


def test_still_missing_after_one_search_refuses_without_another_repair() -> None:
    provider = RepairOwnerProvider()
    retriever = SecondSearchFindsOwner(answer_exists=False)

    result = run_structured_evidence_agent(provider, owner_toolbox(retriever), "日志系统负责人是谁？")

    assert result["report"]["answerable"] is False
    assert result["report"]["evidence"] == []
    assert result["agentGraph"]["repairCount"] == 1
    assert len(retriever.queries) == 2
    assert len(provider.ledgers) == 2


def test_critic_cannot_start_a_revision_after_verifier_spent_the_repair_allowance() -> None:
    provider = RepairOwnerProvider(reject_answer=True)
    retriever = SecondSearchFindsOwner()

    result = run_structured_evidence_agent(provider, owner_toolbox(retriever), "日志系统负责人是谁？")

    assert result["report"]["answerable"] is False
    assert result["agentGraph"]["repairCount"] == 1
    assert result["agentGraph"]["revisionCount"] == 0
    assert result["agentGraph"]["criticReview"]["modelGatePassed"] is False
    assert result["agentGraph"]["criticReview"]["deterministicGatePassed"] is True
    assert provider.critic_calls == 1
    assert len(retriever.queries) == 2


def test_critic_missing_requirement_can_use_the_single_targeted_search() -> None:
    class MissingOnceProvider(ComparisonProvider):
        def _completion(self, messages, tools=None, tool_choice=None):
            if tool_choice["function"]["name"] != "submit_critic_review":
                return super()._completion(messages, tools, tool_choice)
            self.critic_calls += 1
            missing = self.critic_calls == 1
            return tool_response("submit_critic_review", {
                "approved": not missing, "answerabilityCorrect": True,
                "allRequiredSlotsCovered": not missing, "contradictionFree": True,
                "externalKnowledgeFree": True, "citationIdsValid": True,
                "missingRequirementIds": ["java-side"] if missing else [],
                "unsupportedClaims": [], "contradictions": [],
                "revisionInstruction": "补充Java的适用场景" if missing else "",
            })

    provider = MissingOnceProvider()
    current_toolbox = toolbox(comparison_segments())

    result = run_structured_evidence_agent(provider, current_toolbox, "Python 与 Java 分别适合什么场景？")

    assert result["report"]["answerable"] is True
    graph = result["agentGraph"]
    assert graph["repairCount"] == graph["retrievalRepairCount"] == 1
    assert graph["repairDetails"]["requirementIds"] == ["java-side"]
    assert graph["revisionCount"] == 0
    assert graph["modelCalls"] == 7
    assert provider.critic_calls == 2
    searches = [item for item in current_toolbox.trace() if item["tool"] == "search_timeline"]
    assert len(searches) == 3  # Two initial slots, only the missing Java slot again.


def test_deictic_audit_rejection_consumes_the_allowance_and_cannot_keep_the_first_verdict() -> None:
    class DeicticOwnerProvider(RefusalProvider):
        verifier_calls = 0

        def _completion(self, messages, tools=None, tool_choice=None):
            if tool_choice["function"]["name"] == "submit_evidence_verification":
                self.verifier_calls += 1
                if self.verifier_calls == 1:
                    return tool_response("submit_evidence_verification", {
                        "requirements": [{
                            "requirementId": "owner", "supported": True, "complete": True,
                            "evidenceIds": ["E001"], "missingInformation": "",
                            "contradictionEvidenceIds": [],
                        }],
                        "overallSufficient": True, "shouldRefuse": False, "refusalReason": "",
                    })
            return super()._completion(messages, tools, tool_choice)

    current_toolbox = toolbox([
        evidence("deictic", "日志系统负责人就是这个人。"),
        evidence("context", "日志系统事故复盘", at=1000),
    ])
    provider = DeicticOwnerProvider()

    result = run_structured_evidence_agent(provider, current_toolbox, "日志系统负责人是谁？")

    assert result["report"]["answerable"] is False
    assert result["agentGraph"]["evidenceVerification"]["overallSufficient"] is False
    assert result["agentGraph"]["repairCount"] == 1
    assert result["agentGraph"]["repairKind"] == "VERIFIER_AUDIT"
    assert result["agentGraph"]["retrievalRepairCount"] == 0
    assert provider.verifier_calls == 2
