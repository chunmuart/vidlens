# -*- coding: utf-8 -*-
"""Agent 阶段 checkpoint 单测。

覆盖三件事：
1. 开关关闭时不读不写；
2. goal / prompt 版本 / 证据指纹不匹配时不复用；
3. structured-v5 Agent 在有 checkpoint 时跳过已完成的 Planner/Retriever。
"""

import json
import unittest

from vidlens.agent import AgentToolbox
from vidlens.agent_structured import PROMPT_VERSION, run_structured_evidence_agent
from vidlens.checkpoint import AgentCheckpoint, checkpoint_enabled, evidence_fingerprint


GOAL = "课程介绍了哪种二叉树遍历顺序？"

PLAN = {
    "answerMode": "SINGLE",
    "requirements": [{
        "requirementId": "R1",
        "subQuestion": GOAL,
        "retrievalQuery": "二叉树 遍历 顺序",
        "evidenceRole": "DIRECT",
        "completionPolicy": "DIRECT",
        "expectedSources": ["ASR"],
        "required": True,
    }],
}

LEDGER = {
    "question": GOAL,
    "answerMode": "SINGLE",
    "requirements": [{
        **PLAN["requirements"][0],
        "candidateEvidenceIds": ["E001"],
        "rawCandidateCount": 1,
    }],
    "evidence": [{
        "evidenceId": "E001",
        "segmentId": 1,
        "source": "ASR",
        "startMs": 0,
        "endMs": 1000,
        "content": "课程介绍了二叉树的前序遍历。",
        "requirementIds": ["R1"],
    }],
    "statistics": {
        "requirementCount": 1,
        "rawCandidateCount": 1,
        "compactEvidenceCount": 1,
    },
}

SEGMENTS = [{
    "segmentId": 1,
    "source": "ASR",
    "startMs": 0,
    "endMs": 1000,
    "content": "课程介绍了二叉树的前序遍历。",
}]


def tool_response(name: str, arguments: dict) -> dict:
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


class RecordingProvider:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def _completion(self, messages, tools=None, tool_choice=None):
        name = tool_choice["function"]["name"]
        self.calls.append(name)
        if name == "submit_evidence_plan":
            return tool_response(name, PLAN)
        if name == "submit_evidence_verification":
            return tool_response(name, {
                "requirements": [{
                    "requirementId": "R1",
                    "supported": True,
                    "complete": True,
                    "supportLevel": "DIRECT",
                    "evidenceIds": ["E001"],
                    "missingInformation": "",
                    "contradictionEvidenceIds": [],
                }],
                "overallSufficient": True,
                "shouldRefuse": False,
                "refusalReason": "",
            })
        if name == "submit_grounded_report":
            return tool_response(name, {
                "answerable": True,
                "finalAnswer": "课程介绍了二叉树的前序遍历。",
                "title": "视频分析报告",
                "conclusions": ["课程介绍了二叉树的前序遍历。"],
                "evidenceIds": ["E001"],
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
        raise AssertionError(f"unexpected tool: {name}")


def make_toolbox() -> AgentToolbox:
    return AgentToolbox(
        metadata={"mediaId": 1, "filename": "demo.mp4", "status": "COMPLETED"},
        segments=SEGMENTS,
        normalize_report=lambda report: report,
        evaluate_report=lambda report, segments: {
            "structuredValid": True,
            "evidenceSupportRate": 1.0,
            "criticPassed": True,
            "citationCount": len(report.get("evidence", [])),
            "supportedCitationCount": len(report.get("evidence", [])),
        },
    )


def snapshot_with_both_stages() -> dict:
    persisted: list[dict] = []
    checkpoint = AgentCheckpoint(
        enabled=True,
        goal=GOAL,
        prompt_version=PROMPT_VERSION,
        evidence_fingerprint=evidence_fingerprint(SEGMENTS),
        persist=persisted.append,
    )
    checkpoint.save("planner", {
        "plan": {"intent": "EVIDENCE_QA", "tasks": []},
        "evidencePlan": PLAN,
        "plannerMode": "LLM_STRUCTURED",
        "modelCalls": 1,
        "contextChars": {"planner": 12},
    })
    checkpoint.save("retriever", {"evidenceLedger": LEDGER})
    return persisted[-1]


class CheckpointFlagTest(unittest.TestCase):
    def test_flag_defaults_to_off(self) -> None:
        self.assertFalse(checkpoint_enabled({}))
        self.assertFalse(checkpoint_enabled({"CHECKPOINT_ENABLED": "0"}))
        self.assertTrue(checkpoint_enabled({"CHECKPOINT_ENABLED": "1"}))
        self.assertTrue(checkpoint_enabled({"CHECKPOINT_ENABLED": "true"}))
        self.assertTrue(checkpoint_enabled({"CHECKPOINT_ENABLED": "on"}))


class DisabledCheckpointTest(unittest.TestCase):
    def test_disabled_checkpoint_never_reads_or_writes(self) -> None:
        persisted: list[dict] = []
        checkpoint = AgentCheckpoint(
            enabled=False,
            goal=GOAL,
            prompt_version=PROMPT_VERSION,
            evidence_fingerprint=evidence_fingerprint(SEGMENTS),
            persist=persisted.append,
        )
        self.assertIsNone(checkpoint.load("planner"))
        checkpoint.save("planner", {"plan": {}})
        self.assertEqual(persisted, [])
        self.assertFalse(checkpoint.summary()["enabled"])


class CheckpointValidationTest(unittest.TestCase):
    def test_round_trip(self) -> None:
        snapshot = snapshot_with_both_stages()
        restored = AgentCheckpoint(
            enabled=True,
            goal=GOAL,
            prompt_version=PROMPT_VERSION,
            evidence_fingerprint=evidence_fingerprint(SEGMENTS),
            data=snapshot,
        )
        planner = restored.load("planner")
        retriever = restored.load("retriever")
        self.assertEqual(planner["evidencePlan"], PLAN)
        self.assertEqual(retriever["evidenceLedger"]["evidence"][0]["content"], LEDGER["evidence"][0]["content"])
        self.assertEqual(restored.restored_stages, ["planner", "retriever"])

    def test_prompt_version_mismatch_invalidates_all_stages(self) -> None:
        snapshot = snapshot_with_both_stages()
        restored = AgentCheckpoint(
            enabled=True,
            goal=GOAL,
            prompt_version=PROMPT_VERSION + "-next",
            evidence_fingerprint=evidence_fingerprint(SEGMENTS),
            data=snapshot,
        )
        self.assertIsNone(restored.load("planner"))
        self.assertIsNone(restored.load("retriever"))

    def test_goal_mismatch_invalidates_all_stages(self) -> None:
        snapshot = snapshot_with_both_stages()
        restored = AgentCheckpoint(
            enabled=True,
            goal="另一个问题",
            prompt_version=PROMPT_VERSION,
            evidence_fingerprint=evidence_fingerprint(SEGMENTS),
            data=snapshot,
        )
        self.assertIsNone(restored.load("planner"))

    def test_evidence_change_invalidates_only_retriever(self) -> None:
        snapshot = snapshot_with_both_stages()
        restored = AgentCheckpoint(
            enabled=True,
            goal=GOAL,
            prompt_version=PROMPT_VERSION,
            evidence_fingerprint=evidence_fingerprint([
                {**SEGMENTS[0], "content": "完全不同的证据内容"},
            ]),
            data=snapshot,
        )
        self.assertIsNotNone(restored.load("planner"))
        self.assertIsNone(restored.load("retriever"))

    def test_malformed_stage_payload_is_ignored(self) -> None:
        checkpoint = AgentCheckpoint(
            enabled=True,
            goal=GOAL,
            prompt_version=PROMPT_VERSION,
            evidence_fingerprint=evidence_fingerprint(SEGMENTS),
            data={
                "version": 1,
                "goal": GOAL,
                "promptVersion": PROMPT_VERSION,
                "evidenceFingerprint": evidence_fingerprint(SEGMENTS),
                "planner": "not-a-dict",
            },
        )
        self.assertIsNone(checkpoint.load("planner"))

    def test_fingerprint_ignores_database_segment_ids(self) -> None:
        first = evidence_fingerprint([{"segmentId": 1, **SEGMENTS[0]}])
        second = evidence_fingerprint([{"segmentId": 999, **SEGMENTS[0]}])
        self.assertEqual(first, second)


class StructuredAgentResumeTest(unittest.TestCase):
    def test_both_stages_restored_skip_planner_and_retriever(self) -> None:
        restored = AgentCheckpoint(
            enabled=True,
            goal=GOAL,
            prompt_version=PROMPT_VERSION,
            evidence_fingerprint=evidence_fingerprint(SEGMENTS),
            data=snapshot_with_both_stages(),
        )
        provider = RecordingProvider()
        toolbox = make_toolbox()
        result = run_structured_evidence_agent(provider, toolbox, GOAL, checkpoint=restored)

        self.assertEqual(
            provider.calls,
            ["submit_evidence_verification", "submit_grounded_report", "submit_critic_review"],
        )
        tool_names = [item["tool"] for item in toolbox.trace()]
        self.assertNotIn("search_timeline", tool_names)
        self.assertNotIn("get_timeline_overview", tool_names)
        self.assertIn("generate_report", tool_names)
        self.assertTrue(result["accepted"])
        self.assertEqual(
            result["agentGraph"]["checkpoint"]["restoredStages"],
            ["planner", "retriever"],
        )

    def test_planner_only_restored_reruns_retrieval(self) -> None:
        snapshot = snapshot_with_both_stages()
        snapshot.pop("retriever", None)
        restored = AgentCheckpoint(
            enabled=True,
            goal=GOAL,
            prompt_version=PROMPT_VERSION,
            evidence_fingerprint=evidence_fingerprint(SEGMENTS),
            data=snapshot,
        )
        provider = RecordingProvider()
        toolbox = make_toolbox()
        result = run_structured_evidence_agent(provider, toolbox, GOAL, checkpoint=restored)

        self.assertEqual(provider.calls[0], "submit_evidence_verification")
        self.assertNotIn("submit_evidence_plan", provider.calls)
        tool_names = [item["tool"] for item in toolbox.trace()]
        self.assertIn("search_timeline", tool_names)
        self.assertEqual(result["agentGraph"]["checkpoint"]["restoredStages"], ["planner"])

    def test_disabled_checkpoint_runs_normal_planner(self) -> None:
        provider = RecordingProvider()
        toolbox = make_toolbox()
        result = run_structured_evidence_agent(
            provider,
            toolbox,
            GOAL,
            checkpoint=AgentCheckpoint.disabled(),
        )

        self.assertEqual(provider.calls[0], "submit_evidence_plan")
        self.assertFalse(result["agentGraph"]["checkpoint"]["enabled"])


if __name__ == "__main__":
    unittest.main()
