from __future__ import annotations

import pytest

from scripts.compare_agent_runs import compare_runs
from scripts.evaluate_real_video import EVALUATOR_VERSION


def saved_run(score: bool | None, *, evidence: str | None = "evidence-v1") -> dict:
    return {
        "datasetId": "regression-set", "runId": "r", "evaluatorVersion": EVALUATOR_VERSION,
        "selectedCasesFingerprint": "same-annotations",
        "results": [{
            "caseId": "q1", "type": "multi_evidence", "state": "COMPLETED", "answerable": True,
            "answerableResult": True, "finalAnswer": "response",
            "answerCheck": {"rulePassed": score, "scoreStatus": "NOT_SCORED" if score is None else "SCORED"},
            "experiment": {"evidenceFingerprint": evidence, "repairCount": 1},
        }],
    }


def test_offline_comparison_requires_matching_versions_to_label_rule_improvement() -> None:
    output = compare_runs(saved_run(False), saved_run(True))
    assert output["comparableForAgentEffect"] is True
    assert output["improvedRuleScreenCount"] == 1
    assert output["before"]["answerRulePassRate"] == 0.0
    assert output["after"]["answerRulePassRate"] == 1.0
    assert output["cases"][0]["beforeExperiment"]["repairCount"] == 1


def test_unscored_is_not_improvement_and_is_excluded_from_denominator() -> None:
    output = compare_runs(saved_run(None), saved_run(True))
    assert output["improvedRuleScreenCount"] == 0
    assert output["before"]["answerRulePassRate"] is None
    assert output["before"]["answerRuleNotScoredCount"] == 1
    assert output["cases"][0]["ruleChange"] == "NOT_SCORED"


@pytest.mark.parametrize("evidence", [None, "different-snapshot"])
def test_missing_or_changed_evidence_cannot_claim_agent_effect(evidence: str | None) -> None:
    output = compare_runs(saved_run(False), saved_run(True, evidence=evidence))
    assert output["comparableForAgentEffect"] is False
    assert output["improvedRuleScreenCount"] == 0
    assert output["warnings"]


def test_legacy_empty_rules_full_score_is_never_used_as_current_baseline() -> None:
    legacy = saved_run(True)
    legacy.pop("evaluatorVersion")
    output = compare_runs(legacy, saved_run(True))
    assert output["before"]["answerRuleScoredCount"] == 0
    assert output["before"]["answerRulePassRate"] is None
    assert output["comparableForAgentEffect"] is False


def test_changed_dataset_or_annotations_rejected() -> None:
    for changes in ({"datasetId": "other"}, {"selectedCasesFingerprint": "changed-gold"}):
        with pytest.raises(ValueError):
            compare_runs(saved_run(False), {**saved_run(True), **changes})
