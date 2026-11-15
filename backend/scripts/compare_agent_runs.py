"""Compare two saved evaluation reports offline; never sends model/API requests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

if __package__:
    from .evaluate_real_video import EVALUATOR_VERSION, aggregate
else:
    from evaluate_real_video import EVALUATOR_VERSION, aggregate


def _experiment(item: dict[str, Any]) -> dict[str, Any]:
    value = item.get("experiment")
    return value if isinstance(value, dict) else {}


def _rule_score(item: dict[str, Any], report: dict[str, Any]) -> bool | None:
    check = item.get("answerCheck") or {}
    if report.get("evaluatorVersion") != EVALUATOR_VERSION or item.get("state") != "COMPLETED":
        return None
    value = check.get("rulePassed")
    return value if isinstance(value, bool) and check.get("scoreStatus") != "NOT_SCORED" else None


def _safe_aggregate(report: dict[str, Any]) -> dict[str, Any]:
    results = []
    for item in report.get("results", []):
        check = dict(item.get("answerCheck") or {})
        check["rulePassed"] = _rule_score(item, report)
        if report.get("evaluatorVersion") != EVALUATOR_VERSION:
            check.update(scoreStatus="NOT_SCORED", scoreReason="EVALUATOR_VERSION_UNVERIFIED")
        results.append({**item, "answerCheck": check})
    return aggregate(results)


def compare_runs(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """Require matching known annotations and expose missing evidence versions.

    A rule-screen improvement is only labeled when both reports use the current
    scorer, the exact selected-case annotation fingerprint matches, and the
    evidence snapshot matches. Even then this is not semantic answer accuracy.
    """
    dataset_before, dataset_after = before.get("datasetId"), after.get("datasetId")
    if dataset_before and dataset_after and dataset_before != dataset_after:
        raise ValueError("两份报告datasetId不同，不能作为同一题集的前后对照")
    annotation_before, annotation_after = before.get("selectedCasesFingerprint"), after.get("selectedCasesFingerprint")
    if annotation_before and annotation_after and annotation_before != annotation_after:
        raise ValueError("两份报告的问题或标注版本不同，请使用同一候选集重新评测")
    warnings = []
    datasets_match = bool(dataset_before and dataset_before == dataset_after)
    annotations_match = bool(annotation_before and annotation_before == annotation_after)
    scorers_match = before.get("evaluatorVersion") == after.get("evaluatorVersion") == EVALUATOR_VERSION
    if not dataset_before or not dataset_after:
        warnings.append("DATASET_ID_MISSING")
    if not annotations_match:
        warnings.append("ANNOTATION_FINGERPRINT_MISSING")
    if not scorers_match:
        warnings.append("EVALUATOR_VERSION_UNVERIFIED_LEGACY_RULE_SCORES_EXCLUDED")

    old = {item["caseId"]: item for item in before.get("results", []) if item.get("caseId")}
    new = {item["caseId"]: item for item in after.get("results", []) if item.get("caseId")}
    shared = sorted(old.keys() & new.keys())
    rows = []
    for case_id in shared:
        left, right = old[case_id], new[case_id]
        left_meta, right_meta = _experiment(left), _experiment(right)
        old_evidence, new_evidence = left_meta.get("evidenceFingerprint"), right_meta.get("evidenceFingerprint")
        evidence_status = "MATCH" if old_evidence and old_evidence == new_evidence else (
            "DIFFERENT" if old_evidence and new_evidence else "UNKNOWN"
        )
        a, b = _rule_score(left, before), _rule_score(right, after)
        comparable = datasets_match and annotations_match and scorers_match and evidence_status == "MATCH"
        change = "NOT_SCORED" if a is None or b is None else (
            "UNCHANGED" if a == b else (
                ("IMPROVED_RULE_SCREEN" if b else "REGRESSED_RULE_SCREEN")
                if comparable else "SCORE_CHANGED_WITH_UNVERIFIED_OR_DIFFERENT_INPUTS"
            )
        )
        rows.append({
            "caseId": case_id,
            "beforeState": left.get("state"), "afterState": right.get("state"),
            "beforeRulePassed": a, "afterRulePassed": b,
            "ruleChange": change,
            "evidenceVersion": evidence_status,
            "beforeExperiment": left_meta, "afterExperiment": right_meta,
            "changedFactors": [
                key for key in ("promptVersion", "model", "retrievalProfile")
                if left_meta.get(key) != right_meta.get(key)
            ],
            "beforeElapsedMs": left.get("elapsedMs"), "afterElapsedMs": right.get("elapsedMs"),
        })
    if any(row["evidenceVersion"] == "UNKNOWN" for row in rows):
        warnings.append("EVIDENCE_FINGERPRINT_MISSING_NO_AGENT_EFFECT_CLAIM")
    if any(row["evidenceVersion"] == "DIFFERENT" for row in rows):
        warnings.append("EVIDENCE_CHANGED_CANNOT_ISOLATE_AGENT_EFFECT")
    if old.keys() != new.keys():
        warnings.append("CASE_RESULT_SETS_DIFFER_PARTIAL_RUNS")
    return {
        "comparisonVersion": "offline-agent-comparison-v1",
        "beforeRunId": before.get("runId"), "afterRunId": after.get("runId"),
        "datasetId": dataset_after or dataset_before,
        "checks": {"sameDataset": datasets_match, "sameAnnotations": annotations_match, "currentScorers": scorers_match},
        "comparableForAgentEffect": bool(
            shared and datasets_match and annotations_match and scorers_match
            and all(row["evidenceVersion"] == "MATCH" for row in rows)
            and old.keys() == new.keys()
        ),
        "sharedCaseCount": len(shared),
        "missingBeforeCaseIds": sorted(new.keys() - old.keys()),
        "missingAfterCaseIds": sorted(old.keys() - new.keys()),
        "improvedRuleScreenCount": sum(row["ruleChange"] == "IMPROVED_RULE_SCREEN" for row in rows),
        "regressedRuleScreenCount": sum(row["ruleChange"] == "REGRESSED_RULE_SCREEN" for row in rows),
        "warnings": warnings,
        "before": _safe_aggregate(before), "after": _safe_aggregate(after),
        "cases": rows,
        "note": "仅读取真实保存报告；规则筛查变化不等于语义正确率。版本缺失单列，未评分排除分母，仍需人工复核回答和引用。",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="离线对照两份已保存Agent评测报告，不调用API")
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = compare_runs(
            json.loads(args.before.read_text(encoding="utf-8")),
            json.loads(args.after.read_text(encoding="utf-8")),
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "sharedCaseCount", "comparableForAgentEffect", "improvedRuleScreenCount", "regressedRuleScreenCount", "warnings",
    )}, ensure_ascii=False))


if __name__ == "__main__":
    main()
