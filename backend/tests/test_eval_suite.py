from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.prepare_eval_suite import ROOT, controlled, validate, real_candidates
from scripts.evaluate_suite import (check_dataset, predictions_by_id, score_answers,
                                    export_review, score_review)
from scripts.evaluate_real_evidence import evaluate_case
from scripts import run_agent_replay
from vidlens.budget import BudgetLimits, BudgetTracker
from vidlens.retrieval import LEXICAL_PROFILE


@pytest.fixture
def suite():
    return controlled(json.loads((ROOT / "evals/scenario_seeds_v1.json").read_text(encoding="utf-8")))


def test_fixture_split_and_grounding(suite):
    data, snapshots = suite
    stats = validate(data)
    assert stats["cases"] == 300
    assert stats["bySplit"] == {"development": 198, "test": 102}
    assert set(stats["byType"].values()) == {50}
    assert all(len(segments) > 8 for segments in snapshots.values())
    groups = {split: {c["sourceGroupId"] for c in data["cases"] if c["split"] == split}
              for split in ("development", "test")}
    assert not groups["development"] & groups["test"]
    contents = {split: {s["content"] for vid in groups[split] for s in snapshots[vid]
                       if "课程安排" not in s["content"]} for split in groups}
    assert not contents["development"] & contents["test"]
    for case in data["cases"]:
        for gold in case["goldEvidence"]:
            matching = [s for s in snapshots[case["videoId"]]
                        if s["startMs"] == gold["startMs"] and s["source"] in gold["sources"]]
            assert len(matching) == 1
    committed = json.loads((ROOT / "evals/suite_v2/controlled_300.json").read_text(encoding="utf-8"))
    assert data == committed


def test_real_candidates_are_not_silently_verified():
    data = real_candidates(ROOT / "evals")
    assert len(data["cases"]) == 58
    assert len(data["videos"]) == 8
    assert all(c["reviewStatus"] == "PENDING_VIDEO_REVIEW" for c in data["cases"])
    with pytest.raises(ValueError, match="尚未"):
        check_dataset(data)
    check_dataset(data, allow_candidates=True)


def test_leakage_rejected(suite):
    data, _ = suite
    data["cases"][0]["split"] = "test"
    with pytest.raises(ValueError, match="leaks"):
        validate(data)


def test_predictions_unknown_and_duplicates(suite):
    cases = suite[0]["cases"]
    with pytest.raises(ValueError, match="unknown"):
        predictions_by_id([{"caseId": "unknown"}], cases)
    row = {"caseId": cases[0]["caseId"]}
    with pytest.raises(ValueError):
        predictions_by_id([row, row], cases)


def test_missing_and_invalid_predictions_cannot_pass(suite):
    cases = suite[0]["cases"][:6]
    negative = cases[-1]
    result = score_answers(cases, {negative["caseId"]: {
        "answerable": "false", "finalAnswer": negative["goldAnswer"]}})
    assert result["rulePassRate"] == 0
    assert result["validPredictionCount"] == 0
    assert result["semanticAccuracy"] is None
    assert result["refusal"]["abstentionRecall"] == 0


def test_wrong_answerability_and_budget_do_not_pass(suite):
    case = suite[0]["cases"][0]
    row = {"answerable": False, "finalAnswer": case["goldAnswer"]}
    assert score_answers([case], {case["caseId"]: row})["rulePassRate"] == 0
    row.update(answerable=True, trace={"budgetExceeded": True})
    assert score_answers([case], {case["caseId"]: row})["validPredictionCount"] == 0


def edit_review(path, **updates):
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields, rows = reader.fieldnames, list(reader)
    rows[0].update(updates)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_review_empty_not_perfect_and_stale_rejected(suite, tmp_path):
    case = suite[0]["cases"][0]
    predictions = {case["caseId"]: {"finalAnswer": "实际模型输出"}}
    path = tmp_path / "review.csv"
    export_review(path, [case], predictions)
    assert score_review(path, [case], predictions)["planComplete"] == {"reviewedCount": 0, "rate": None}
    edit_review(path, reviewer="reviewer-a", planComplete="1")
    assert score_review(path, [case], predictions)["planComplete"]["rate"] == 1
    predictions[case["caseId"]]["finalAnswer"] = "另一次输出"
    with pytest.raises(ValueError, match="预测已变化"):
        score_review(path, [case], predictions)


def test_refusal_no_citation_not_semantic_full_marks(suite, tmp_path):
    case = suite[0]["cases"][5]
    predictions = {case["caseId"]: {"finalAnswer": "证据不足"}}
    path = tmp_path / "review.csv"
    export_review(path, [case], predictions)
    edit_review(path, reviewer="reviewer-a", citationSupported="1")
    with pytest.raises(ValueError, match="NA"):
        score_review(path, [case], predictions)


def test_retrieval_counts_source_and_multiple_evidence(suite):
    data, snapshots = suite
    case = data["cases"][2]
    segments = snapshots[case["videoId"]]
    # Correct ASR alone covers half, not all of this two-evidence question.
    correct_asr = next(s for s in segments if s["segmentId"] == case["videoId"] + "-s1")
    retriever = SimpleNamespace(search=lambda *a, **kw: {"matches": [correct_asr]})
    result = evaluate_case(case, segments, [1, 3, 8], retriever=retriever)
    assert result["recallAt1"] == .5
    assert not result["completeHitAt1"]


def test_agent_replay_never_receives_gold(suite, monkeypatch):
    data, snapshots = suite
    case = data["cases"][0]
    captured = {}
    def fake_agent(provider, toolbox, goal):
        captured.update(goal=goal, metadata=toolbox.metadata)
        return {"report": {"answerable": False, "finalAnswer": "视频证据不足"},
                "agentGraph": {"evidencePlan": []}, "evaluation": {}}
    monkeypatch.setattr(run_agent_replay, "run_structured_evidence_agent", fake_agent)
    provider = SimpleNamespace(provider_usage_summary=lambda: {}, provider_usage_events=lambda: [])
    result = run_agent_replay.run_one(case, snapshots[case["videoId"]], provider, lambda x: x,
                                     lambda *a: {}, BudgetTracker(BudgetLimits()).start(), LEXICAL_PROFILE)
    assert captured["goal"] == case["question"]
    assert "goldAnswer" not in captured["metadata"]
    assert result["state"] == "COMPLETED"
