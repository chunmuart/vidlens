"""Separate offline retrieval, answer and human planning evaluation. No provider calls."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prepare_eval_suite import save, validate
from scripts.evaluate_real_evidence import evaluate_profile, load_snapshots
from scripts.evaluate_real_video import evaluate_status_answer, refusal_decision_metrics
from vidlens.retrieval import RETRIEVER_PROFILES, get_retriever_profile

VERSION = "suite-scorer-v1"


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def ratio(n, d):
    return round(n / d, 4) if d else None


def check_dataset(data, allow_candidates=False):
    validate(data)
    if not data["cases"]:
        raise ValueError("empty dataset")
    pending = [c["caseId"] for c in data["cases"]
               if c.get("reviewStatus") not in ("FIXTURE_DEFINED", "VIDEO_VERIFIED")]
    if pending and not allow_candidates:
        raise ValueError("题目尚未回查原视频；仅调试可加 --allow-candidates，结果不可视为正式指标")


def predictions_by_id(raw, cases):
    rows = raw.get("results", raw.get("predictions", [])) if isinstance(raw, dict) else raw
    if not isinstance(rows, list):
        raise ValueError("predictions must be a JSON list or {results:[...]}")
    known = {c["caseId"] for c in cases}
    result = {}
    for row in rows:
        cid = row.get("caseId")
        if cid not in known or cid in result:
            raise ValueError(f"unknown or duplicate prediction caseId: {cid}")
        graph = (row.get("trace") or {}).get("graph") or {}
        result[cid] = {**row, "plan": row.get("plan", graph.get("evidencePlan", []))}
    return result


def score_answers(cases, predictions):
    details, decisions = [], []
    for case in cases:
        pred = predictions.get(case["caseId"], {})
        answerable = pred.get("answerableResult", pred.get("answerable"))
        answer = pred.get("finalAnswer")
        valid = (isinstance(answerable, bool) and isinstance(answer, str) and bool(answer.strip())
                 and not (pred.get("trace") or {}).get("budgetExceeded")
                 and pred.get("state", "COMPLETED") == "COMPLETED")
        check = evaluate_status_answer(case, {"answerable": answerable, "finalAnswer": answer})
        if not valid:
            check.update(rulePassed=False, scoreStatus="FAILED", scoreReason="MISSING_OR_INVALID_PREDICTION")
        details.append({"caseId": case["caseId"], "type": case["type"], "valid": valid, **check})
        decisions.append({"answerable": case["answerable"], "answerableResult": answerable if valid else None,
                          "finalAnswer": answer, "trace": pred.get("trace")})
    scored = [r for r in details if r["rulePassed"] is not None]
    return {"caseCount": len(cases), "predictionCount": len(predictions),
            "validPredictionCount": sum(r["valid"] for r in details),
            "ruleScoredCount": len(scored), "notScoredCount": len(details) - len(scored),
            "rulePassRate": ratio(sum(r["rulePassed"] is True for r in scored), len(scored)),
            "refusal": refusal_decision_metrics(decisions),
            "byType": {kind: {"count": len(rows), "rulePassRate": ratio(
                sum(r["rulePassed"] is True for r in rows), sum(r["rulePassed"] is not None for r in rows))}
                for kind in sorted({c["type"] for c in cases})
                for rows in [[r for r in details if r["type"] == kind]]},
            "semanticAccuracy": None, "citationSemanticSupportRate": None, "cases": details}


REVIEW_FIELDS = ["caseId", "predictionFingerprint", "question", "goldPlan", "actualPlan", "finalAnswer",
                 "planComplete", "noExtraGoal", "answerCorrect", "citationSupported", "reviewer", "notes"]


def export_review(path, cases, predictions):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        for case in cases:
            pred = predictions.get(case["caseId"], {})
            writer.writerow({"caseId": case["caseId"], "predictionFingerprint": digest(pred), "question": case["question"],
                             "goldPlan": json.dumps(case.get("goldPlan", []), ensure_ascii=False),
                             "actualPlan": json.dumps(pred.get("plan", []), ensure_ascii=False),
                             "finalAnswer": pred.get("finalAnswer", "")})


def score_review(path, cases, predictions):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    known = {c["caseId"]: c for c in cases}
    seen = set()
    labels = {k: [] for k in ("planComplete", "noExtraGoal", "answerCorrect", "citationSupported")}
    for row in rows:
        cid = row["caseId"]
        if cid not in known or cid in seen:
            raise ValueError(f"unknown or duplicate review: {cid}")
        seen.add(cid)
        for field, values in labels.items():
            value = row.get(field, "").strip()
            if value not in ("", "NA", "0", "1"):
                raise ValueError(f"{cid}: {field} must be blank/NA/0/1")
            if value in ("0", "1"):
                if row.get("predictionFingerprint") != digest(predictions.get(cid, {})):
                    raise ValueError("预测已变化，请重新导出审核表，不能复用旧评分")
                if not row.get("reviewer", "").strip() or cid not in predictions:
                    raise ValueError("有评分必须填写审核人且存在实际预测")
                if field == "citationSupported" and not known[cid]["answerable"]:
                    raise ValueError("无答案题的引用支持率请填NA，不能因没有引用得满分")
                values.append(int(value))
    return {field: {"reviewedCount": len(values), "rate": ratio(sum(values), len(values))}
            for field, values in labels.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["retrieval", "answer", "planning"])
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--review", type=Path, help="人工评分CSV，0/1；无评分留空，不适用填NA")
    parser.add_argument("--export-review", type=Path)
    parser.add_argument("--profile", choices=sorted(RETRIEVER_PROFILES), default="lexical")
    parser.add_argument("--allow-candidates", action="store_true")
    args = parser.parse_args()
    data = json.loads(args.dataset.read_text(encoding="utf-8-sig"))
    check_dataset(data, args.allow_candidates)
    metadata = {"scorerVersion": VERSION, "datasetId": data["datasetId"],
                "datasetFingerprint": digest(data), "kind": data.get("kind"),
                "stage": args.stage, "splitCounts": dict(Counter(c["split"] for c in data["cases"])),
                "candidateDebug": args.allow_candidates,
                "claimScope": "受控回归结果不代表真实视频准确率；候选题调试结果不可作为正式指标。"}
    if args.stage == "retrieval":
        if not args.snapshot:
            parser.error("retrieval requires --snapshot")
        snapshots = load_snapshots(args.snapshot)
        metadata["snapshotFingerprint"] = digest(snapshots)
        result = evaluate_profile(data, snapshots, get_retriever_profile(args.profile))
        result["note"] = "只输入本轮question，不输入gold或standaloneQuestion；追问题不代表完整Agent的历史理解能力。时间窗命中不证明语义正确。"
    else:
        if not args.predictions:
            parser.error("answer/planning requires --predictions（真实运行输出，不能填gold）")
        raw = json.loads(args.predictions.read_text(encoding="utf-8-sig"))
        predictions = predictions_by_id(raw, data["cases"])
        metadata["predictionsFingerprint"] = digest(raw)
        result = score_answers(data["cases"], predictions) if args.stage == "answer" else {}
        if args.export_review:
            if args.review and args.review.resolve() == args.export_review.resolve():
                parser.error("导出路径不能覆盖输入评分表")
            export_review(args.export_review, data["cases"], predictions)
        if args.review:
            result["humanReview"] = score_review(args.review, data["cases"], predictions)
            metadata["reviewFingerprint"] = hashlib.sha256(args.review.read_bytes()).hexdigest()
        if args.stage == "planning" and not args.review and not args.export_review:
            parser.error("planning requires --review or --export-review")
    save(args.output, {**metadata, **result})
    print(json.dumps({**metadata, **{k: v for k, v in result.items() if k not in ("cases", "segmentSnapshots")}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
