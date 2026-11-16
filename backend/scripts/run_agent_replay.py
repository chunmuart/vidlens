"""Replay the production structured Agent on fixed evidence, without upload/ASR/OCR/queue."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_suite import check_dataset, digest
from scripts.prepare_eval_suite import save
from scripts.evaluate_real_evidence import load_snapshots
from vidlens.agent import AgentToolbox
from vidlens.agent_structured import PROMPT_VERSION, run_structured_evidence_agent
from vidlens.budget import BudgetExceeded, BudgetLimits, BudgetTracker
from vidlens.followup import rewrite_question
from vidlens.retrieval import RETRIEVER_PROFILES, get_retriever_profile


def run_one(case, segments, provider, normalize, evaluate, budget, profile):
    """Only question/history/metadata/segments reach Agent, never reference labels."""
    started = time.perf_counter()
    prediction = {"caseId": case["caseId"], "state": "FAILED", "finalAnswer": "",
                  "answerableResult": None}
    rewrite = {"query": case["question"], "rewritten": False, "needsClarification": False}
    try:
        if case.get("history"):
            rewrite = rewrite_question(provider, case["question"], case["history"], "")
        if rewrite.get("needsClarification"):
            prediction["state"] = "NEEDS_CLARIFICATION"
        else:
            toolbox = AgentToolbox(metadata={"mediaId": case["videoId"], "filename": case["videoId"],
                                             "status": "COMPLETED"}, segments=segments,
                                   normalize_report=normalize, evaluate_report=evaluate,
                                   retriever=profile.create(segments), budget=budget)
            result = run_structured_evidence_agent(provider, toolbox, rewrite["query"])
            report = result["report"]
            prediction.update(state="COMPLETED", finalAnswer=report.get("finalAnswer", ""),
                              answerableResult=report.get("answerable"), report=report,
                              evaluation=result.get("evaluation"),
                              plan=result["agentGraph"].get("evidencePlan"),
                              trace={"graph": result["agentGraph"], "tools": toolbox.trace()})
    except BudgetExceeded as exc:
        prediction.update(state="BUDGET_EXCEEDED", errorType=type(exc).__name__, trace={"budgetExceeded": True})
    except Exception as exc:
        # Provider exception text may contain secrets; keep only the exception class.
        prediction["errorType"] = type(exc).__name__
    prediction.update(rewrite=rewrite, elapsedMs=round((time.perf_counter()-started)*1000, 2),
                      providerUsage=provider.provider_usage_summary(), providerRequests=provider.provider_usage_events())
    return prediction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--max-cases", type=int, default=3)
    parser.add_argument("--max-calls", type=int, default=36)
    parser.add_argument("--max-tokens", type=int, default=50000)
    parser.add_argument("--max-seconds", type=int, default=600)
    parser.add_argument("--profile", choices=sorted(RETRIEVER_PROFILES), default="lexical")
    parser.add_argument("--allow-candidates", action="store_true")
    parser.add_argument("--execute", action="store_true", help="实际请求已配置模型；默认只检查输入")
    args = parser.parse_args()
    if min(args.max_cases, args.max_calls, args.max_tokens, args.max_seconds) <= 0:
        parser.error("各项上限必须为正数")
    dataset = json.loads(args.dataset.read_text(encoding="utf-8-sig"))
    check_dataset(dataset, args.allow_candidates)
    snapshots = load_snapshots(args.snapshot)
    selected = dataset["cases"][:args.max_cases]
    if any(not snapshots.get(c["videoId"]) for c in selected):
        parser.error("选中题目缺少证据快照")
    output = {"datasetId": dataset["datasetId"], "datasetFingerprint": digest(dataset),
              "selectedCasesFingerprint": digest(selected), "snapshotFingerprint": digest(snapshots),
              "scope": "FIXED_EVIDENCE_STRUCTURED_AGENT_REPLAY", "kind": dataset.get("kind"),
              "candidateDebug": args.allow_candidates, "promptVersion": PROMPT_VERSION,
              "profile": args.profile, "executed": args.execute, "selectedCaseCount": len(selected),
              "note": "追问题采用生产改写函数后接结构化Agent，区别于线上追问回答链路；不包含上传/提取/队列。",
              "results": []}
    if args.execute:
        from dotenv import load_dotenv
        load_dotenv(args.env_file, override=False)
        required = ("AI_BASE_URL", "AI_API_KEY", "AI_MODEL")
        if not all(os.getenv(key) for key in required):
            parser.error("缺少AI_BASE_URL / AI_API_KEY / AI_MODEL；不会使用Mock模型代替")
        from vidlens.main import OpenAICompatibleProvider, normalize_analysis_result, evaluate_result
        budget = BudgetTracker(BudgetLimits(max_rounds=args.max_calls, max_tokens=args.max_tokens,
                                           max_duration_ms=args.max_seconds*1000)).start()
        output["model"] = os.environ["AI_MODEL"]
        output["limits"] = budget.snapshot()["limits"]
        save(args.output, output)
        for case in selected:
            try:
                budget.check()
            except BudgetExceeded:
                output["budgetStopped"] = True
                break
            provider = OpenAICompatibleProvider(*(os.environ[k] for k in required))
            provider.budget = budget
            prediction = run_one(case, snapshots[case["videoId"]], provider,
                                 normalize_analysis_result, evaluate_result, budget,
                                 get_retriever_profile(args.profile))
            output["results"].append(prediction)
            output["budget"] = budget.snapshot()
            save(args.output, output)
            if prediction["state"] == "BUDGET_EXCEEDED":
                output["budgetStopped"] = True
                break
    save(args.output, output)
    print(json.dumps({k: v for k, v in output.items() if k != "results"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
