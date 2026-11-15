"""Build a small regression baseline from existing annotations without API calls.

The imported labels have NOT been rechecked against the original videos by this
script. The output marks every case as pending manual video review. This set is
for development/regression; its source videos are not a fresh held-out test set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


EVAL_ROOT = Path(__file__).resolve().parents[1] / "evals"
SOURCES = ("real_video_eval.json", "heldout_retrieval_eval.json", "final_heldout_retrieval_eval.json")


def build_baseline(root: Path = EVAL_ROOT) -> dict[str, Any]:
    datasets = []
    fingerprints = {}
    for filename in SOURCES:
        raw = (root / filename).read_bytes()
        datasets.append((filename, json.loads(raw.decode("utf-8"))))
        fingerprints[filename] = hashlib.sha256(raw).hexdigest()

    # 原始集的15道正例已有显式答案规则，涵盖ASR、OCR及多证据。
    selected = [
        (SOURCES[0], case)
        for case in datasets[0][1]["cases"] if case.get("answerable") is True
    ][:15]
    # 不同视频各选一条负例，避免5条拒答题全部来自同一个视频。
    negative_video_ids: set[str] = set()
    for filename, dataset in datasets:
        for case in dataset["cases"]:
            if case.get("answerable") is False and case["videoId"] not in negative_video_ids:
                selected.append((filename, case))
                negative_video_ids.add(case["videoId"])
                if len(negative_video_ids) == 5:
                    break
        if len(negative_video_ids) == 5:
            break

    cases = []
    for filename, original in selected:
        case = dict(original)
        case["sourceDataset"] = filename
        case["reviewStatus"] = "PENDING_VIDEO_REVIEW"
        if case.get("answerable") is False and not case.get("requiredAnswerGroups"):
            # 仅检查明确拒答的措辞，不把它当人工语义评分。
            case["requiredAnswerGroups"] = [[
                "无法从视频确定", "视频未明确说明", "视频没有说明", "证据不足", "没有足够依据",
            ]]
            case["forbiddenAnswerPatterns"] = []
            case["ruleNote"] = "通用拒答措辞筛查；须结合结构化answerable决策及人工核对。"
        cases.append(case)

    video_ids = {case["videoId"] for case in cases}
    videos = {
        video["videoId"]: video
        for _, dataset in datasets for video in dataset["videos"] if video["videoId"] in video_ids
    }
    return {
        "schemaVersion": "1.0",
        "datasetId": "vidlens-agent-regression-baseline-v1",
        "description": "20题开发/回归候选集：15道可回答、5道应拒答；全部标签导入自既有材料，尚待人工回查原视频。不能作为新视频泛化成绩。",
        "reviewStatus": "PENDING_VIDEO_REVIEW",
        "sourceFingerprints": fingerprints,
        "topK": [1, 3, 8],
        "sourceEnum": ["ASR", "OCR", "ASR+OCR", "NONE"],
        "videos": list(videos.values()),
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="整理20题Agent回归候选集，不调用模型")
    parser.add_argument("--output", type=Path, default=EVAL_ROOT / "agent_baseline_v1.json")
    args = parser.parse_args()
    baseline = build_baseline()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(baseline, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "caseCount": len(baseline["cases"]), "reviewStatus": baseline["reviewStatus"]}))


if __name__ == "__main__":
    main()
